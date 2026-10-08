"""Deterministic, resumable sample loading for the 49D recipe."""

from __future__ import annotations

import bisect
import hashlib
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def collate_samples(samples):
    from cosmos_framework.data.vfm.joint_dataloader import (
        IterativeJointDataLoader,
        custom_collate_fn,
    )

    targets = [tuple(s["target_hw"].tolist()) if "target_hw" in s else None for s in samples]
    if any(t is not None for t in targets):
        if None in targets or len(set(targets)) != 1:
            raise ValueError(f"ViGAR microbatch mixes image resolution buckets: {targets}")
        shapes = []
        for sample in samples:
            items = sample["video"]
            items = items if isinstance(items, list) else [items]
            spatial = [tuple(item.shape[-2:]) for item in items]
            if len(set(spatial)) != 1:
                raise ValueError("Goal and rollout must share spatial shape")
            shapes.append(spatial[0])
        if len(set(shapes)) != 1:
            raise ValueError("ViGAR microbatch mixes composed canvas shapes")
    batch = {}
    for key, values in custom_collate_fn(samples).items():
        if key in IterativeJointDataLoader._MULTI_ITEM_KEYS:
            batch[key] = [v if isinstance(v, list) else [v] for v in values]
        elif isinstance(values, torch.Tensor):
            batch[key] = [values[i : i + 1] for i in range(len(samples))]
        else:
            batch[key] = values
    return batch


def fixed_indices(dataset, count, seed):
    """Cover tasks and legal start/interior/end windows, with fixed episode choices."""
    rng = np.random.default_rng(seed)
    by_task = {}
    offset = 0
    for ep in dataset.episodes:
        by_task.setdefault(ep["name"], []).append((offset, ep))
        offset += ep["weighted"]
    names = sorted(by_task)
    result = []
    for i in range(count):
        name = names[i % len(names)]
        offset, ep = by_task[name][int(rng.integers(len(by_task[name])))]
        position = (0, ep["weighted"] // 2, ep["weighted"] - 1)[i % 3]
        result.append(offset + position)
    return result


class WindowSampler(Sampler):
    """Cursor counts consumed samples, so worker prefetch cannot advance resume."""

    def __init__(self, size, rank, world_size, seed=42, indices=None, *, include_position=False):
        self.size, self.rank, self.world_size = size, int(rank), int(world_size)
        self.seed, self.cursor = seed, 0
        self.indices = indices
        self.include_position = include_position

    def __iter__(self):
        cursor = self.cursor
        cached_epoch, order = None, None
        while True:
            global_position = cursor * self.world_size + self.rank
            if self.indices is not None:
                index = self.indices[global_position % len(self.indices)]
            else:
                epoch, position = divmod(global_position, self.size)
                if epoch != cached_epoch:
                    order = np.random.default_rng(self.seed + epoch).permutation(self.size)
                    cached_epoch = epoch
                index = int(order[position])
            yield (index, global_position) if self.include_position else index
            cursor += 1

    def state_dict(self):
        fingerprint = hashlib.sha256(repr(self.indices).encode()).hexdigest()
        return dict(
            cursor=self.cursor,
            size=self.size,
            rank=self.rank,
            world_size=self.world_size,
            seed=self.seed,
            indices_sha256=fingerprint,
        )

    def load_state_dict(self, state):
        current = self.state_dict()
        if any(state[k] != v for k, v in current.items() if k != "cursor"):
            raise ValueError("Sampler population/topology/order changed on resume")
        self.cursor = state["cursor"]


class BucketWindowSampler(WindowSampler):
    """One bucket per distributed optimizer step, with replacement within buckets.

    Each step is reconstructed from its ID, so consumed-cursor resume never
    replays old steps or depends on worker prefetch. Window weights are the
    existing weighted episode spans, not episode counts or raw frame counts.
    """

    def __init__(
        self,
        dataset,
        rank,
        world_size,
        seed,
        indices=None,
        *,
        batch_size=1,
        accumulation_steps=1,
        include_position=False,
    ):
        super().__init__(len(dataset), rank, world_size, seed, indices, include_position=include_position)
        if any(type(v) is not int or v < 1 for v in (batch_size, accumulation_steps, world_size)):
            raise ValueError("Bucket sampler batch size, accumulation and world size must be positive integers")
        if not 0 <= rank < world_size:
            raise ValueError("Invalid bucket sampler rank")
        self.local_step_size = batch_size * accumulation_steps
        self.pools = {}
        if indices is not None:
            if not indices:
                raise ValueError("Fixed bucket population must not be empty")
            for index in indices:
                ep, _ = dataset.locate(index)
                self.pools.setdefault(ep["bucket_id"], []).append((index, 1))
        else:
            start = 0
            for ep in dataset.episodes:
                self.pools.setdefault(ep["bucket_id"], []).append((start, ep["weighted"]))
                start += ep["weighted"]
        self.bucket_ids = sorted(self.pools)
        self.cumulative = {}
        self.bucket_stops = []
        total = 0
        for key in self.bucket_ids:
            self.cumulative[key] = np.cumsum([length for _, length in self.pools[key]], dtype=np.int64)
            total += int(self.cumulative[key][-1])
            self.bucket_stops.append(total)
        self.contract = dict(
            **dataset.resolution_bucket_record(),
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            sampling="weighted_windows_with_replacement",
            seed_version=2,
        )

    def __iter__(self):
        cursor, cached_step = self.cursor, None
        while True:
            step, offset = divmod(cursor, self.local_step_size)
            if step != cached_step:
                seed = int.from_bytes(
                    hashlib.sha256(f"vigar-buckets-v2:{self.seed}:{step}".encode()).digest(), "big"
                )
                rng = np.random.default_rng(seed)
                bucket = self.bucket_ids[
                    bisect.bisect_right(self.bucket_stops, int(rng.integers(self.bucket_stops[-1])))
                ]
                cumulative = self.cumulative[bucket]
                draws = rng.integers(int(cumulative[-1]), size=self.local_step_size * self.world_size)
                cached_step = step
            position = cursor * self.world_size + self.rank
            draw = int(draws[offset * self.world_size + self.rank])
            span = bisect.bisect_right(cumulative, draw)
            previous = int(cumulative[span - 1]) if span else 0
            index = self.pools[bucket][span][0] + draw - previous
            yield (index, position) if self.include_position else index
            cursor += 1

    def state_dict(self):
        return {**super().state_dict(), "resolution_buckets": self.contract}


class CaptionDropoutDataset(Dataset):
    """Seed native caption dropout by stream occurrence, independent of worker RNG.

    The consumed sampler cursor reproduces this seed after resume even when
    workers prefetched unused samples. Repeated windows get fresh decisions.
    Restoring Python RNG also keeps data dropout separate from model noise.
    Randomized goal datasets also receive the occurrence explicitly and draw
    from their own RNG namespace, including when caption dropout is zero.
    """

    def __init__(self, dataset, seed, *, include_goal_position=False):
        self.dataset, self.seed = dataset, seed
        self.include_goal_position = include_goal_position

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        index, position = key
        seed = hashlib.sha256(f"vigar-caption-v2:{self.seed}:{position}".encode()).digest()
        state = random.getstate()
        try:
            random.seed(seed)
            return self.dataset[key if self.include_goal_position else index]
        finally:
            random.setstate(state)


class StatefulWindowLoader:
    def __init__(
        self,
        dataset,
        rank=0,
        world_size=1,
        seed=42,
        batch_size=1,
        num_workers=0,
        fixed_sample_count=None,
        accumulation_steps=1,
        prefetch_factor=2,
    ):
        self.dataset, self.batch_size = dataset, batch_size
        if num_workers and (type(prefetch_factor) is not int or prefetch_factor < 1):
            raise ValueError("prefetch_factor must be a positive integer with workers enabled")
        rate = getattr(dataset, "cfg_dropout_rate", 0.0)
        self.caption_dropout = dict(rate=rate, seed_version=2) if rate else None
        raw = getattr(dataset, "_dataset", None)
        self.goal_sampling = raw.goal_sampling_record() if getattr(raw, "random_goal_sampling", False) else None
        text_record = getattr(raw, "text_conditioning_record", None)
        self.text_conditioning = text_record() if text_record is not None else None
        image_record = getattr(raw, "image_composition_record", None)
        self.image_composition = image_record() if image_record is not None else None
        self.tail_windows = None
        if getattr(raw, "include_tail_windows", False):
            self.tail_windows = dict(
                version=1,
                selection_sha256=hashlib.sha256(
                    json.dumps(raw.selection_record(), sort_keys=True).encode()
                ).hexdigest(),
            )
        indices = None if fixed_sample_count is None else fixed_indices(dataset._dataset, fixed_sample_count, seed)
        use_occurrence = bool(rate or self.goal_sampling is not None)
        self.sampler = (
            BucketWindowSampler(
                raw,
                rank,
                world_size,
                seed,
                indices,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                include_position=use_occurrence,
            )
            if getattr(raw, "img_size_buckets", ())
            else WindowSampler(len(dataset), rank, world_size, seed, indices, include_position=use_occurrence)
        )
        self.loader = DataLoader(
            CaptionDropoutDataset(dataset, seed, include_goal_position=self.goal_sampling is not None)
            if use_occurrence
            else dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=collate_samples,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            **({"prefetch_factor": prefetch_factor} if num_workers else {}),
            generator=torch.Generator().manual_seed(seed),
        )

    def __iter__(self):
        for batch in self.loader:
            self.sampler.cursor += self.batch_size
            yield batch

    def state_dict(self):
        state = self.sampler.state_dict()
        if self.caption_dropout is not None:
            state["caption_dropout"] = self.caption_dropout
        if self.tail_windows is not None:
            state["tail_windows"] = self.tail_windows
        if self.goal_sampling is not None:
            state["goal_sampling"] = self.goal_sampling
        if self.text_conditioning is not None:
            state["text_conditioning"] = self.text_conditioning
        if self.image_composition is not None:
            state["image_composition"] = self.image_composition
        return state

    def load_state_dict(self, state):
        if state.get("text_conditioning") != self.text_conditioning:
            raise ValueError(
                "Text conditioning changed on resume; use model.resume_training=false "
                "and a new experiment directory for weights-only initialization"
            )
        if state.get("resolution_buckets") != getattr(self.sampler, "contract", None):
            raise ValueError(
                "Resolution bucket sampling changed on resume; use model.resume_training=false in a new experiment"
            )
        if state.get("image_composition") != self.image_composition:
            raise ValueError(
                "Image composition changed on resume; use model.resume_training=false "
                "and a new experiment directory for weights-only initialization"
            )
        if state.get("goal_sampling") != self.goal_sampling:
            raise ValueError(
                "Goal sampling changed on resume; use model.resume_training=false "
                "and a new experiment directory for weights-only initialization"
            )
        if state.get("tail_windows") != self.tail_windows:
            raise ValueError(
                "Tail-window training contract changed on resume; use model.resume_training=false "
                "and a new experiment directory for weights-only initialization"
            )
        if state.get("caption_dropout") != self.caption_dropout:
            raise ValueError(
                "Caption dropout changed on resume; keep the saved rate for exact resume or use "
                "model.resume_training=false for weights-only initialization"
            )
        self.sampler.load_state_dict(state)


class EvaluationWindows(Dataset):
    """Shard one seeded test-window shuffle across ranks, without replacement."""

    def __init__(self, dataset, count, rank, world_size, seed):
        if count % world_size:
            raise ValueError("eval_count must be divisible by world size for native FSDP lockstep")
        self.dataset = dataset
        size = len(dataset)
        if count and size < world_size:
            raise ValueError("Evaluation needs at least one distinct window per rank; use fewer ranks")
        # Every rank must perform the same number of FSDP forwards. Stop at
        # dataset exhaustion, dropping at most world_size-1 windows rather
        # than padding the shuffle with duplicate samples.
        count = min(count, size // world_size * world_size)
        self.indices = np.random.default_rng(seed).permutation(size)[:count].tolist()
        self.sample_ids = list(range(rank, count, world_size))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, position):
        sample_id = self.sample_ids[position]
        index = self.indices[sample_id]
        raw = self.dataset._dataset
        physical = raw.physical_sample(index)
        decoded = raw[index]
        sample = self.dataset.prepare_sample(decoded, physical["robot_type"])
        return dict(
            sample_id=sample_id,
            index=index,
            physical={k: v for k, v in physical.items() if k not in {"normalizer", "episode"}},
            sample=sample,
            target_video=decoded["video"],
            pixel_mask=decoded.get("video_pixel_mask"),
            camera_boxes=decoded.get("camera_boxes", {}),
            caption=decoded["ai_caption"],
        )


def identity_sample(sample):
    return sample


def evaluation_loader(
    dataset, count, rank, world_size, *, seed=20260824, num_workers=0, pin_memory=False, prefetch_factor=1
):
    windows = EvaluationWindows(dataset, count, rank, world_size, seed)
    return DataLoader(
        windows,
        batch_size=None,
        collate_fn=identity_sample,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(seed),
        **({"prefetch_factor": prefetch_factor, "multiprocessing_context": "spawn"} if num_workers else {}),
    )
