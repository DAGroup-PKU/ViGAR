"""Read 0824 LeRobot v3 manifests without changing stored absolute actions.

Geometry and normalization use the recipe-local 0824 helpers.
The physical reader is independent of Cosmos tokenization and GPU runtimes.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import Dataset

from recipes.GoalWAM.data.action_layout import resolve_action_layout
from recipes.GoalWAM.data.normalization import Normalizer, parse_bounds_99_clip_limit
from recipes.GoalWAM.data.relative_action import (
    check_anchor_head_quaternion,
    propagate_relative_action_validity,
    propagate_state_arm_eef_validity,
    to_relative_action,
    transform_state_arm_eef_coordinate,
    zero_chassis_state,
)

from .goal_sampling import GOAL_SAMPLING_VERSION, episode_goal_timing, resolve_goal_sampling, select_goal
from .images import (
    CAMERA_KEYS,
    PreparedVideo,
    camera_layout,
    compose_goal_image,
    validate_goal_image_composition,
)
from .parquet_cache import ParquetEpisodeCache
from .resolution import assign_bucket, episode_resolution, validate_buckets
from .text_conditioning import segment_text_intervals, select_text, validate_text_conditioning


CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
ROBOT_DOMAINS = {
    "robotwin_aloha_agilex": 16,
    "astribot_s1": 22,
    "agibot": 21,
    "cobot": 23,
}
ACTION_LAYOUT = {
    "joint": [0, 23],
    "eef": [26, 49],
    "left_arm": [0, 7],
    "right_arm": [8, 15],
    "gripper_dims": [7, 15, 33, 41],
    "chassis_xy": [23, 25],
    "chassis_angle": 25,
    "left_eef_pos": [26, 29],
    "right_eef_pos": [34, 37],
}
ATOMIC_GROUPS = (
    slice(23, 26),
    slice(26, 29),
    slice(29, 33),
    slice(34, 37),
    slice(37, 41),
    slice(42, 45),
    slice(45, 49),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_paths(manifest):
    """Validate named manifest paths; retain single paths for historical profiles."""
    if isinstance(manifest, (str, Path)) and str(manifest).strip():
        return {None: Path(manifest)}
    if not isinstance(manifest, Mapping) or not manifest:
        raise ValueError("Manifest paths must be a path or a nonempty source-name -> path mapping")
    paths = {}
    for source, path in manifest.items():
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Manifest mapping requires nonempty string source names")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"Manifest path for {source!r} must be a nonempty string")
        paths[source] = Path(path)
    return paths


def load_manifests(manifest):
    """Resolve source/dataset names, without merging collisions silently."""
    paths = manifest_paths(manifest)
    entries = {}
    for source, path in paths.items():
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"{path}: manifest must be a nonempty mapping")
        for name, entry in raw.items():
            if not isinstance(name, str) or not name or not isinstance(entry, dict):
                raise ValueError(f"{path}: expected named dataset-entry mappings")
            name = f"{source}/{name}" if source is not None else name
            if name in entries:
                raise ValueError(f"Duplicate dataset name across manifests: {name}")
            entries[name] = entry
    if None in paths:
        return str(paths[None].resolve()), digest(paths[None]), entries
    return (
        {source: str(path.resolve()) for source, path in paths.items()},
        {source: digest(path) for source, path in paths.items()},
        entries,
    )


def validate_mask(value, *, context):
    mask = torch.as_tensor(value)
    if mask.dtype != torch.bool or mask.shape[-1:] != (49,):
        raise ValueError(f"{context}: expected boolean mask ending in 49, got {mask.shape}/{mask.dtype}")
    for group in ATOMIC_GROUPS:
        if (mask[..., group].any(-1) != mask[..., group].all(-1)).any():
            raise ValueError(f"{context}: partially valid atomic group {group}")
    return mask


def validate_values(values, valid, *, context):
    if values.shape != valid.shape or values.shape[-1] != 49:
        raise ValueError(f"{context}: mismatched value/validity shapes")
    if not torch.isfinite(values[valid]).all():
        raise ValueError(f"{context}: nonfinite valid values")
    for group in (slice(29, 33), slice(37, 41), slice(45, 49)):
        q = values[..., group][valid[..., group].all(-1)]
        if q.numel() and not torch.allclose(q.norm(dim=-1), torch.ones_like(q[..., 0]), atol=1e-3, rtol=0):
            raise ValueError(f"{context}: nonunit quaternion {group}")


class ContractNormalizer:
    """Keep state/action scales separate and retain the original asset metadata."""

    def __init__(self, path, norm_type="meanstd", state_arm_eef_coordinate="head_camera"):
        self.path = str(Path(path).resolve())
        self.sha256 = digest(path)
        data = json.loads(Path(path).read_text())
        self.metadata = data.get("metadata", {})
        if self.metadata.get("state_arm_eef_coordinate") != state_arm_eef_coordinate:
            raise ValueError(f"{path}: statistics must describe {state_arm_eef_coordinate} state")
        stats = data["norm_stats"]
        if norm_type == "meanstd":
            required = ("mean", "std")
        elif norm_type == "bounds_999_clip":
            required = ("q001", "q999")
        elif norm_type == "bounds_99_woclip" or parse_bounds_99_clip_limit(norm_type) is not None:
            required = ("q01", "q99")
        else:
            raise ValueError(f"Unsupported GoalWAM normalization: {norm_type}")
        for key in ("observation.state", "action"):
            for field in required:
                value = np.asarray(stats[key][field])
                if value.shape != (49,) or not np.isfinite(value).all():
                    raise ValueError(f"{path}: invalid {key}/{field}")
        self.norm_type = norm_type
        self.normalizer = Normalizer(stats, norm_type={"observation.state": norm_type, "action": norm_type})

    def normalize(self, value, key, mask):
        value = value.masked_fill(~mask, 0)
        return self.normalizer.normalize({key: value})[key].masked_fill(~mask, 0)

    def normalize_action(self, value):
        return self.normalizer.normalize({"action": value})["action"]

    def denormalize_action(self, value):
        # Physical commands retain FP32 precision even when the sampler uses BF16.
        return self.normalizer.unnormalize({"action": value.float()})["action"]


class FrameReader:
    """PTS-based exact PyAV decoding of shared LeRobot video parts."""

    def __init__(self, path, fps):
        import av

        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        if self.stream.average_rate is None or abs(float(self.stream.average_rate) - fps) > 1e-3:
            self.container.close()
            raise ValueError(f"{path}: encoded video FPS differs from dataset metadata")
        self.stream.thread_count = 1
        self.fps = fps
        self.cache = OrderedDict()

    def read(self, timestamps):
        wanted = [round(float(t) * self.fps) for t in timestamps]
        missing = sorted(set(wanted) - self.cache.keys())
        # A distant goal must not force decoding the entire episode between
        # the short rollout and that goal. Keep nearby requests in one pass;
        # use the same exact PTS/keyframe seek for gaps exceeding one second
        # (at least 32 frames, above the ordinary 4/12-frame rollout spacing).
        groups = []
        for index in missing:
            if not groups or index - groups[-1][-1] > max(32, round(self.fps)):
                groups.append([])
            groups[-1].append(index)
        for group in groups:
            self.container.seek(
                int((group[0] / self.fps) / self.stream.time_base),
                stream=self.stream,
                backward=True,
                any_frame=False,
            )
            remaining = set(group)
            for frame in self.container.decode(self.stream):
                if frame.pts is None:
                    raise ValueError("Video frame has no PTS")
                timestamp = float(frame.pts * self.stream.time_base)
                index = round(timestamp * self.fps)
                if index in remaining:
                    if abs(timestamp - index / self.fps) > 0.51 / self.fps:
                        raise ValueError("Video timestamp exceeds nearest-frame tolerance")
                    self.cache[index] = frame.to_ndarray(format="rgb24")
                    remaining.remove(index)
                if not remaining or index > group[-1]:
                    break
            if remaining:
                raise ValueError(f"Missing video frames {sorted(remaining)}")
        result = np.stack([self.cache[i] for i in wanted])
        for i in wanted:
            self.cache.move_to_end(i)
        while len(self.cache) > 96:
            self.cache.popitem(last=False)
        return result

    def close(self):
        self.container.close()


class LeRobot0824Dataset(Dataset):
    """One current state, 48 commands, and a separate recorded visual goal.

    Optional tail windows retain every anchor, mask unavailable actions and use
    only complete real Wan video blocks (1 + 4*n frames), never repeated futures.
    Manifests select complete populations by default. Episode holdout is an
    explicit compatibility option for the historical migration reference only.
    """

    def __init__(
        self,
        manifest,
        norm_stat_files,
        *,
        split="all",
        training=None,
        seed=42,
        eval_episodes=5,
        horizon=48,
        video_stride=4,
        resolution="384x320",
        norm_type="meanstd",
        supervise_head_eef=False,
        supervise_arm_head_torso=True,
        dataset_names=None,
        domain_ids=None,
        assume_native_astribot_basis=False,
        img_size=None,
        enable_cameras=None,
        state_arm_eef_coordinate="head_camera",
        action_layout=None,
        include_robot_type_text_context=False,
        text_conditioning="episode",
        include_tail_windows=False,
        goal_sampling=None,
        goal_image_composition="multi_view",
        img_size_buckets=None,
        parquet_cache_dir=None,
    ):
        if split not in {"train", "eval", "all"}:
            raise ValueError(f"Unknown split {split}")
        if horizon != 48 or video_stride != 4:
            raise ValueError("GoalWAM 0824 contract requires horizon=48, video_stride=4")
        self.manifest, self.manifest_sha256, raw = load_manifests(manifest)
        self.split, self.seed = split, seed
        self.training = split == "train" if training is None else training
        self.goal_sampling = resolve_goal_sampling(goal_sampling)
        self.random_goal_sampling = self.training and not self.goal_sampling.terminal_only
        self.horizon, self.video_stride = horizon, video_stride
        if type(include_tail_windows) is not bool:
            raise ValueError("include_tail_windows must be boolean")
        self.include_tail_windows = include_tail_windows
        self.img_size = None if img_size is None else tuple(img_size)
        self.img_size_buckets = validate_buckets(img_size_buckets, self.img_size)
        self.enable_cameras = tuple(enable_cameras or ("head", "left", "right"))
        if (
            not self.enable_cameras
            or set(self.enable_cameras) - CAMERA_KEYS.keys()
            or len(set(self.enable_cameras)) != len(self.enable_cameras)
        ):
            raise ValueError("enable_cameras must select unique torso/head/left/right cameras")
        self.cameras = tuple(CAMERA_KEYS[name] for name in CAMERA_KEYS if name in self.enable_cameras)
        self.goal_image_composition = validate_goal_image_composition(goal_image_composition, self.enable_cameras)
        if self.img_size is None:
            self.height, self.width = map(int, resolution.split("x"))
            if self.height % 32 or self.width % 32 or self.height % 3:
                raise ValueError("Legacy canvas requires edges divisible by 32 and height divisible by 3")
            if self.enable_cameras != ("head", "left", "right"):
                raise ValueError("Set img_size to use configurable cameras with letterboxing")
        else:
            if len(self.img_size) != 2 or any(not isinstance(v, int) or v < 2 or v % 2 for v in self.img_size):
                raise ValueError("img_size must contain even positive HEIGHT WIDTH values")
            (self.height, self.width), _ = camera_layout(self.img_size, self.enable_cameras)
        self.resolution = f"{self.height}x{self.width}"
        self.supervise_head_eef = supervise_head_eef
        self.supervise_arm_head_torso = supervise_arm_head_torso
        self.layout = resolve_action_layout(action_layout or ACTION_LAYOUT)
        if self.layout != resolve_action_layout(ACTION_LAYOUT):
            raise ValueError("GoalWAM consumes the canonical 0824 49-D action_layout")
        if state_arm_eef_coordinate not in {"head_camera", "base"}:
            raise ValueError("state_arm_eef_coordinate must be base or head_camera")
        self.state_arm_eef_coordinate = state_arm_eef_coordinate
        self.include_robot_type_text_context = include_robot_type_text_context
        self.text_conditioning = validate_text_conditioning(text_conditioning)
        self.domain_ids = {**ROBOT_DOMAINS, **(domain_ids or {})}
        self.entries, self.episodes, self.stops = {}, [], []
        self.exclusions = []
        self.assume_native_astribot_basis = assume_native_astribot_basis
        self.metadata_assumptions = []
        self._rows, self._readers = OrderedDict(), OrderedDict()
        self._parquet_cache = ParquetEpisodeCache(parquet_cache_dir) if parquet_cache_dir else None
        self._camera_warnings = OrderedDict()
        self.normalizers = {
            k: ContractNormalizer(v, norm_type, state_arm_eef_coordinate) for k, v in norm_stat_files.items()
        }
        for name, entry in raw.items():
            if dataset_names is not None and name not in dataset_names:
                continue
            self._add_entry(name, entry, split, eval_episodes)
        if not self.stops:
            raise ValueError("No legal GoalWAM windows in selection")

    def _add_entry(self, name, entry, split, eval_episodes):
        root = Path(entry["data_path"])
        info = json.loads((root / "meta/info.json").read_text())
        robot = entry["robot_type"]
        if entry["dataset_type"] != "lerobot" or info["codebase_version"] != "v3.0":
            raise ValueError(f"{name}: only LeRobot v3 is supported")
        if robot not in self.normalizers or robot not in self.domain_ids:
            raise ValueError(f"{name}: missing normalizer or domain ID for {robot}")
        if not 0 <= self.domain_ids[robot] < 32:
            raise ValueError(f"{name}: domain ID exceeds native embedding table")
        basis = info.get("eef_gripper_frame")
        if basis is None and robot == "astribot_s1" and self.assume_native_astribot_basis:
            self.metadata_assumptions.append(dict(dataset=name, field="eef_gripper_frame", assumed="astribot_s1"))
            warnings.warn(
                f"{name}: missing EEF-basis marker; using explicitly configured native Astribot S1 basis", stacklevel=2
            )
        elif basis != "astribot_s1":
            raise ValueError(f"{name}: missing or incompatible EEF basis")
        if info.get("video_layout", "uniform") not in {
            "uniform",
            "per_episode_variable",
        }:
            raise ValueError(f"{name}: invalid video layout")
        for key in ("observation.state", "action"):
            if info["features"][key]["shape"] != [49]:
                raise ValueError(f"{name}: {key} must have 49 dimensions")
        available_cameras = []
        for key in self.cameras:
            feature = info["features"].get(key)
            shape = feature.get("shape") if isinstance(feature, dict) else None
            valid = (
                isinstance(shape, (tuple, list))
                and len(shape) == 3
                and all(type(v) is int and v > 0 for v in shape)
                and shape[-1] == 3
            )
            valid = valid or (
                isinstance(feature, dict)
                and feature.get("dtype") == "video"
                and shape is None
                and info.get("video_layout") == "per_episode_variable"
            )
            if valid:
                available_cameras.append(key)
            elif key not in (CAMERA_KEYS["left"], CAMERA_KEYS["right"]):
                raise ValueError(f"{name}: required camera {key} is absent or invalid")
        if not available_cameras:
            raise ValueError(f"{name}: no valid enabled cameras")
        action_mask = validate_mask(info["action_dim_mask"], context=name)
        state_mask = validate_mask(info.get("state_dim_mask", info["action_dim_mask"]), context=name)
        rate = entry.get("acceleration_rate", 1)
        if isinstance(rate, bool) or int(rate) != rate or rate < 1:
            raise ValueError(f"{name}: acceleration_rate must be a positive integer")
        sampling = float(entry.get("sampling_rate", 1))
        if not math.isfinite(sampling) or sampling <= 0:
            raise ValueError(f"{name}: invalid sampling_rate")
        fps = float(info["fps"])
        if float(entry.get("fps", fps)) != fps:
            raise ValueError(f"{name}: manifest/metadata FPS mismatch")
        columns = [
            c
            for c in pq.ParquetFile(next((root / "meta/episodes").glob("chunk-*/*.parquet"))).schema_arrow.names
            if not c.startswith("stats/")
        ]
        eps = []
        for file in sorted((root / "meta/episodes").glob("chunk-*/*.parquet")):
            eps.extend(pq.read_table(file, columns=columns).to_pylist())
        task_rows = pq.read_table(root / "meta/tasks.parquet").to_pylist()
        task_texts = {row.get("task", row.get("__index_level_0__")) for row in task_rows}
        if not task_texts or None in task_texts:
            raise ValueError(f"{name}: cannot resolve meta/tasks.parquet text")
        ignore = set(entry.get("ignore_episodes", []))
        held_out = set()
        if split != "all":
            ids = sorted(int(e["episode_index"]) for e in eps if e["episode_index"] not in ignore)
            rng = np.random.default_rng(self.seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
            held_out = set(rng.permutation(ids)[: min(eval_episodes, max(0, len(ids) - 1))].tolist())
        self.entries[name] = {
            "root": root,
            "info": info,
            "robot": robot,
            "fps": fps,
            "rate": int(rate),
            "action_mask": action_mask,
            "state_mask": state_mask,
            "sampling_rate": sampling,
            "available_cameras": available_cameras,
        }
        for ep in sorted(eps, key=lambda e: e["episode_index"]):
            eid = int(ep["episode_index"])
            if eid in ignore or (split == "train" and eid in held_out) or (split == "eval" and eid not in held_out):
                continue
            annotation = json.loads(ep["annotation"])
            text = annotation["task"]["command"]["en"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{name}/{eid}: missing instruction")
            if text not in task_texts or text not in ep["tasks"]:
                raise ValueError(f"{name}/{eid}: annotation command disagrees with episode/tasks metadata")
            if float(annotation["fps"]) != fps or abs(float(annotation["duration"]) - ep["length"] / fps) > 1 / fps:
                raise ValueError(f"{name}/{eid}: inconsistent annotation FPS/duration")
            dimensions = episode_resolution(info, ep, available_cameras, context=f"{name}/{eid}")
            if annotation.get("resolution") != list(dimensions):
                raise ValueError(f"{name}/{eid}: annotation/metadata resolution mismatch")
            start, end, goal_endpoints = episode_goal_timing(annotation, ep["length"], fps, context=f"{name}/{eid}")
            count = end - start if self.include_tail_windows else end - start - self.horizon * rate
            if count <= 0:
                self.exclusions.append({"dataset": name, "episode": eid, "reason": "short_video_window"})
                continue
            weighted = max(1, round(count * sampling)) if self.training else count
            # Keep only the derived annotation fields below. Retaining every
            # parsed JSON tree makes full GC scan millions of unused containers
            # on large populations; the original JSON remains in metadata.
            self.episodes.append(
                {
                    "name": name,
                    "metadata": ep,
                    "text": text,
                    "text_segments": segment_text_intervals(annotation, fps)
                    if self.text_conditioning != "episode"
                    else [],
                    "start": start,
                    "end": end,
                    "count": count,
                    "weighted": weighted,
                    "goal_endpoints": goal_endpoints,
                    "source_hw": dimensions,
                    "bucket_id": assign_bucket(dimensions, self.img_size_buckets)[0] if self.img_size_buckets else 0,
                }
            )
            self.stops.append((self.stops[-1] if self.stops else 0) + weighted)

    def __len__(self):
        return self.stops[-1]

    def goal_sampling_record(self):
        # Leave legacy terminal-only fingerprints unchanged. Mixture runs must
        # detect annotation/timing changes even with tail windows disabled.
        if not self.random_goal_sampling:
            return None
        population = [
            dict(
                dataset=ep["name"],
                episode=ep["metadata"]["episode_index"],
                start=ep["start"],
                end=ep["end"],
                count=ep["count"],
                weighted=ep["weighted"],
                endpoints=ep["goal_endpoints"],
                fps=self.entries[ep["name"]]["fps"],
                rate=self.entries[ep["name"]]["rate"],
            )
            for ep in self.episodes
        ]
        return dict(
            version=GOAL_SAMPLING_VERSION,
            seed=self.seed,
            config=self.goal_sampling.to_dict(),
            manifest_sha256=self.manifest_sha256,
            population_sha256=hashlib.sha256(json.dumps(population, sort_keys=True).encode()).hexdigest(),
        )

    def text_conditioning_record(self):
        # Omission preserves historical episode-only sampler/checkpoint contracts.
        mode = getattr(self, "text_conditioning", "episode")
        if mode == "episode":
            return None
        population = [
            dict(
                dataset=ep["name"],
                episode=ep["metadata"]["episode_index"],
                text=ep["text"],
                segments=ep["text_segments"],
                start=ep["start"],
                end=ep["end"],
                count=ep["count"],
                weighted=ep["weighted"],
            )
            for ep in self.episodes
        ]
        return dict(
            version=1,
            mode=mode,
            include_robot_type_text_context=self.include_robot_type_text_context,
            manifest_sha256=self.manifest_sha256,
            population_sha256=hashlib.sha256(json.dumps(population, sort_keys=True).encode()).hexdigest(),
        )

    def image_composition_record(self):
        missing = {
            name: [key for key in self.cameras if key not in entry["available_cameras"]]
            for name, entry in self.entries.items()
        }
        missing = {name: keys for name, keys in missing.items() if keys}
        if self.goal_image_composition == "multi_view" and not missing and not self.img_size_buckets:
            return None
        return dict(
            version=2 if self.goal_image_composition == "head_only" or self.img_size_buckets else 1,
            goal_image_composition=self.goal_image_composition,
            missing_cameras=missing,
            resolution=self.resolution,
            img_size=self.img_size,
            enable_cameras=self.enable_cameras,
            **({"resolution_buckets": self.resolution_bucket_record()} if self.img_size_buckets else {}),
        )

    def resolution_bucket_record(self):
        if not self.img_size_buckets:
            return None
        population = [
            (
                ep["name"],
                ep["metadata"]["episode_index"],
                ep["source_hw"],
                ep["bucket_id"],
                ep["start"],
                ep["end"],
                ep["count"],
                ep["weighted"],
                self.entries[ep["name"]]["rate"],
                self.entries[ep["name"]]["fps"],
            )
            for ep in self.episodes
        ]
        return dict(
            version=1,
            buckets=self.img_size_buckets,
            manifest_sha256=self.manifest_sha256,
            population_sha256=hashlib.sha256(json.dumps(population, sort_keys=True).encode()).hexdigest(),
        )

    def target_img_size(self, episode):
        return self.img_size_buckets[episode["bucket_id"]] if self.img_size_buckets else self.img_size

    def selection_record(self):
        text_record = self.text_conditioning_record()
        return {
            "manifest_sha256": self.manifest_sha256,
            "split": self.split,
            "seed": self.seed,
            "horizon": self.horizon,
            "video_stride": self.video_stride,
            "include_tail_windows": self.include_tail_windows,
            "resolution": self.resolution,
            "img_size": self.img_size,
            "image_processing": "letterbox" if self.img_size is not None else "legacy_resize",
            "enable_cameras": self.enable_cameras,
            "supervise_head_eef": self.supervise_head_eef,
            "supervise_arm_head_torso": self.supervise_arm_head_torso,
            "state_arm_eef_coordinate": self.state_arm_eef_coordinate,
            "include_robot_type_text_context": self.include_robot_type_text_context,
            "domain_ids": self.domain_ids,
            "normalizers": {
                k: dict(sha256=n.sha256, norm_type=n.norm_type, metadata=n.metadata)
                for k, n in self.normalizers.items()
            },
            "episodes": [
                dict(
                    dataset=e["name"],
                    episode=e["metadata"]["episode_index"],
                    start=e["start"],
                    end=e["end"],
                    count=e["count"],
                    weighted=e["weighted"],
                )
                for e in self.episodes
            ],
            "exclusions": self.exclusions,
            **({"metadata_assumptions": self.metadata_assumptions} if self.metadata_assumptions else {}),
            **({"goal_sampling": self.goal_sampling_record()} if self.random_goal_sampling else {}),
            **({"text_conditioning": text_record} if text_record is not None else {}),
            **(
                {"image_composition": self.image_composition_record()}
                if self.image_composition_record() is not None
                else {}
            ),
        }

    def locate(self, index):
        if isinstance(index, tuple):
            index, _ = index
        if not 0 <= index < len(self):
            raise IndexError(index)
        slot = bisect.bisect_right(self.stops, index)
        ep = self.episodes[slot]
        offset = index - (self.stops[slot - 1] if slot else 0)
        offset = min(ep["count"] - 1, offset * ep["count"] // ep["weighted"])
        return ep, ep["start"] + offset

    def get_shuffle_blocks(self):
        return [(0 if i == 0 else self.stops[i - 1], stop) for i, stop in enumerate(self.stops)]

    def _episode_rows(self, ep):
        name, meta = ep["name"], ep["metadata"]
        key = name, meta["episode_index"]
        if key not in self._rows:
            entry = self.entries[name]
            cache = getattr(self, "_parquet_cache", None)
            if cache is not None:
                table = cache.read_episode(entry["root"], entry["info"], meta)
                columns = ["observation.state", "action", "frame_index"]
                columns.extend(
                    k for k in ("action_valid_mask", "observation.state_valid_mask") if k in table.column_names
                )
                table = table.select(columns)
            else:
                dataset = pads.dataset(
                    str(entry["root"] / "data"),
                    format="parquet",
                    exclude_invalid_files=True,
                )
                columns = ["observation.state", "action", "frame_index"]
                columns.extend(
                    k for k in ("action_valid_mask", "observation.state_valid_mask") if k in dataset.schema.names
                )
                table = dataset.to_table(
                    columns=columns,
                    filter=pads.field("episode_index") == meta["episode_index"],
                )
            table = table.sort_by("frame_index")
            if table["frame_index"].to_pylist() != list(range(meta["length"])):
                raise ValueError(f"{key}: noncontiguous episode rows")
            tensors = {}
            for column in columns:
                if column == "frame_index":
                    continue
                values = table[column].combine_chunks()
                flattened = pc.list_flatten(values)
                if values.null_count or flattened.null_count:
                    raise ValueError(f"{key}: {column} contains null values")
                lengths = pc.list_value_length(values).to_numpy(zero_copy_only=False)
                if not np.equal(lengths, 49).all():
                    raise ValueError(f"{key}: {column} rows must have 49 values")
                array = flattened.to_numpy(zero_copy_only=False).reshape(-1, 49)
                tensors[column] = torch.tensor(array, dtype=torch.bool if "mask" in column else torch.float32)
            self._rows[key] = tensors
            while len(self._rows) > 2:
                self._rows.popitem(last=False)
        self._rows.move_to_end(key)
        return self._rows[key]

    def physical_sample(self, index):
        # Integer access is a deterministic preview; training supplies the
        # consumed global occurrence so repeated windows get new goals.
        index, occurrence = index if isinstance(index, tuple) else (index, index)
        ep, start = self.locate(index)
        entry = self.entries[ep["name"]]
        goal = select_goal(
            self.goal_sampling,
            start=start,
            end=ep["end"],
            fps=entry["fps"],
            rate=entry["rate"],
            horizon=self.horizon,
            endpoints=ep["goal_endpoints"],
            seed=self.seed,
            occurrence=occurrence,
            training=self.training,
        )
        rows = self._episode_rows(ep)
        indices = start + torch.arange(self.horizon) * entry["rate"]
        action_time_valid = indices < ep["end"]
        read_indices = indices.clamp(max=ep["end"] - 1)
        state = rows["observation.state"][start].clone()
        actions = rows["action"][read_indices].clone()
        sm = entry["state_mask"].clone()
        am = entry["action_mask"].expand_as(actions).clone()
        for key, selected, mask in (
            ("observation.state_valid_mask", start, sm),
            ("action_valid_mask", read_indices, am),
        ):
            if key in rows:
                frame_mask = validate_mask(rows[key][selected], context=key)
                if (frame_mask & ~mask).any():
                    warnings.warn(
                        f"{ep['name']}: {key} widens dataset mask; dataset mask takes precedence",
                        stacklevel=2,
                    )
                mask &= frame_mask
        am &= action_time_valid[:, None]
        validate_values(state, sm, context=f"{ep['name']}/{start}/state")
        validate_values(actions, am, context=f"{ep['name']}/{start}/action")
        state.masked_fill_(~sm, 0)
        actions.masked_fill_(~am, 0)
        check_anchor_head_quaternion(state, sm, self.layout, context=ep["name"])
        relative_mask = propagate_relative_action_validity(am, sm, self.layout)
        relative = to_relative_action(state, actions, self.layout)
        model_state = transform_state_arm_eef_coordinate(state, self.state_arm_eef_coordinate, self.layout)
        model_sm = propagate_state_arm_eef_validity(sm, self.state_arm_eef_coordinate, self.layout)
        if not self.supervise_head_eef:
            relative_mask[..., 42:49] = False
            model_sm[42:49] = False
        if not self.supervise_arm_head_torso:
            relative_mask[..., self.layout.joint] = False
            model_sm[self.layout.joint] = False
        relative = relative.masked_fill(~relative_mask, 0)
        model_state = zero_chassis_state(model_state, self.layout).masked_fill(~model_sm, 0)
        normalizer = self.normalizers[entry["robot"]]
        norm_state = normalizer.normalize(model_state, "observation.state", model_sm)
        norm_actions = normalizer.normalize(relative, "action", relative_mask)
        # Wan compresses four video transitions per latent. Truncate to a real
        # 1+4*n prefix BEFORE encoding so no latent contains repeated tail RGB.
        # With <16*r future raw frames, only the clean current frame remains;
        # all recorded action targets are still supervised independently.
        video_blocks = min(
            self.horizon // (self.video_stride * 4), (ep["end"] - 1 - start) // (entry["rate"] * self.video_stride * 4)
        )
        video_indices = start + torch.arange(1 + 4 * video_blocks) * self.video_stride * entry["rate"]
        return {
            "anchor_state": state,
            "anchor_valid": sm,
            "absolute_actions": actions,
            "relative_actions": relative,
            "relative_valid": relative_mask,
            "model_state": model_state,
            "action": torch.cat([norm_state[None], norm_actions]),
            "action_valid_mask": torch.cat([model_sm[None], relative_mask]),
            "dataset": ep["name"],
            "robot_type": entry["robot"],
            "episode_index": int(ep["metadata"]["episode_index"]),
            "start": start,
            "action_indices": indices,
            "action_time_valid": action_time_valid,
            "video_indices": video_indices,
            **goal,
            "normalizer": normalizer,
            "episode": ep,
        }

    def _video(self, ep, camera, indices):
        entry, meta = self.entries[ep["name"]], ep["metadata"]
        path = entry["root"] / entry["info"]["video_path"].format(
            video_key=camera,
            chunk_index=meta[f"videos/{camera}/chunk_index"],
            file_index=meta[f"videos/{camera}/file_index"],
        )
        key = str(path)
        if key not in self._readers:
            self._readers[key] = FrameReader(path, entry["fps"])
            while len(self._readers) > 6:
                self._readers.popitem(last=False)[1].close()
        self._readers.move_to_end(key)
        timestamps = [i / entry["fps"] + meta[f"videos/{camera}/from_timestamp"] for i in indices]
        frames = self._readers[key].read(timestamps)
        if entry["info"].get("video_layout") != "per_episode_variable":
            expected = tuple(entry["info"]["features"][camera]["shape"][:2])
            if frames.shape[1:3] != expected:
                raise ValueError(f"{ep['name']}/{camera}: decoded resolution differs from metadata")
        return frames

    def _camera_frames(self, ep, camera, indices):
        """Absent or undecodable wrist RGB is a black placeholder, never a head fallback."""
        from av.error import FFmpegError

        if camera not in self.entries[ep["name"]]["available_cameras"]:
            return None
        try:
            frames = self._video(ep, camera, indices)
            if (
                not isinstance(frames, np.ndarray)
                or frames.dtype != np.uint8
                or frames.ndim != 4
                or frames.shape[0] != len(indices)
                or frames.shape[-1] != 3
                or min(frames.shape[1:3]) < 1
            ):
                raise ValueError("Expected nonempty uint8 RGB camera frames")
            return torch.from_numpy(frames).permute(0, 3, 1, 2)
        except (FFmpegError, OSError, ValueError, TypeError, KeyError, IndexError) as error:
            if camera not in (CAMERA_KEYS["left"], CAMERA_KEYS["right"]):
                raise
            key = (ep["name"], ep["metadata"]["episode_index"], camera)
            if key not in self._camera_warnings:
                warnings.warn(f"{key}: unavailable wrist images; using black pixels ({error})", stacklevel=2)
                self._camera_warnings[key] = None
                while len(self._camera_warnings) > 64:
                    self._camera_warnings.popitem(last=False)
            return None

    def __getitem__(self, index):
        physical = self.physical_sample(index)
        ep, entry = physical["episode"], self.entries[physical["dataset"]]
        indices = physical["video_indices"].tolist()
        goal_index = [physical["goal_index"]]
        rollout_views, goal_views = {}, {}
        for name, camera in CAMERA_KEYS.items():
            if camera not in self.cameras:
                continue
            if self.goal_image_composition == "head_only" and name in ("left", "right"):
                rollout_views[name] = self._camera_frames(ep, camera, indices)
                continue
            frames = self._camera_frames(ep, camera, indices + goal_index)
            if frames is not None:
                rollout_views[name], goal_views[name] = frames[:-1], frames[-1:]
            else:
                # A missing goal frame must not discard valid current/rollout RGB.
                rollout_views[name] = self._camera_frames(ep, camera, indices)
                goal_views[name] = self._camera_frames(ep, camera, goal_index)
        img_size = self.target_img_size(ep)
        canvas, pixel_mask, boxes = compose_goal_image(
            rollout_views, img_size, self.enable_cameras, resolution=self.resolution
        )
        goal, goal_mask, goal_boxes = compose_goal_image(
            goal_views, img_size, self.enable_cameras, self.goal_image_composition, resolution=self.resolution
        )
        image_metadata = dict(
            video_pixel_mask=pixel_mask, camera_boxes=boxes, goal_pixel_mask=goal_mask, goal_camera_boxes=goal_boxes
        )
        caption = select_text(ep["text"], ep["text_segments"], physical["start"], self.text_conditioning)
        if self.include_robot_type_text_context:
            caption = f"<robot_type>{entry['robot']}</robot_type> {caption}"
        return {
            "video": canvas.permute(1, 0, 2, 3).contiguous(),
            "goal_frame": goal.permute(1, 0, 2, 3).contiguous(),
            "action": physical["action"],
            "action_valid_mask": physical["action_valid_mask"],
            "ai_caption": caption,
            "mode": "policy",
            "viewpoint": "concat_view",
            "fps": torch.tensor(entry["fps"] / entry["rate"] / self.video_stride),
            "conditioning_fps": torch.tensor(entry["fps"] / entry["rate"] / self.video_stride),
            "action_fps": torch.tensor(entry["fps"] / entry["rate"]),
            "domain_id": torch.tensor(self.domain_ids[entry["robot"]]),
            "dataset_name": ep["name"],
            "episode_index": physical["episode_index"],
            "window_start_frame": physical["start"],
            "goal_frame_index": physical["goal_index"],
            **{key: value for key, value in physical.items() if key.startswith("goal_") and key != "goal_index"},
            **image_metadata,
            **(
                {"target_hw": torch.tensor(img_size), "image_bucket_id": ep["bucket_id"]}
                if self.img_size_buckets
                else {}
            ),
        }

    def close(self):
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        if self._parquet_cache is not None:
            self._parquet_cache.close()


class LeRobot0824SFTDataset(Dataset):
    def __init__(self, dataset, *, tokenizer_config=None, max_action_dim=64, cfg_dropout_rate=0.0):
        from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

        if not 0.0 <= cfg_dropout_rate <= 1.0:
            raise ValueError("cfg_dropout_rate must be in [0, 1]")
        self.cfg_dropout_rate = cfg_dropout_rate
        self._dataset = dataset
        self._transform = ActionTransformPipeline(
            pad_keys=["video", "goal_frame"],
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=cfg_dropout_rate,
            max_action_dim=max_action_dim,
            action_video_downsample_factor=4,
            goal_frame_key="goal_frame",
            goal_frame_injection="generator",
            goal_layout="concat",
            append_viewpoint_info=False,
            append_duration_fps_timestamps=False,
            append_resolution_info=False,
            allow_partial_video=getattr(dataset, "include_tail_windows", False),
        )
        if dataset.img_size is not None:
            self._transform.video_resize = PreparedVideo()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        ep, _ = self._dataset.locate(index)
        robot = self._dataset.entries[ep["name"]]["robot"]
        return self.prepare_sample(self._dataset[index], robot)

    def prepare_sample(self, sample, robot):
        from cosmos_framework.data.vfm.action.action_processing import ActionProcessingRecord

        data = self._transform(dict(sample), self._dataset.resolution)
        data["action_processing_record"] = ActionProcessingRecord(49, self._dataset.normalizers[robot])
        return data

    def get_shuffle_blocks(self):
        return self._dataset.get_shuffle_blocks()
