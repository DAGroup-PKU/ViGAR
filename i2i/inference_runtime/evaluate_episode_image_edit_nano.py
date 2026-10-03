# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Evaluate the local episode image-edit Cosmos3-Nano SFT checkpoint.

The eval path intentionally mirrors the SFT entrypoint:
TOML config -> model/dataloader instantiation -> DCP model load -> trainer.validate().
It writes a small metrics JSON beside the usual Cosmos run outputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger as logging
from PIL import Image
from tqdm import tqdm

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, ema, misc
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.config import Config
from cosmos_framework.utils.context_managers import (
    data_loader_init,
    distributed_init,
    model_init,
)
from cosmos_framework.utils.lazy_config import instantiate


CHECKPOINT_RE = re.compile(r"iter_(\d{9})$")
_SYSTEM_PROMPT_IMAGE_EDITING = (
    "You are a helpful assistant who will edit images based on the user's instructions."
)
_SUBGOAL_CASES = ("all", "current_segment", "next_segment_tail", "final_segment")


def _per_rank_iteration_count(total: int) -> int:
    """Partition an exact global sample count across torchrun ranks.

    ``MapDistributor`` assigns the first remainder samples to the lowest ranks. Matching that
    partition here lets a multi-GPU evaluation cover a finite map-style validation set exactly
    once, instead of rounding every rank up and regenerating samples in the next epoch.
    """

    if total < 0:
        raise ValueError(f"iteration total must be >= 0, got {total}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(
            f"Invalid distributed environment: RANK={rank}, WORLD_SIZE={world_size}"
        )
    base, remainder = divmod(total, world_size)
    return base + int(rank < remainder)


def _local_map_sample_count(total: int, num_workers: int) -> int:
    """Return this rank's samples for MapDistributor's rank/worker striding."""

    if num_workers < 1:
        num_workers = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    total_streams = world_size * num_workers
    first_stream = rank * num_workers
    return sum(
        (total - stream_id + total_streams - 1) // total_streams
        for stream_id in range(first_stream, first_stream + num_workers)
        if stream_id < total
    )


def _dataloader_sample_count(dataloader: torch.utils.data.DataLoader) -> int:
    """Return the finite map-style dataset size beneath a CosmosDataLoader.

    The public dataloader is intentionally infinite (it repeats epochs), but generation-only
    evaluation needs one exact pass. Its iterable wrapper retains the MapDistributor, whose
    length is the underlying selected dataset length.
    """

    distributor = getattr(getattr(dataloader, "dataset", None), "_distributor", None)
    if distributor is None:
        raise TypeError(
            "Cannot derive evaluation sample count: dataloader has no dataflow distributor"
        )
    try:
        count = len(distributor)
    except TypeError as exc:
        raise TypeError(
            "Cannot derive evaluation sample count from a non-map distributor"
        ) from exc
    if count < 1:
        raise ValueError("Derived evaluation sample count is zero")
    return int(count)


class EvalMetricsCallback(Callback):
    """Collect validation metrics and write them to a rank-0 JSON file."""

    def __init__(
        self,
        *,
        metrics_path: Path,
        checkpoint_path: Path,
        checkpoint_iteration: int,
        max_val_iter: int | None,
    ) -> None:
        super().__init__()
        self.metrics_path = metrics_path
        self.checkpoint_path = checkpoint_path
        self.checkpoint_iteration = checkpoint_iteration
        self.max_val_iter = max_val_iter
        self.started_at = 0.0
        self.loss_sum: torch.Tensor | None = None
        self.sample_size: torch.Tensor | None = None
        self.iter_count: torch.Tensor | None = None
        self.pbar: Any | None = None

    def on_validation_start(
        self,
        model: ImaginaireModel,
        dataloader_val: torch.utils.data.DataLoader,
        iteration: int = 0,
    ) -> None:
        del model, iteration
        self.started_at = time.time()
        self.loss_sum = torch.tensor(0.0, device="cuda")
        self.sample_size = torch.tensor(0.0, device="cuda")
        self.iter_count = torch.tensor(0, device="cuda", dtype=torch.long)
        if distributed.is_rank0():
            try:
                total = (
                    self.max_val_iter
                    if self.max_val_iter is not None
                    else len(dataloader_val)
                )
            except TypeError:
                total = self.max_val_iter
            self.pbar = tqdm(total=total, desc="Evaluating", dynamic_ncols=True)

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, output_batch, iteration
        assert self.loss_sum is not None
        assert self.sample_size is not None
        assert self.iter_count is not None
        batch_size = float(misc.get_data_batch_size(data_batch))
        self.loss_sum += loss.detach().float() * batch_size
        self.sample_size += batch_size
        self.iter_count += 1
        if self.pbar is not None:
            elapsed_sec = time.time() - self.started_at
            self.pbar.set_postfix(
                {
                    "loss": f"{loss.detach().float().item():.4f}",
                    "iter": int(self.iter_count.item()),
                    "time": f"{elapsed_sec:.1f}s",
                }
            )
            self.pbar.update(1)

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model
        assert self.loss_sum is not None
        assert self.sample_size is not None
        assert self.iter_count is not None
        dist.all_reduce(self.loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.sample_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.iter_count, op=dist.ReduceOp.SUM)
        elapsed_sec = time.time() - self.started_at
        val_loss = (
            (self.loss_sum / self.sample_size).item()
            if self.sample_size.item() > 0
            else float("nan")
        )
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

        metrics = {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_iteration": self.checkpoint_iteration,
            "iteration": iteration,
            "max_val_iter": self.max_val_iter,
            "val_loss": val_loss,
            "validation_iters_total_across_ranks": int(self.iter_count.item()),
            "sample_size_total_across_ranks": float(self.sample_size.item()),
            "elapsed_sec": elapsed_sec,
            "world_size": distributed.get_world_size(),
        }
        if distributed.is_rank0():
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.metrics_path.with_suffix(self.metrics_path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            tmp_path.replace(self.metrics_path)
            logging.info(f"Wrote eval metrics: {self.metrics_path}")


def _metadata_item(value: Any, index: int = 0) -> Any:
    """Extract one scalar-ish value from the nested dataloader metadata format."""

    while isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[index if len(value) > index else 0]
        index = 0
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _safe_stem(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:max_len] or "sample"


def _task_dir_name(url: str, dataset_name: str = "", task_name: str = "") -> str:
    """Task subfolder for a sample: the LeRobot dataset dir name (e.g. ``adjust_bottle``).

    Prefers the per-dataset ``dataset_name`` suffix set by the multi-root factory
    (``episode_image_editing_val:adjust_bottle_lerobot``); otherwise infers the dataset
    root from the video path (``<root>/videos/...`` for LeRobot, ``<root>/video/...`` for
    flat). Strips a trailing ``_lerobot`` for cleaner directory names.
    """
    name = task_name.strip()
    if dataset_name and ":" in dataset_name:
        name = dataset_name.split(":")[-1]
    if not name and url:
        parts = Path(url).parts
        for anchor in ("videos", "video"):
            if anchor in parts:
                idx = parts.index(anchor)
                if idx > 0:
                    name = parts[idx - 1]
                    break
    name = name or "dataset"
    if name.endswith("_lerobot"):
        name = name[: -len("_lerobot")]
    return _safe_stem(name)


def _reasoner_text_metrics(prediction: str, target: str) -> dict[str, float]:
    """Case/punctuation-insensitive exact match and bag-of-token F1."""

    pred_tokens = re.findall(r"[a-z0-9]+", prediction.lower())
    target_tokens = re.findall(r"[a-z0-9]+", target.lower())
    exact_match = float(pred_tokens == target_tokens and bool(target_tokens))
    if not pred_tokens or not target_tokens:
        return {"exact_match": exact_match, "token_f1": 0.0}
    overlap = sum((Counter(pred_tokens) & Counter(target_tokens)).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    token_f1 = (
        0.0
        if precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {"exact_match": exact_match, "token_f1": token_f1}


def _demo_variant(url: str) -> str:
    """clean vs randomized demo subfolder, inferred from the dataset path.

    RoboTwin sources encode this as ``demo_clean`` / ``demo_randomized`` path segments.
    When neither marker is present (e.g. robotwin_demo10), default to ``demo_randomized``.
    """
    parts = [p for p in re.split(r"[/\\]", str(url).lower()) if p]
    if any("clean" in p for p in parts):
        return "demo_clean"
    if any("random" in p for p in parts):
        return "demo_randomized"
    return "demo_randomized"


def _extract_image_pair(
    data_batch: dict[str, Any], sample_idx: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    images = data_batch["images"][sample_idx]
    if not isinstance(images, (list, tuple)) or len(images) < 2:
        raise ValueError(
            f"Expected image-edit sample with source and target images, got {type(images)}"
        )
    source = images[0].detach().cpu()
    target = images[-1].detach().cpu()
    return source, target


def _uint8_chw_to_unit(image: torch.Tensor) -> torch.Tensor:
    while image.ndim > 3 and image.shape[0] == 1:
        image = image.squeeze(0)
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
    return image.float().div(255.0).clamp(0.0, 1.0)


def _image_edit_reasoner_messages(
    prompt: str, system_prompt: str
) -> list[dict[str, str]]:
    """Use the same system/user chat template as the episode image-edit dataset."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _decoded_to_unit_image(decoded: torch.Tensor) -> torch.Tensor:
    """Convert a decoded VAE tensor in [-1, 1] to a single CHW image in [0, 1]."""

    image = decoded.detach().float().cpu()
    if image.ndim == 5:  # [B, C, T, H, W]
        image = image[0, :, -1]
    elif image.ndim == 4:  # [C, T, H, W] or [B, C, H, W]
        if image.shape[0] == 1:
            image = image[0]
        else:
            image = image[:, -1]
    if image.ndim != 3:
        raise ValueError(
            f"Expected decoded image/video tensor, got shape {tuple(decoded.shape)}"
        )
    return ((image + 1.0) / 2.0).clamp(0.0, 1.0)


def _resize_like(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if image.shape[-2:] == reference.shape[-2:]:
        return image
    resized = F.interpolate(
        image.unsqueeze(0),
        size=reference.shape[-2:],
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)
    return resized.clamp(0.0, 1.0)


def _save_unit_image(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (
        image.detach()
        .float()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .add(0.5)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    Image.fromarray(np.ascontiguousarray(arr), mode="RGB").save(path)


def _image_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = pred.float() - target.float()
    mae = diff.abs().mean().item()
    mse = diff.square().mean().item()
    psnr = float("inf") if mse == 0.0 else -10.0 * math.log10(mse)
    return {"mae": mae, "mse": mse, "psnr": psnr}


def _classify_subgoal_mapping(
    *,
    input_frame_index: int,
    target_frame_index: int,
    source_segment_index: int,
    target_segment_index: int,
    source_segment_start_frame: int,
    source_segment_end_frame_exclusive: int,
    target_segment_end_frame_exclusive: int,
    num_segments: int,
    tail_fraction: float,
) -> str:
    """Validate the segment-final contract and return its metric subgroup."""

    if not 0.0 <= tail_fraction <= 1.0:
        raise ValueError(f"tail_fraction must be in [0, 1], got {tail_fraction}")
    if not 0 <= source_segment_index < num_segments:
        raise ValueError(
            f"Invalid source segment {source_segment_index} for num_segments={num_segments}"
        )
    if (
        not source_segment_start_frame
        <= input_frame_index
        < source_segment_end_frame_exclusive
    ):
        raise ValueError(
            "Input frame is outside its declared source segment: "
            f"frame={input_frame_index}, segment=[{source_segment_start_frame}, "
            f"{source_segment_end_frame_exclusive})"
        )

    segment_length = source_segment_end_frame_exclusive - source_segment_start_frame
    tail_frames = math.ceil(segment_length * tail_fraction)
    in_tail = (
        tail_frames > 0
        and input_frame_index >= source_segment_end_frame_exclusive - tail_frames
    )
    is_final_segment = source_segment_index == num_segments - 1
    expected_target_segment = (
        source_segment_index + 1
        if in_tail and not is_final_segment
        else source_segment_index
    )
    if target_segment_index != expected_target_segment:
        raise ValueError(
            "Subgoal segment mapping violates the configured tail rule: "
            f"source_segment={source_segment_index}, input_frame={input_frame_index}, "
            f"tail_fraction={tail_fraction}, expected_target_segment={expected_target_segment}, "
            f"actual_target_segment={target_segment_index}"
        )
    expected_target_frame = target_segment_end_frame_exclusive - 1
    if target_frame_index != expected_target_frame:
        raise ValueError(
            "Subgoal target is not the target segment's final frame: "
            f"expected={expected_target_frame}, actual={target_frame_index}"
        )
    if is_final_segment:
        return "final_segment"
    return (
        "next_segment_tail"
        if expected_target_segment > source_segment_index
        else "current_segment"
    )


def _merge_json_file(path: Path, update: dict[str, Any]) -> None:
    payload: dict[str, Any] = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(update)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp_path.replace(path)


def _wandb_log_generation(
    sample_info: dict[str, Any], metrics: dict[str, Any], step: int, max_images: int
) -> None:
    if not distributed.is_rank0() or max_images <= 0:
        return
    try:
        import wandb
    except Exception:
        return
    if wandb.run is None:
        return

    log_payload: dict[str, Any] = {
        f"generation/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))
    }
    if int(sample_info["sample_iter"]) < max_images:
        log_payload["generation/grid"] = wandb.Image(
            sample_info["grid_path"],
            caption=f"{sample_info['key']}: {sample_info['prompt']}",
        )
    wandb.log(log_payload, step=step)


def _run_generation_eval(
    *,
    config: Config,
    model: ImaginaireModel,
    dataloader_val: torch.utils.data.DataLoader,
    checkpoint_path: Path,
    checkpoint_iteration: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    generation_total = args.max_gen_iter_total
    if generation_total == -1:
        generation_total = _dataloader_sample_count(dataloader_val)
    valid_gen_iter = None
    if generation_total is not None:
        # FSDP generation performs collectives, so every rank must execute the same number of
        # forwards. MapDistributor can assign one fewer sample to trailing ranks when the dataset
        # size is not divisible by world_size * num_workers. Those ranks run one padded forward
        # from the next epoch, but exclude it from outputs and metrics.
        valid_gen_iter = _local_map_sample_count(
            generation_total, int(getattr(dataloader_val, "num_workers", 0))
        )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        max_gen_iter = math.ceil(generation_total / world_size)
    else:
        max_gen_iter = args.max_gen_iter
    if max_gen_iter <= 0:
        return {"generation_enabled": False, "max_gen_iter": max_gen_iter}

    rank = distributed.get_rank()
    target_mode = (
        os.environ.get("EPISODE_IMAGE_EDIT_TARGET_MODE", "episode_final")
        .strip()
        .lower()
    )
    reference_kind = (
        "subgoal_frame" if target_mode == "segment_final" else "episode_final_frame"
    )
    if args.generate_reasoner_text and not getattr(
        model.config, "reasoner_gen_prompt_only", False
    ):
        raise ValueError(
            "--generate-reasoner-text requires model.config.reasoner_gen_prompt_only=true so the "
            "image generator remains restricted to the original prompt K/V."
        )
    generated_root = Path(config.job.path_local) / "generated"
    generated_root.mkdir(parents=True, exist_ok=True)
    metadata_path = generated_root / f"metadata_rank_{rank:03d}.jsonl"
    # Each row is mae, mse, psnr, count. Row 0 is overall; the other rows are the
    # auditable subgoal-mapping groups in ``_SUBGOAL_CASES``.
    stats = torch.zeros((len(_SUBGOAL_CASES), 4), device="cuda", dtype=torch.float64)
    reasoner_stats = torch.zeros(
        3, device="cuda", dtype=torch.float64
    )  # exact, token_f1, count
    elapsed = torch.zeros(1, device="cuda", dtype=torch.float64)
    started_at = time.time()
    pbar: tqdm | None = None
    if distributed.is_rank0():
        pbar = tqdm(
            total=max_gen_iter, desc="Generating eval images", dynamic_ncols=True
        )

    model.eval()
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        with ema.ema_scope(model, enabled=model.config.ema.enabled):
            for gen_iter, data_batch in enumerate(dataloader_val):
                if gen_iter >= max_gen_iter:
                    break

                source_u8, target_u8 = _extract_image_pair(data_batch)
                source = _uint8_chw_to_unit(source_u8)
                target = _uint8_chw_to_unit(target_u8)

                key = str(
                    _metadata_item(data_batch.get("__key__"))
                    or f"sample_{gen_iter:05d}"
                )
                url = str(_metadata_item(data_batch.get("__url__")) or "")
                dataset_name = str(_metadata_item(data_batch.get("dataset_name")) or "")
                task_name = str(_metadata_item(data_batch.get("task_name")) or "")
                prompt = str(_metadata_item(data_batch.get("ai_caption")) or "")
                reasoner_prompt = str(
                    _metadata_item(data_batch.get("reasoner_prompt")) or prompt
                )
                reasoner_target_text = str(
                    _metadata_item(data_batch.get("reasoner_target_text")) or ""
                )
                episode_id = _metadata_item(data_batch.get("episode_id"))
                input_frame_index = _metadata_item(data_batch.get("input_frame_index"))
                target_frame_index = _metadata_item(
                    data_batch.get("target_frame_index")
                )
                source_segment_index = _metadata_item(
                    data_batch.get("source_segment_index")
                )
                target_segment_index = _metadata_item(
                    data_batch.get("target_segment_index")
                )
                source_segment_start_frame = _metadata_item(
                    data_batch.get("source_segment_start_frame")
                )
                source_segment_end_frame_exclusive = _metadata_item(
                    data_batch.get("source_segment_end_frame_exclusive")
                )
                target_segment_start_frame = _metadata_item(
                    data_batch.get("target_segment_start_frame")
                )
                target_segment_end_frame_exclusive = _metadata_item(
                    data_batch.get("target_segment_end_frame_exclusive")
                )
                num_segments = _metadata_item(data_batch.get("num_segments"))
                subgoal_case: str | None = None
                if target_mode == "segment_final":
                    required_mapping_fields = {
                        "input_frame_index": input_frame_index,
                        "target_frame_index": target_frame_index,
                        "source_segment_index": source_segment_index,
                        "target_segment_index": target_segment_index,
                        "source_segment_start_frame": source_segment_start_frame,
                        "source_segment_end_frame_exclusive": source_segment_end_frame_exclusive,
                        "target_segment_end_frame_exclusive": target_segment_end_frame_exclusive,
                        "num_segments": num_segments,
                    }
                    missing = [
                        name
                        for name, value in required_mapping_fields.items()
                        if value is None
                    ]
                    if missing:
                        raise ValueError(
                            f"Segment-final eval sample is missing mapping metadata: {missing}"
                        )
                    subgoal_case = _classify_subgoal_mapping(
                        input_frame_index=int(input_frame_index),
                        target_frame_index=int(target_frame_index),
                        source_segment_index=int(source_segment_index),
                        target_segment_index=int(target_segment_index),
                        source_segment_start_frame=int(source_segment_start_frame),
                        source_segment_end_frame_exclusive=int(
                            source_segment_end_frame_exclusive
                        ),
                        target_segment_end_frame_exclusive=int(
                            target_segment_end_frame_exclusive
                        ),
                        num_segments=int(num_segments),
                        tail_fraction=float(
                            os.environ.get(
                                "EPISODE_IMAGE_EDIT_NEXT_SUBGOAL_TAIL_FRACTION", "0.15"
                            )
                        ),
                    )
                seed = int(
                    args.seed_base + checkpoint_iteration + rank * 1_000_003 + gen_iter
                )

                reasoner_generated_text: str | None = None
                reasoner_text_metrics: dict[str, float] | None = None
                if args.generate_reasoner_text:
                    # Match reasoner training exactly: VAE(source-frame) prefix followed by the
                    # original episode-level chat prompt. The generated subtask is recorded for
                    # inspection but is NOT fed back into image generation; prompt-only K/V stays
                    # source vision + original prompt.
                    source_for_vae = (
                        source_u8.to(device="cuda", dtype=torch.float32)
                        .div(127.5)
                        .sub(1.0)
                        .unsqueeze(0)
                        .unsqueeze(2)
                    )
                    reasoner_source_latents = (
                        model.encode(source_for_vae).contiguous().float()
                    )
                    reasoner_generated_text = model.generate_reasoner_text(
                        [reasoner_prompt],
                        max_new_tokens=args.reasoner_max_new_tokens,
                        reasoner_vision_latents=[reasoner_source_latents],
                        prompt_builder=lambda prompt: _image_edit_reasoner_messages(
                            prompt, args.reasoner_system_prompt
                        ),
                        do_sample=False,
                        seed=seed,
                    )[0]
                    if reasoner_target_text:
                        reasoner_text_metrics = _reasoner_text_metrics(
                            reasoner_generated_text, reasoner_target_text
                        )
                        reasoner_stats[0] += reasoner_text_metrics["exact_match"]
                        reasoner_stats[1] += reasoner_text_metrics["token_f1"]
                        reasoner_stats[2] += 1.0

                    # Keep image generation on the same original task supplied to the reasoner.
                    # This is intentionally independent of annotation-bearing evaluation fields.
                    data_batch[model.input_caption_key] = [reasoner_prompt]
                    prompt = reasoner_prompt

                data_batch = misc.to(data_batch, device="cuda")
                outputs = model.generate_samples_from_batch(
                    data_batch,
                    guidance=args.guidance,
                    n_sample=1,
                    num_steps=args.num_steps,
                    shift=args.shift,
                    sigma_max=args.sigma_max,
                    seed=[seed],
                )
                pred = _decoded_to_unit_image(model.decode(outputs["vision"][0]))
                pred = _resize_like(pred, target)
                source = _resize_like(source, target)

                if valid_gen_iter is not None and gen_iter >= valid_gen_iter:
                    continue

                sample_metrics = _image_metrics(pred, target)
                metric_values = torch.tensor(
                    [
                        sample_metrics["mae"],
                        sample_metrics["mse"],
                        0.0
                        if math.isinf(sample_metrics["psnr"])
                        else sample_metrics["psnr"],
                        1.0,
                    ],
                    device="cuda",
                    dtype=torch.float64,
                )
                stats[0] += metric_values
                if subgoal_case is not None:
                    stats[_SUBGOAL_CASES.index(subgoal_case)] += metric_values

                # Group all samples from an episode under one directory, then include both frame
                # indices in the leaf so multiple subgoal predictions cannot overwrite each other.
                task = _task_dir_name(url, dataset_name, task_name)
                demo = _demo_variant(url)
                if episode_id is None:
                    episode_dir = generated_root / task / demo / _safe_stem(key)
                    sample_dir = episode_dir
                else:
                    input_name = (
                        "unknown"
                        if input_frame_index is None
                        else f"{int(input_frame_index):06d}"
                    )
                    target_name = (
                        "unknown"
                        if target_frame_index is None
                        else f"{int(target_frame_index):06d}"
                    )
                    episode_dir = (
                        generated_root / task / demo / f"episode_{int(episode_id):06d}"
                    )
                    sample_dir = (
                        episode_dir / f"input_{input_name}_target_{target_name}"
                    )
                input_path = sample_dir / "input.png"
                target_path = sample_dir / "target.png"
                pred_path = sample_dir / "generated.png"
                grid_path = sample_dir / "grid.png"
                input_prompt_path = sample_dir / "input_prompt.txt"
                reasoner_path = sample_dir / "reasoner_generated.txt"
                _save_unit_image(input_path, source)
                _save_unit_image(target_path, target)
                _save_unit_image(pred_path, pred)
                _save_unit_image(grid_path, torch.cat([source, target, pred], dim=2))
                # Persist the exact text supplied to the generator. For generator-only subgoal
                # training this is the episode-and-subtask caption, not ``reasoner_prompt``.
                input_prompt_path.write_text(prompt + "\n", encoding="utf-8")
                if reasoner_generated_text is not None:
                    reasoner_path.write_text(
                        reasoner_generated_text + "\n", encoding="utf-8"
                    )

                sample_info = {
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_iteration": checkpoint_iteration,
                    "rank": rank,
                    "sample_iter": gen_iter,
                    "seed": seed,
                    "key": key,
                    "url": url,
                    "dataset_name": dataset_name,
                    "task_name": task_name,
                    "task": task,
                    "demo_variant": demo,
                    "prompt": prompt,
                    "reasoner_prompt": reasoner_prompt,
                    "reasoner_system_prompt": args.reasoner_system_prompt,
                    "reasoner_target_text": reasoner_target_text,
                    "reasoner_generated_text": reasoner_generated_text,
                    "reasoner_text_metrics": reasoner_text_metrics,
                    "generator_reasoner_kv_context": (
                        "source_vision_plus_original_prompt"
                        if args.generate_reasoner_text
                        else None
                    ),
                    "reasoner_generation_context": (
                        "vae_source_vision_plus_original_prompt"
                        if args.generate_reasoner_text
                        else None
                    ),
                    "episode_id": episode_id,
                    "input_frame_index": input_frame_index,
                    "target_frame_index": target_frame_index,
                    "source_segment_index": source_segment_index,
                    "target_segment_index": target_segment_index,
                    "source_segment_start_frame": source_segment_start_frame,
                    "source_segment_end_frame_exclusive": source_segment_end_frame_exclusive,
                    "target_segment_start_frame": target_segment_start_frame,
                    "target_segment_end_frame_exclusive": target_segment_end_frame_exclusive,
                    "num_segments": num_segments,
                    "subgoal_case": subgoal_case,
                    "reference_kind": reference_kind,
                    "episode_output_dir": str(episode_dir),
                    "sample_output_dir": str(sample_dir),
                    "input_path": str(input_path),
                    "target_path": str(target_path),
                    "generated_path": str(pred_path),
                    "grid_path": str(grid_path),
                    "input_prompt_path": str(input_prompt_path),
                    "reasoner_generated_path": str(reasoner_path)
                    if reasoner_generated_text is not None
                    else None,
                    "metrics": sample_metrics,
                }
                metadata_file.write(json.dumps(sample_info, sort_keys=True) + "\n")
                metadata_file.flush()
                _wandb_log_generation(
                    sample_info, sample_metrics, checkpoint_iteration, args.wandb_images
                )

                if pbar is not None:
                    pbar.set_postfix(
                        {
                            "mae": f"{sample_metrics['mae']:.4f}",
                            "mse": f"{sample_metrics['mse']:.4f}",
                            "psnr": f"{sample_metrics['psnr']:.2f}",
                            "time": f"{time.time() - started_at:.1f}s",
                        }
                    )
                    pbar.update(1)

    if pbar is not None:
        pbar.close()
    elapsed[0] = time.time() - started_at
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    dist.all_reduce(reasoner_stats, op=dist.ReduceOp.SUM)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

    count = int(stats[0, 3].item())
    generation_metrics = {
        "generation_enabled": True,
        "generation_checkpoint_path": str(checkpoint_path),
        "generation_checkpoint_iteration": checkpoint_iteration,
        "generation_max_gen_iter_per_rank": max_gen_iter,
        "generation_num_steps": args.num_steps,
        "generation_guidance": args.guidance,
        "generation_shift": args.shift,
        "generation_sigma_max": args.sigma_max,
        "generation_seed_base": args.seed_base,
        "generation_reference_kind": reference_kind,
        "generation_reasoner_text_enabled": args.generate_reasoner_text,
        "generation_samples_total_across_ranks": count,
        "generation_elapsed_sec": float(elapsed[0].item()),
        "generation_output_dir": str(Path(config.job.path_local) / "generated"),
        "generation_metadata_glob": str(
            Path(config.job.path_local) / "generated" / "metadata_rank_*.jsonl"
        ),
    }
    if count > 0:
        generation_metrics.update(
            {
                "generation_mae": float((stats[0, 0] / stats[0, 3]).item()),
                "generation_mse": float((stats[0, 1] / stats[0, 3]).item()),
                "generation_psnr": float((stats[0, 2] / stats[0, 3]).item()),
            }
        )
    reasoner_count = int(reasoner_stats[2].item())
    if reasoner_count > 0:
        generation_metrics.update(
            {
                "generation_reasoner_text_samples": reasoner_count,
                "generation_reasoner_exact_match": float(
                    (reasoner_stats[0] / reasoner_stats[2]).item()
                ),
                "generation_reasoner_token_f1": float(
                    (reasoner_stats[1] / reasoner_stats[2]).item()
                ),
            }
        )
    generation_metrics["generation_metrics_by_subgoal_case"] = {
        case: {
            "samples": case_count,
            **(
                {
                    "mae": float((stats[index, 0] / stats[index, 3]).item()),
                    "mse": float((stats[index, 1] / stats[index, 3]).item()),
                    "psnr": float((stats[index, 2] / stats[index, 3]).item()),
                }
                if case_count > 0
                else {}
            ),
        }
        for index, case in enumerate(_SUBGOAL_CASES[1:], start=1)
        for case_count in [int(stats[index, 3].item())]
    }
    if generation_total is not None:
        generation_metrics["generation_max_gen_iter_total"] = generation_total

    if distributed.is_rank0():
        _merge_json_file(
            Path(config.job.path_local) / "eval_metrics.json", generation_metrics
        )
        logging.info(
            f"Wrote generated eval images under: {Path(config.job.path_local) / 'generated'}"
        )
    return generation_metrics


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_train_run_dir() -> Path:
    return (
        _repo_root()
        / "outputs"
        / "arena_train"
        / "cosmos3"
        / "episode_image_edit"
        / "cosmos3-episode-image-edit-nano"
    )


def _resolve_checkpoint(checkpoint: str, train_run_dir: Path) -> Path:
    ckpt_arg = Path(checkpoint).expanduser()
    if not ckpt_arg.is_absolute() and not ckpt_arg.exists():
        repo_relative = _repo_root() / ckpt_arg
        if repo_relative.exists():
            ckpt_arg = repo_relative
    if checkpoint == "latest":
        latest_file = train_run_dir / "checkpoints" / "latest_checkpoint.txt"
        if not latest_file.is_file():
            raise FileNotFoundError(f"latest checkpoint file not found: {latest_file}")
        latest_name = latest_file.read_text(encoding="utf-8").strip()
        ckpt_path = train_run_dir / "checkpoints" / latest_name
    elif ckpt_arg.is_dir() and (ckpt_arg / "latest_checkpoint.txt").is_file():
        latest_name = (
            (ckpt_arg / "latest_checkpoint.txt").read_text(encoding="utf-8").strip()
        )
        ckpt_path = ckpt_arg / latest_name
    elif (
        ckpt_arg.is_dir()
        and (ckpt_arg / "checkpoints" / "latest_checkpoint.txt").is_file()
    ):
        latest_file = ckpt_arg / "checkpoints" / "latest_checkpoint.txt"
        latest_name = latest_file.read_text(encoding="utf-8").strip()
        ckpt_path = ckpt_arg / "checkpoints" / latest_name
    else:
        ckpt_path = ckpt_arg

    ckpt_path = ckpt_path.resolve()
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {ckpt_path}")
    if not (ckpt_path / "model" / ".metadata").is_file():
        raise FileNotFoundError(
            f"DCP model metadata not found under checkpoint: {ckpt_path / 'model'}"
        )
    return ckpt_path


def _checkpoint_iteration(checkpoint_path: Path) -> int:
    match = CHECKPOINT_RE.match(checkpoint_path.name)
    if not match:
        return 0
    return int(match.group(1))


def _build_extra_overrides(
    args: argparse.Namespace, checkpoint_path: Path, checkpoint_iteration: int
) -> list[str]:
    job_name = (
        args.job_name or f"cosmos3-episode-image-edit-nano-eval-{checkpoint_path.name}"
    )
    overrides = [
        f"job.name={job_name}",
        f"job.wandb_mode={args.wandb_mode}",
        f"checkpoint.load_path={checkpoint_path}",
        "checkpoint.load_training_state=false",
        "checkpoint.only_load_scheduler_state=false",
        "checkpoint.keys_to_skip_loading=[]",
        "checkpoint.dcp_async_mode_enabled=false",
        "checkpoint.load_ema_to_reg=false",
        "trainer.run_validation=true",
        "trainer.run_validation_on_start=false",
        "dataloader_val.num_workers=0",
        "dataloader_val.persistent_workers=false",
        "dataloader_val.prefetch_factor=null",
        "model.config.compile.enabled=false",
        f"trainer.max_iter={checkpoint_iteration + 1 if checkpoint_iteration > 0 else 1}",
    ]
    if not args.ema:
        # Default: skip the second fp32 net_ema copy (~2x weights) and load the
        # checkpoint's EMA weights into the single bf16 net. Samples the same EMA
        # weights at ~half the GPU memory. Pass --ema for the heavy net_ema path.
        overrides.append("model.config.ema.enabled=false")
        overrides.append("checkpoint.load_ema_to_reg_single_net=true")
    if args.max_val_iter_total is not None:
        overrides.append(
            f"trainer.max_val_iter={_per_rank_iteration_count(args.max_val_iter_total)}"
        )
    elif args.max_val_iter is not None:
        if args.max_val_iter.lower() == "none":
            overrides.append("trainer.max_val_iter=null")
        else:
            overrides.append(f"trainer.max_val_iter={int(args.max_val_iter)}")
    overrides.extend(args.overrides)
    return overrides


def _keep_eval_callbacks(trainer: Any) -> None:
    keep = {
        ("cosmos_framework.utils.callback", "LowPrecisionCallback"),
        ("cosmos_framework.utils.callback", "ProgressBarCallback"),
        ("cosmos_framework.utils.callback", "WandBCallback"),
        ("cosmos_framework.callbacks.load_pretrained", "LoadPretrained"),
        ("cosmos_framework.callbacks.wandb_log_eval", "WandbCallback"),
    }
    trainer.callbacks._callbacks = [
        cb
        for cb in trainer.callbacks._callbacks
        if (cb.__class__.__module__, cb.__class__.__name__) in keep
    ]


def _append_metrics_callback(
    config: Config,
    trainer: Any,
    checkpoint_path: Path,
    checkpoint_iteration: int,
) -> EvalMetricsCallback:
    metrics_path = Path(config.job.path_local) / "eval_metrics.json"
    metrics_callback = EvalMetricsCallback(
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        checkpoint_iteration=checkpoint_iteration,
        max_val_iter=config.trainer.max_val_iter,
    )
    metrics_callback.config = config
    metrics_callback.trainer = trainer
    trainer.callbacks._callbacks.append(metrics_callback)
    return metrics_callback


@logging.catch(reraise=True)
def launch(args: argparse.Namespace) -> None:
    if args.dataset_path is not None:
        os.environ["EPISODE_IMAGE_EDIT_DATASET_PATH"] = str(
            args.dataset_path.expanduser().resolve()
        )

    checkpoint_path = _resolve_checkpoint(args.checkpoint, args.train_run_dir)
    checkpoint_iteration = _checkpoint_iteration(checkpoint_path)
    output_root = args.output_root.resolve()
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(output_root)

    extra_overrides = _build_extra_overrides(
        args, checkpoint_path, checkpoint_iteration
    )
    config = load_experiment_from_toml(args.sft_toml, extra_overrides=extra_overrides)

    with distributed_init():
        distributed.init()

    config.validate()
    config.freeze()  # type: ignore[attr-defined]
    trainer = config.trainer.type(config)
    if not args.full_callbacks:
        _keep_eval_callbacks(trainer)

    with model_init():
        model = instantiate(config.model)

    with data_loader_init():
        dataloader_val = instantiate(config.dataloader_val)

    metrics_callback = _append_metrics_callback(
        config, trainer, checkpoint_path, checkpoint_iteration
    )

    model = model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    _ = trainer.checkpointer.load(model)
    trainer.callbacks.on_train_start(model, iteration=checkpoint_iteration)

    if args.skip_validation:
        logging.info(
            "Skipping validation loss pass (--skip-validation); running generation evaluation only."
        )
    else:
        logging.info(
            f"Evaluating checkpoint {checkpoint_path} at iteration {checkpoint_iteration} "
            f"with max_val_iter={config.trainer.max_val_iter}"
        )
        trainer.validate(model, dataloader_val, iteration=checkpoint_iteration)
    generation_metrics = _run_generation_eval(
        config=config,
        model=model,
        dataloader_val=dataloader_val,
        checkpoint_path=checkpoint_path,
        checkpoint_iteration=checkpoint_iteration,
        args=args,
    )
    trainer.callbacks.on_train_end(model, iteration=checkpoint_iteration)
    trainer.callbacks.on_app_end()
    trainer.checkpointer.finalize()
    distributed.barrier()

    if distributed.is_rank0():
        logging.info(f"Eval output dir: {config.job.path_local}")
        logging.info(f"Eval metrics: {metrics_callback.metrics_path}")
        if generation_metrics.get("generation_enabled"):
            logging.info(
                f"Generated eval images: {generation_metrics['generation_output_dir']}"
            )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the episode image-edit Cosmos3-Nano SFT checkpoint."
    )
    parser.add_argument(
        "--sft-toml",
        type=Path,
        default=Path(__file__).resolve().parent
        / "toml"
        / "sft_config"
        / "episode_image_edit_nano.toml",
        help="Episode image-edit SFT TOML config.",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help=(
            "'latest', a training run dir, a checkpoints dir with latest_checkpoint.txt, "
            "or an iter_XXXXXXXXX DCP checkpoint directory."
        ),
    )
    parser.add_argument(
        "--train-run-dir",
        type=Path,
        default=_default_train_run_dir(),
        help="Training run directory used when --checkpoint=latest.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_repo_root() / "outputs" / "eval",
        help="IMAGINAIRE_OUTPUT_ROOT for eval outputs.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Validation dataset root containing video/ and instructions/. Overrides EPISODE_IMAGE_EDIT_DATASET_PATH.",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Eval job name. Defaults to checkpoint-derived name.",
    )
    parser.add_argument(
        "--max-val-iter",
        default="50",
        help=(
            "Override trainer.max_val_iter. The first-frame val set has one sample per episode, so set this "
            ">= the episode count to cover the whole set (e.g. 550 for the place_a2b_left_combined LeRobot dataset; "
            "the default 50 covers only the first 50 episodes). Use 'none' for unbounded (loops the val stream). "
            "Omit to keep the TOML value."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        default=os.environ.get("WANDB_MODE", "offline"),
        choices=["online", "offline", "disabled"],
        help="W&B mode for the eval run.",
    )
    parser.add_argument(
        "--full-callbacks",
        action="store_true",
        help="Keep the full training callback stack instead of the lean eval callback subset.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation-loss forwards and run only generated-image evaluation.",
    )
    parser.add_argument(
        "--generate-reasoner-text",
        action="store_true",
        help=(
            "Autoregressively decode the reasoner response from the original episode prompt for "
            "every generated sample and write it beside the image outputs. Requires "
            "reasoner_gen_prompt_only."
        ),
    )
    parser.add_argument(
        "--reasoner-max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of autoregressively generated reasoner tokens per sample.",
    )
    parser.add_argument(
        "--reasoner-system-prompt",
        default=_SYSTEM_PROMPT_IMAGE_EDITING,
        help="System prompt used for autoregressive reasoner-text generation.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help=(
            "Sample EMA weights via a full fp32 net_ema copy (~2x weight memory, the legacy path). "
            "Default off: net_ema is disabled and its EMA weights are loaded into the single bf16 net."
        ),
    )
    parser.add_argument(
        "--max-gen-iter",
        type=int,
        default=50,
        help=(
            "Number of validation batches to sample per rank for generated-image metrics. The first-frame "
            "val set has one sample per episode; set this >= episode count to cover the whole set (e.g. 550 "
            "for place_a2b_left_combined; the default 50 covers only the first 50 episodes). Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-gen-iter-total",
        type=int,
        default=None,
        help=(
            "Exact number of generated validation samples across all ranks. This is partitioned "
            "without overlap across torchrun ranks and takes precedence over --max-gen-iter. Use "
            "-1 to derive the exact count from the selected map-style evaluation dataset."
        ),
    )
    parser.add_argument(
        "--max-val-iter-total",
        type=int,
        default=None,
        help=(
            "Exact number of validation batches across all ranks. This is partitioned without "
            "overlap and takes precedence over --max-val-iter."
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=35,
        help="Diffusion sampling steps for generated images.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.5,
        help="Classifier-free guidance for generated images.",
    )
    parser.add_argument(
        "--shift", type=float, default=5.0, help="Sampler shift for generated images."
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=80.0,
        help="Sampler sigma_max for generated images.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=1234,
        help="Base seed for deterministic eval generation.",
    )
    parser.add_argument(
        "--wandb-images",
        type=int,
        default=4,
        help="Maximum rank-0 generated grids to log to W&B offline for each checkpoint.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra-style overrides, e.g. trainer.max_val_iter=20.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    launch(parse_args())
