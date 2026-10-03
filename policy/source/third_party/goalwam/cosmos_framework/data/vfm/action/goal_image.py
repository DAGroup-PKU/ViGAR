# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Goal-image conditioning helpers shared by training and serving.

The goal image (last frame of a training episode, or the terminal frame of the
rule-based expert at eval time) conditions the action policy through the
REASONER (AR) branch: it is encoded by the frozen Qwen3-VL vision tower and its
embeddings are scattered at ``<|image_pad|>`` placeholder positions inside the
AR text sequence, exactly like Qwen3-VL multimodal understanding. The trained
generation tower reads it via the existing joint attention.

This module owns the two artifacts every goal-conditioned sample needs:

* the placeholder string prepended to the caption (reserves the token slots),
* the ViT-ready ``pixel_values``/``image_grid_thw`` tensors (the slot content).

Sizing is pinned per layout (no smart-resize) so training and eval preprocess
bit-identically. The image-processor settings mirror the Qwen3-VL-8B
``preprocessor_config.json`` (``patch_size=16``, ``temporal_patch_size=2``,
``merge_size=2``, ``image_mean=image_std=0.5``); we construct the processor
from explicit kwargs so dataloader workers never touch the HF cache.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

GOAL_COND_MODES = ("text_cond", "goal_frame_cond", "text_goal_frame_cond")
GOAL_LAYOUTS = ("concat", "concat_320x384", "cam_high")

# Fixed goal-image sizes per layout, (H, W). concat matches the RoboTwin
# 3-camera inverted-T training canvas; concat_320x384 is the landscape AgiBot
# LH variant of the same layout (16:9 cameras); cam_high is the head camera
# resized. All are multiples of the ViT effective stride (patch 16 x merge 2 = 32).
GOAL_LAYOUT_HW = {
    "concat": (384, 320),
    "concat_320x384": (320, 384),
    "cam_high": (256, 320),
}

# Qwen3-VL special-token ids (asserted against the live tokenizer at model
# setup — see OmniMoTModel.set_up_tokenizers).
GOAL_IMAGE_PAD_TOKEN_ID = 151655  # <|image_pad|>
_VISION_START = "<|vision_start|>"
_VISION_END = "<|vision_end|>"
_IMAGE_PAD = "<|image_pad|>"

_PATCH_SIZE = 16
_TEMPORAL_PATCH_SIZE = 2
_MERGE_SIZE = 2

_image_processor = None


def cond_uses_goal_image(cond: str) -> bool:
    """Whether the conditioning mode consumes a goal image."""
    if cond not in GOAL_COND_MODES:
        raise ValueError(f"Unknown cond={cond!r}; expected one of {GOAL_COND_MODES}")
    return cond in ("goal_frame_cond", "text_goal_frame_cond")


def validate_goal_layout(goal_layout: str) -> str:
    if goal_layout not in GOAL_LAYOUTS:
        raise ValueError(
            f"Unknown goal_layout={goal_layout!r}; expected one of {GOAL_LAYOUTS}"
        )
    return goal_layout


def _get_image_processor():
    global _image_processor
    if _image_processor is None:
        from transformers import Qwen2VLImageProcessor

        # Explicit kwargs pin the Qwen3-VL-8B preprocessor_config.json values;
        # NOTE image_mean/std are 0.5 (Qwen3-VL), NOT the class's CLIP defaults.
        _image_processor = Qwen2VLImageProcessor(
            do_resize=False,
            patch_size=_PATCH_SIZE,
            temporal_patch_size=_TEMPORAL_PATCH_SIZE,
            merge_size=_MERGE_SIZE,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
    return _image_processor


def expected_goal_grid(goal_layout: str) -> tuple[int, int, int]:
    """Pre-merge ViT patch grid (t, h, w) for a layout."""
    h, w = GOAL_LAYOUT_HW[validate_goal_layout(goal_layout)]
    return (1, h // _PATCH_SIZE, w // _PATCH_SIZE)


def expected_goal_token_count(goal_layout: str) -> int:
    """Number of merged ViT tokens (= placeholder count) for a layout."""
    t, gh, gw = expected_goal_grid(goal_layout)
    return (t * gh * gw) // (_MERGE_SIZE * _MERGE_SIZE)


def preprocess_goal_image(
    goal_chw_uint8: torch.Tensor, goal_layout: str
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """uint8 [3,H,W] goal frame -> ViT inputs.

    Returns:
        pixel_values: [N_patches, 1536] float32 (flattened 16x16x3x2 patches).
        grid_thw: [1, 3] long, PRE-merge patch grid.
        n_merged_tokens: number of ``<|image_pad|>`` placeholders to reserve.
    """
    if goal_chw_uint8.ndim != 3 or goal_chw_uint8.shape[0] != 3:
        raise ValueError(
            f"goal image must be [3,H,W], got {tuple(goal_chw_uint8.shape)}"
        )
    if goal_chw_uint8.dtype != torch.uint8:
        raise ValueError(f"goal image must be uint8, got {goal_chw_uint8.dtype}")

    target_hw = GOAL_LAYOUT_HW[validate_goal_layout(goal_layout)]
    img = goal_chw_uint8
    if tuple(img.shape[-2:]) != target_hw:
        img = (
            F.interpolate(
                img.unsqueeze(0).float(),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .clamp(0, 255)
            .to(torch.uint8)
        )

    # HWC uint8 numpy for the HF image processor (do_resize=False → size untouched).
    out = _get_image_processor()(
        images=[img.permute(1, 2, 0).numpy()], return_tensors="pt"
    )
    pixel_values = out["pixel_values"].to(torch.float32)  # [N_patches, 1536]
    grid_thw = out["image_grid_thw"].to(torch.long)  # [1, 3]

    exp_grid = expected_goal_grid(goal_layout)
    got_grid = tuple(int(v) for v in grid_thw[0].tolist())
    if got_grid != exp_grid:
        raise RuntimeError(
            f"goal image grid mismatch for layout={goal_layout!r}: got {got_grid}, expected {exp_grid} "
            "(transformers image-processor behavior drift?)"
        )
    n_patches = got_grid[0] * got_grid[1] * got_grid[2]
    if pixel_values.shape != (
        n_patches,
        3 * _TEMPORAL_PATCH_SIZE * _PATCH_SIZE * _PATCH_SIZE,
    ):
        raise RuntimeError(
            f"goal pixel_values shape {tuple(pixel_values.shape)} does not match grid {got_grid}"
        )
    n_merged_tokens = n_patches // (_MERGE_SIZE * _MERGE_SIZE)
    return pixel_values, grid_thw, n_merged_tokens


def goal_placeholder_string(n_tokens: int) -> str:
    """Placeholder block reserving ``n_tokens`` reasoner slots for the goal image."""
    return _VISION_START + _IMAGE_PAD * int(n_tokens) + _VISION_END


def merged_grid(grid_thw: torch.Tensor) -> tuple[int, int, int]:
    """PRE-merge [1,3] grid -> post-merge (t, h, w) used for MRoPE positions."""
    t, h, w = (int(v) for v in grid_thw[0].tolist())
    return (t, h // _MERGE_SIZE, w // _MERGE_SIZE)
