# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action normalization helpers."""

import json
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.utils import log


def load_action_stats(
    stats_path: str,
    stats_key: str | None = None,
) -> dict[str, np.ndarray]:
    """Load pre-computed normalization stats from a JSON file.

    ``stats_key`` selects one named block from a bundled stats file.  The
    default keeps compatibility with the original flat action-stats schema.
    """
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Action normalization stats not found at {stats_path}."
        )
    log.info(f"Loading action normalization stats from {stats_path}")
    with path.open("r") as f:
        raw = json.load(f)
    if stats_key is not None:
        selected = raw.get(stats_key) if isinstance(raw, dict) else None
        if not isinstance(selected, dict):
            raise ValueError(
                f"Normalization stats key {stats_key!r} was not found in {stats_path}"
            )
        raw = selected
    stat_keys = {"mean", "std", "min", "max", "q01", "q99"}
    stats = {
        key: np.array(value, dtype=np.float32)
        for key, value in raw.items()
        if key in stat_keys
    }
    if not stats:
        location = f"{stats_path}" if stats_key is None else f"{stats_path}[{stats_key}]"
        raise ValueError(
            f"No normalization statistics (mean/std/min/max/q01/q99) in {location}; "
            "is this a bundled stats file that needs a stats_key?"
        )
    return stats


def normalize_action(
    action: torch.Tensor,
    method: str,
    stats: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Normalize action tensor."""
    if method == "quantile":
        q01, q99 = stats["q01"], stats["q99"]
        denom = (q99 - q01).clamp(min=1e-8)
        return 2.0 * (action - q01) / denom - 1.0
    if method == "meanstd":
        return ((action - stats["mean"]) / stats["std"].clamp(min=1e-8)).clamp(
            -5.0, 5.0
        )
    if method == "minmax":
        lo, hi = stats["min"], stats["max"]
        denom = (hi - lo).clamp(min=1e-8)
        return 2.0 * (action - lo) / denom - 1.0
    raise ValueError(f"Unknown normalization method: {method!r}")


def denormalize_action(
    action: torch.Tensor,
    method: str,
    stats: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Denormalize action tensor."""
    if method == "quantile":
        q01, q99 = stats["q01"], stats["q99"]
        return 0.5 * (action + 1.0) * (q99 - q01) + q01
    if method == "meanstd":
        return action * stats["std"] + stats["mean"]
    if method == "minmax":
        lo, hi = stats["min"], stats["max"]
        return 0.5 * (action + 1.0) * (hi - lo) + lo
    raise ValueError(f"Unknown normalization method: {method!r}")
