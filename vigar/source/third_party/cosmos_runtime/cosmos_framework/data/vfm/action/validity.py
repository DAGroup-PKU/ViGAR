"""Element validity for sparse action layouts and padded time steps."""

from __future__ import annotations

import torch


def action_validity(mask: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Resolve a raw [D] or [T,D] boolean mask without changing action layout."""
    mask = torch.as_tensor(mask, device=action.device)
    if mask.dtype != torch.bool:
        raise TypeError("action_valid_mask must be boolean")
    if mask.shape == action.shape[-1:]:
        mask = mask.expand_as(action)
    if mask.shape != action.shape:
        raise ValueError(f"action_valid_mask {tuple(mask.shape)} != action {tuple(action.shape)}")
    return mask


def mask_action(action: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return action if mask is None else action.masked_fill(~action_validity(mask, action), 0)


def batch_action_validity(data_batch: dict, dense_actions: list[torch.Tensor] | None):
    """Keep optional masks aligned when a mixed batch contains non-action samples."""
    raw = data_batch.get("action_valid_mask")
    if raw is None or dense_actions is None:
        return None
    masks = raw if isinstance(raw, list) else [raw]
    masks = [m[0] if isinstance(m, list) else m for m in masks]
    original = data_batch.get("action")
    original = original if isinstance(original, list) else [original]
    original = [a[0] if isinstance(a, list) else a for a in original]
    if len(masks) != len(original):
        raise ValueError("Action masks must align with original sample slots")
    masks = [m for m, a in zip(masks, original, strict=True) if a is not None]
    if len(masks) != len(dense_actions):
        raise ValueError("Action masks must align with dense action samples")
    return [None if m is None else action_validity(m, a) for m, a in zip(masks, dense_actions, strict=True)]
