# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Rectified-flow matching loss (vision / action / sound modalities).

Extracted from OmniMoTModel._compute_flow_matching_loss. The loss math is
unchanged; the only structural change is that ``tensor_kwargs_fp32`` is now
passed explicitly instead of being read from ``self``.
"""

from __future__ import annotations

import torch

from cosmos_framework.model.vfm.diffusion.rectified_flow import RectifiedFlow


def compute_flow_matching_loss(
    pred: list[torch.Tensor],
    target: list[torch.Tensor],
    condition_mask: list[torch.Tensor],
    timesteps: torch.Tensor,
    has_valid_tokens: bool,
    rectified_flow: RectifiedFlow,
    tensor_kwargs_fp32: dict,
    raw_action_dim: list[torch.Tensor] | None = None,
    normalize_by_active: bool = False,
    exclude_fully_conditioned_items: bool = False,
    action_valid_mask: list[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute flow matching loss for a modality.

    Args:
        pred: Predicted velocity field (list of tensors, one per sample).
        target: Target velocity field (list of tensors, one per sample).
            Under rectified flow the target is ``v = eps - x0``.
        condition_mask: Mask where 1 = clean/conditioning, 0 = noisy/generation (list of tensors).
        timesteps: Diffusion timesteps for time weighting. Shape [B,1] for
            base/teacher_forcing (all frames share one timestep) or [B,T_max]
            for diffusion_forcing (per-frame independent timesteps). Time weights
            are applied per-frame before averaging, so non-uniform weight functions
            are handled correctly.
        has_valid_tokens: Whether this modality has valid noisy tokens.
        rectified_flow: The rectified flow object for time weighting.
        tensor_kwargs_fp32: Dict of dtype/device kwargs forwarded to
            ``rectified_flow.train_time_weight``.
        normalize_by_active: When True, normalize per-instance loss by the count of
            active (noisy) elements rather than all elements. Preserves the
            ``sum / active_count`` semantics needed for distillation critics where
            conditioned frames contribute no signal and should not dilute the
            denominator.
        exclude_fully_conditioned_items: Exclude fully clean items from the
            final scalar mean while preserving their zero entries in the
            per-instance output. This keeps callback tensors aligned while a
            clean goal/source item cannot dilute its generated target's loss.

    Returns:
        tuple: A tuple containing two elements:
            - Flow matching loss (or dummy loss for gradient consistency).
            - Per-instance loss (or dummy loss for gradient consistency).
    """
    if not has_valid_tokens:
        # Dummy loss to maintain backward graph consistency across ranks
        dummy_loss = 0.0 * sum(p.sum() for p in pred)
        return dummy_loss, dummy_loss.unsqueeze(0)  # make per-instance loss 1-D

    # condition_mask[i] is T-first with trailing singletons: [T,1,1] vision, [T,1] action.
    # tw_i gets the same shape so w(σ_t) broadcasts element-wise over non-T dims.
    per_instance_losses = []
    per_instance_weighted_losses = []
    active_item_indexes = (
        [i for i, mask in enumerate(condition_mask) if torch.any(mask < 1)]
        if exclude_fully_conditioned_items
        else list(range(len(pred)))
    )

    for i in range(len(pred)):
        T_i = condition_mask[i].shape[0]
        valid_i = action_valid_mask[i] if action_valid_mask is not None else None
        if valid_i is not None:
            valid_i = valid_i.to(device=pred[i].device, dtype=torch.bool)
            if valid_i.shape != pred[i].shape:
                raise ValueError("Action loss validity must match [T,D] predictions")
            # Clear invalid values before arithmetic: 0 * NaN still produces NaN,
            # and masking after squaring can also leave NaN gradients.
            sqerr_i = (pred[i].masked_fill(~valid_i, 0) - target[i].masked_fill(~valid_i, 0)) ** 2
        else:
            sqerr_i = (pred[i] - target[i]) ** 2  # vision:[C,T,H,W]  action/sound:[T,D]
        noisy_mask_i = 1.0 - condition_mask[i]  # vision:[T,1,1]  action/sound:[T,1]
        if raw_action_dim is not None and raw_action_dim[i] is not None:
            sqerr_i = sqerr_i[:, : raw_action_dim[i]]
            if valid_i is not None:
                valid_i = valid_i[:, : raw_action_dim[i]]
        if valid_i is not None:
            noisy_mask_i = noisy_mask_i * valid_i
        use_active_count = normalize_by_active or valid_i is not None
        if use_active_count:
            active_count = (noisy_mask_i.sum() * (sqerr_i.numel() // noisy_mask_i.numel())).clamp(min=1)
            per_instance_losses.append((sqerr_i * noisy_mask_i).sum() / active_count)  # []
        else:
            per_instance_losses.append((sqerr_i * noisy_mask_i).mean())  # []

        ts_i = timesteps[i, :T_i] if timesteps.dim() > 1 else timesteps[i]  # DF:[T_i]  TF:[1]
        tw_i = rectified_flow.train_time_weight(ts_i, tensor_kwargs_fp32)  # DF:[T_i]  TF:[1]
        tw_i = tw_i.reshape(-1, *([1] * (condition_mask[i].ndim - 1)))  # vision:[T_i,1,1]  action/sound:[T_i,1]
        if use_active_count:
            per_instance_weighted_losses.append((sqerr_i * tw_i * noisy_mask_i).sum() / active_count)
        else:
            per_instance_weighted_losses.append((sqerr_i * tw_i * noisy_mask_i).mean())

    per_instance_loss = torch.stack(per_instance_losses)  # [B]
    per_instance_weighted_loss = torch.stack(per_instance_weighted_losses)  # [B]
    if active_item_indexes:
        weighted_loss = per_instance_weighted_loss[active_item_indexes].mean()
    else:
        weighted_loss = 0.0 * sum(p.sum() for p in pred)
    return (
        weighted_loss,  # []
        per_instance_loss,  # [B]
    )
