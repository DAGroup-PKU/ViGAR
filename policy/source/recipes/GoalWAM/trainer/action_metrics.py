"""Physical 49D action metrics used by GoalWAM evaluation."""

from typing import Dict

import numpy as np
import torch

from ..data.action_layout import ActionLayout, max_required_dim


def _quat_geodesic_deg_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Shortest sign-invariant geodesic angle between xyzw quaternions."""
    pred_norm = np.linalg.norm(pred, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    pred_unit = pred / np.maximum(pred_norm, eps)
    target_unit = target / np.maximum(target_norm, eps)
    dot = np.abs(np.sum(pred_unit * target_unit, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def _masked_mean_abs_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    return float(np.abs(pred - target)[mask].mean())


def _action_valid_mask_np(joint_mask, *, timesteps: int, action_dim: int) -> np.ndarray:
    """Expand the loader's full action-validity mask to ``(T, D)``."""
    if joint_mask is None:
        return np.ones((timesteps, action_dim), dtype=bool)
    if isinstance(joint_mask, torch.Tensor):
        joint_mask = joint_mask.detach().cpu().numpy()
    mask = np.asarray(joint_mask, dtype=bool)
    if mask.ndim == 1:
        mask = np.broadcast_to(mask, (timesteps, mask.shape[0]))
    elif mask.ndim != 2:
        raise ValueError(f"joint_mask must be (D,) or (T,D), got {mask.shape}.")
    if mask.shape[0] < timesteps:
        raise ValueError(f"joint_mask has {mask.shape[0]} timesteps, expected at least {timesteps}.")
    mask = mask[:timesteps]
    if mask.shape[1] == action_dim:
        return mask
    expanded = np.zeros((timesteps, action_dim), dtype=bool)
    copied_dim = min(mask.shape[1], action_dim)
    expanded[:, :copied_dim] = mask[:, :copied_dim]
    return expanded


def _sample_metrics(
    pred,
    target,
    joint_mask=None,
    *,
    action_layout: ActionLayout,
) -> Dict[str, float]:
    """Compute masked generation metrics from physical action arrays."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.shape[0] == 0:
        return {}

    metrics = {}
    action_valid = _action_valid_mask_np(
        joint_mask,
        timesteps=pred.shape[0],
        action_dim=pred.shape[-1],
    )

    joint_dims = np.asarray([*range(0, 7), *range(8, 15), *range(16, 23)], dtype=np.int64)
    if pred.shape[-1] > int(joint_dims.max()):
        joints_pred = pred[:, joint_dims]
        joints_target = target[:, joint_dims]
        joints_valid = action_valid[:, joint_dims]
        ade = _masked_mean_abs_np(joints_pred, joints_target, joints_valid)
        fde = _masked_mean_abs_np(joints_pred[-1], joints_target[-1], joints_valid[-1])
        if ade is not None:
            metrics["joint_ade"] = ade
        if fde is not None:
            metrics["joint_fde"] = fde
        if pred.shape[0] >= 2:
            vel_valid = joints_valid[1:] & joints_valid[:-1]
            vel = _masked_mean_abs_np(np.diff(joints_pred, axis=0), np.diff(joints_target, axis=0), vel_valid)
            if vel is not None:
                metrics["joint_vel_diff"] = vel
    if pred.shape[-1] >= max_required_dim(action_layout.left_eef_pos, action_layout.right_eef_pos):
        eef_pred = np.concatenate([pred[:, action_layout.left_eef_pos], pred[:, action_layout.right_eef_pos]], axis=-1)
        eef_target = np.concatenate(
            [target[:, action_layout.left_eef_pos], target[:, action_layout.right_eef_pos]], axis=-1
        )
        eef_valid = np.concatenate(
            [
                action_valid[:, action_layout.left_eef_pos],
                action_valid[:, action_layout.right_eef_pos],
            ],
            axis=-1,
        )
        ade = _masked_mean_abs_np(eef_pred, eef_target, eef_valid)
        fde = _masked_mean_abs_np(eef_pred[-1], eef_target[-1], eef_valid[-1])
        if ade is not None:
            metrics["eef_pose_ade"] = ade
        if fde is not None:
            metrics["eef_pose_fde"] = fde
        if pred.shape[0] >= 2:
            vel_valid = eef_valid[1:] & eef_valid[:-1]
            vel = _masked_mean_abs_np(np.diff(eef_pred, axis=0), np.diff(eef_target, axis=0), vel_valid)
            if vel is not None:
                metrics["eef_pose_vel_diff"] = vel
    if pred.shape[-1] >= max_required_dim(action_layout.left_eef_quat, action_layout.right_eef_quat):
        eef_quat_pred = np.stack(
            [pred[:, action_layout.left_eef_quat], pred[:, action_layout.right_eef_quat]], axis=-2
        )
        eef_quat_target = np.stack(
            [target[:, action_layout.left_eef_quat], target[:, action_layout.right_eef_quat]], axis=-2
        )
        eef_rot_error_deg = _quat_geodesic_deg_np(eef_quat_pred, eef_quat_target)
        eef_quat_valid = np.stack(
            [
                action_valid[:, action_layout.left_eef_quat].all(axis=-1),
                action_valid[:, action_layout.right_eef_quat].all(axis=-1),
            ],
            axis=-1,
        )
        if eef_quat_valid.any():
            metrics["eef_rot_ade"] = float(eef_rot_error_deg[eef_quat_valid].mean())
        if eef_quat_valid[-1].any():
            metrics["eef_rot_fde"] = float(eef_rot_error_deg[-1][eef_quat_valid[-1]].mean())
    if pred.shape[-1] >= max_required_dim(action_layout.chassis_xy):
        chassis_pred = pred[:, action_layout.chassis_xy]
        chassis_target = target[:, action_layout.chassis_xy]
        chassis_valid = action_valid[:, action_layout.chassis_xy]
        pose_ade = _masked_mean_abs_np(
            chassis_pred,
            chassis_target,
            chassis_valid,
        )
        if pose_ade is not None:
            metrics["chassis_pose_ade"] = pose_ade
        if pred.shape[0] >= 2:
            vel_valid = chassis_valid[1:] & chassis_valid[:-1]
            vel = _masked_mean_abs_np(
                np.diff(chassis_pred, axis=0),
                np.diff(chassis_target, axis=0),
                vel_valid,
            )
            if vel is not None:
                metrics["chassis_vel_diff"] = vel
    if pred.shape[-1] >= max_required_dim(action_layout.chassis_angle):
        chassis_w_diff = _masked_mean_abs_np(
            pred[:, action_layout.chassis_angle],
            target[:, action_layout.chassis_angle],
            action_valid[:, action_layout.chassis_angle],
        )
        if chassis_w_diff is not None:
            metrics["chassis_w_diff"] = chassis_w_diff

    return metrics
