"""Physical action metrics and future-image metrics, separate from flow losses."""

import numpy as np
import torch


def physical_action_metrics(prediction, target, mask, layout):
    # Compute joint/EEF/chassis ADE/FDE, adjacent-step differences,
    # and quaternion geodesic errors. Inputs here are already physical and GT
    # is the original unclipped target, not an inverse of clipped normalization.
    from .action_metrics import _sample_metrics

    # Preserve the former per-arm metric's validation: the shared angular
    # formula otherwise turns a zero quaternion into a misleading 180 degrees.
    for group in (layout.left_eef_quat, layout.right_eef_quat):
        valid = mask[:, group].all(axis=-1)
        if np.any(np.linalg.norm(prediction[valid, group], axis=-1) < 1e-8):
            raise ValueError("Generated valid quaternion has zero norm")
    result = _sample_metrics(prediction, target, joint_mask=mask, action_layout=layout)
    dims = list(layout.gripper_dims)
    valid = mask[:, dims]
    if valid.any():
        result["gripper_mae_closedness"] = float(np.abs(prediction[:, dims] - target[:, dims])[valid].mean())
    return result


def image_metrics(prediction, target, pixel_mask=None, camera_boxes=None):
    """RGB errors in [0,1], excluding letterbox pixels and clean current frame.

    MSE/MAE reduce over channels, genuine future frames and real image pixels.
    PSNR has a 120 dB numerical ceiling. Temporal error compares adjacent future
    differences only; it does not conflate current-frame reconstruction with motion.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Image metrics require matching CTHW videos")
    if prediction.shape[1] < 1:
        raise ValueError("Image metrics require a current frame")
    pred, gt = prediction.float() / 255, target.float() / 255
    valid = torch.ones(target.shape[-2:], dtype=torch.bool) if pixel_mask is None else pixel_mask.bool()
    if not valid.any():
        raise ValueError("No real image pixels in the evaluation canvas")

    def errors(delta, mask):
        values = delta[..., mask]
        mse = values.square().mean()
        return dict(mae=float(values.abs().mean()), mse=float(mse), psnr=float(-10 * mse.clamp_min(1e-12).log10()))

    result = {}
    if target.shape[1] > 1:
        result.update({f"video_{key}": value for key, value in errors(pred[:, 1:] - gt[:, 1:], valid).items()})
    result.update({f"current_frame_{key}": value for key, value in errors(pred[:, :1] - gt[:, :1], valid).items()})
    result["video_valid_pixels"] = int(valid.sum())
    result["video_future_frames"] = target.shape[1] - 1
    if target.shape[1] > 2:
        motion_error = torch.diff(pred[:, 1:], dim=1) - torch.diff(gt[:, 1:], dim=1)
        result["video_temporal_mae"] = float(motion_error[..., valid].abs().mean())
    for camera, (y, x, h, w) in (camera_boxes or {}).items():
        camera_mask = torch.zeros_like(valid)
        camera_mask[y : y + h, x : x + w] = valid[y : y + h, x : x + w]
        if camera_mask.any() and target.shape[1] > 1:
            result.update(
                {f"video_{camera}_{key}": value for key, value in errors(pred[:, 1:] - gt[:, 1:], camera_mask).items()}
            )
    return result


def aggregate_metrics(rows):
    """Equal per-window means with independent availability counts."""
    keys = sorted({key for row in rows for key in row["metrics"]})
    counts = {key: sum(key in row["metrics"] for row in rows) for key in keys}
    mean = {key: float(np.mean([row["metrics"][key] for row in rows if key in row["metrics"]])) for key in keys}
    by_robot = {}
    for robot in sorted({row["robot_type"] for row in rows}):
        selected = [row for row in rows if row["robot_type"] == robot]
        by_robot[robot] = {
            key: float(np.mean([row["metrics"][key] for row in selected if key in row["metrics"]]))
            for key in keys
            if any(key in row["metrics"] for row in selected)
        }
    return dict(samples=rows, mean=mean, counts=counts, by_robot=by_robot)
