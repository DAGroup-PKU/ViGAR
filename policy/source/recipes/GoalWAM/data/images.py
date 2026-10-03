"""Aspect-preserving camera composition for the native GoalWAM video stream."""

import math

import torch
import torch.nn.functional as F


CAMERA_KEYS = {
    "torso": "observation.images.cam_torso",
    "head": "observation.images.cam_high",
    "left": "observation.images.cam_left_wrist",
    "right": "observation.images.cam_right_wrist",
}


def validate_goal_image_composition(mode, enable_cameras):
    if mode != "multi_view":
        raise ValueError("goal_image_composition must be multi_view")
    return mode


def valid_camera_frames(frames):
    """Missing/empty/malformed RGB is unavailable; black RGB is still valid."""
    return (
        isinstance(frames, torch.Tensor)
        and frames.ndim == 4
        and frames.shape[0] > 0
        and frames.shape[1] == 3
        and min(frames.shape[-2:]) > 0
        and (frames.dtype == torch.uint8 or (frames.is_floating_point() and bool(torch.isfinite(frames).all())))
    )


def camera_layout(img_size, enable_cameras):
    """Full-size torso/head rows above a row of half-size wrist views.

    HEIGHT/WIDTH describe a full-size camera cell. Center-pad the final canvas
    to multiples of 32 (Wan stride 16 followed by native 2x2 patchification).
    Disabled cameras contribute no observations; their unused cells stay black.
    """
    h, w = img_size
    boxes, y = {}, 0
    for name in ("torso", "head"):
        if name in enable_cameras:
            boxes[name] = [y, 0, h, w]
            y += h
    for name, x in (("left", 0), ("right", w // 2)):
        if name in enable_cameras:
            boxes[name] = [y, x, h // 2, w // 2]
    if any(name in enable_cameras for name in ("left", "right")):
        y += h // 2
    height, width = math.ceil(y / 32) * 32, math.ceil(w / 32) * 32
    for box in boxes.values():
        box[0] += (height - y) // 2
        box[1] += (width - w) // 2
    return (height, width), boxes


def letterbox(frames, size):
    """TCHW uint8 -> centered black letterbox, plus a real-pixel mask."""
    h, w = size
    scale = min(h / frames.shape[-2], w / frames.shape[-1])
    rh = min(h, max(1, round(frames.shape[-2] * scale)))
    rw = min(w, max(1, round(frames.shape[-1] * scale)))
    resized = F.interpolate(frames.float(), size=(rh, rw), mode="bilinear", align_corners=False)
    top, left = (h - rh) // 2, (w - rw) // 2
    result = F.pad(resized.round().clamp(0, 255).byte(), (left, w - rw - left, top, h - rh - top))
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[top : top + rh, left : left + rw] = True
    return result, mask


def compose_cameras(views, img_size, enable_cameras):
    """Compose configured cells; unavailable wrists stay black and invalid."""
    shape, boxes = camera_layout(img_size, enable_cameras)
    for name in boxes:
        if name not in ("left", "right") and not valid_camera_frames(views.get(name)):
            raise ValueError(f"Required {name} camera images are missing or invalid")
    first = next((frames for name, frames in views.items() if name in boxes and valid_camera_frames(frames)), None)
    if first is None:
        raise ValueError("No valid camera images to compose")
    canvas = torch.zeros(first.shape[0], 3, *shape, dtype=torch.uint8)
    valid = torch.zeros(shape, dtype=torch.bool)
    for name, (y, x, h, w) in boxes.items():
        value = views.get(name)
        if not valid_camera_frames(value):
            if name in ("left", "right"):
                continue
            raise ValueError(f"Required {name} camera images are missing or invalid")
        if value.shape[0] != first.shape[0]:
            raise ValueError("Camera frame counts must match")
        frames, mask = letterbox(value, (h, w))
        canvas[:, :, y : y + h, x : x + w] = frames
        valid[y : y + h, x : x + w] = mask
    return canvas, valid, boxes


def compose_goal_image(views, img_size, enable_cameras, mode="multi_view", *, resolution="384x320"):
    """Compose a multi-view goal with the same layout as the observation canvas."""
    validate_goal_image_composition(mode, enable_cameras)
    if img_size is not None:
        return compose_cameras(views, img_size, enable_cameras)
    return compose_legacy_cameras(views, resolution)


def compose_legacy_cameras(views, resolution):
    """Historical stretch-to-cell composition, with black unavailable wrists."""
    # Legacy profiles stretch each view into the original 2:1 mosaic cells.
    head = views.get("head")
    if not valid_camera_frames(head):
        raise ValueError("Required head camera image is missing or invalid")
    height, width = map(int, resolution.split("x"))
    shape = (height, width)
    boxes = {
        "head": [0, 0, 2 * height // 3, width],
        "left": [2 * height // 3, 0, height // 3, width // 2],
        "right": [2 * height // 3, width // 2, height // 3, width // 2],
    }
    canvas = torch.zeros(head.shape[0], 3, *shape, dtype=torch.uint8)
    mask = torch.zeros(shape, dtype=torch.bool)
    for name, (y, x, h, w) in boxes.items():
        frames = views.get(name)
        if not valid_camera_frames(frames):
            continue
        if frames.shape[0] != head.shape[0]:
            raise ValueError("Camera frame counts must match")
        canvas[:, :, y : y + h, x : x + w] = (
            F.interpolate(frames.float(), size=(h, w), mode="bilinear", align_corners=False)
            .round()
            .clamp(0, 255)
            .byte()
        )
        mask[y : y + h, x : x + w] = True
    return canvas, mask, boxes


class PreparedVideo:
    """Keep the recipe's already letterboxed canvas through native transforms."""

    def __call__(self, sample, resolution):
        h, w = sample["video"].shape[-2:]
        if h % 32 or w % 32:
            raise ValueError("Prepared GoalWAM canvas edges must be divisible by 32")
        sample["image_size"] = torch.tensor([h, w, h, w], dtype=torch.float32)
        return sample
