import math
import numpy as np
import torch

CANVAS_HW = (384, 320)
VIEWS = {
    "head": ("head_camera", 0, 0, 256, 320, 96),
    "left_wrist": ("left_camera", 256, 0, 128, 160, 48),
    "right_wrist": ("right_camera", 256, 160, 128, 160, 48),
}


def ee_point(pose, tcp_offset_m=0.12):
    p = np.asarray(pose, dtype=np.float64)
    if p.shape != (7,) or not np.isfinite(p).all():
        raise ValueError("Expected finite world XYZ + quaternion WXYZ")
    q = p[3:]
    norm = np.linalg.norm(q)
    if norm < 1e-08:
        raise ValueError("Zero quaternion")
    (w, x, y, z) = q / norm
    local_x = np.array(
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]
    )
    return p[:3] + float(tcp_offset_m) * local_x


def project(point, extrinsic_cv, intrinsic_cv):
    e = np.asarray(extrinsic_cv, dtype=np.float64)
    k = np.asarray(intrinsic_cv, dtype=np.float64)
    if e.shape not in ((3, 4), (4, 4)) or k.shape != (3, 3):
        raise ValueError("Expected OpenCV world-to-camera 3x4/4x4 and K 3x3")
    if not np.isfinite(e).all() or not np.isfinite(k).all():
        raise ValueError("Nonfinite calibration")
    camera = e[:3] @ np.r_[point, 1.0]
    if not np.isfinite(camera).all() or camera[2] <= 1e-08:
        return None
    uvz = k @ camera
    return (uvz[:2] / uvz[2]).tolist()


def rectangular_canvas(views, sides=None):
    if set(views) != set(VIEWS):
        raise ValueError("Three target-frame calibrated views required")
    mask = np.zeros(CANVAS_HW, dtype=np.float32)
    boxes = []
    for name, (_, oy, ox, h, w, default_side) in VIEWS.items():
        (sh, sw) = views[name]["source_hw"]
        side = int((sides or {}).get(name, default_side))
        if min(sh, sw, side) <= 0:
            raise ValueError("Positive image dimensions/window size required")
        for arm, point in views[name]["ee_pixels"].items():
            if point is None:
                continue
            (x, y) = map(float, point)
            if not (0 <= x < sw and 0 <= y < sh):
                continue
            (cx, cy) = ((x + 0.5) * w / sw - 0.5, (y + 0.5) * h / sh - 0.5)
            (left, top) = (
                math.floor(cx + 0.5) - side // 2,
                math.floor(cy + 0.5) - side // 2,
            )
            (x0, x1) = (max(0, left), min(w, left + side))
            (y0, y1) = (max(0, top), min(h, top + side))
            mask[oy + y0 : oy + y1, ox + x0 : ox + x1] = 1
            boxes.append(
                dict(
                    view=name,
                    arm=arm,
                    center=[ox + cx, oy + cy],
                    xyxy=[ox + x0, oy + y0, ox + x1, oy + y1],
                )
            )
    return (torch.from_numpy(mask), boxes)
