"""Expert target-frame rectangular ROI. No inference or conditioning changes."""
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

CANVAS_HW = (384, 320)
# name: HDF camera, y, x, height, width, fixed output-pixel square side
VIEWS = {
    'head': ('head_camera', 0, 0, 256, 320, 96),
    'left_wrist': ('left_camera', 256, 0, 128, 160, 48),
    'right_wrist': ('right_camera', 256, 160, 128, 160, 48),
}


def ee_point(pose, tcp_offset_m=0.12):
    """RoboTwin _trans_endpose: TCP = saved EE + R(wxyz) @ [.12,0,0].

    Explicitly set offset=0 for a dataset already storing TCP. This convention
    was inspected in the stage-success-v7 simulator, not inferred from pixels.
    """
    p = np.asarray(pose, dtype=np.float64)
    if p.shape != (7,) or not np.isfinite(p).all():
        raise ValueError('Expected finite world XYZ + quaternion WXYZ')
    q = p[3:]
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError('Zero quaternion')
    w, x, y, z = q / norm
    local_x = np.array([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)])
    return p[:3] + float(tcp_offset_m) * local_x


def project(point, extrinsic_cv, intrinsic_cv):
    e = np.asarray(extrinsic_cv, dtype=np.float64)
    k = np.asarray(intrinsic_cv, dtype=np.float64)
    if e.shape not in ((3, 4), (4, 4)) or k.shape != (3, 3):
        raise ValueError('Expected OpenCV world-to-camera 3x4/4x4 and K 3x3')
    if not np.isfinite(e).all() or not np.isfinite(k).all():
        raise ValueError('Nonfinite calibration')
    camera = e[:3] @ np.r_[point, 1.]
    if not np.isfinite(camera).all() or camera[2] <= 1e-8:
        return None
    uvz = k @ camera
    return (uvz[:2] / uvz[2]).tolist()


def rectangular_canvas(views, sides=None):
    """Fixed windows, union of both arms, independently clipped to each view.

    A projected point is not an occlusion test. Behind-camera/out-of-frame
    points are omitted, never moved onto an image border.
    """
    if set(views) != set(VIEWS):
        raise ValueError('Three target-frame calibrated views required')
    mask = np.zeros(CANVAS_HW, dtype=np.float32)
    boxes = []
    for name, (_, oy, ox, h, w, default_side) in VIEWS.items():
        sh, sw = views[name]['source_hw']
        side = int((sides or {}).get(name, default_side))
        if min(sh, sw, side) <= 0:
            raise ValueError('Positive image dimensions/window size required')
        for arm, point in views[name]['ee_pixels'].items():
            if point is None:
                continue
            x, y = map(float, point)
            if not (0 <= x < sw and 0 <= y < sh):
                continue
            # Matches OpenCV/PyTorch half-pixel resize coordinates.
            cx, cy = (x+.5)*w/sw-.5, (y+.5)*h/sh-.5
            left, top = math.floor(cx+.5)-side//2, math.floor(cy+.5)-side//2
            x0, x1 = max(0, left), min(w, left+side)
            y0, y1 = max(0, top), min(h, top+side)
            mask[oy+y0:oy+y1, ox+x0:ox+x1] = 1
            boxes.append(dict(view=name, arm=arm, center=[ox+cx, oy+cy],
                              xyxy=[ox+x0, oy+y0, ox+x1, oy+y1]))
    return torch.from_numpy(mask), boxes


def image_digest(image):
    if image.dtype != torch.uint8 or tuple(image.shape) != (3, *CANVAS_HW):
        raise ValueError('Bind ROI before normalization: uint8 CHW 3x384x320')
    return hashlib.sha256(image.contiguous().cpu().numpy().tobytes()).hexdigest()


class BoundROIDataset:
    """Wrap the existing sampler, preserving every sampled input/GT/caption.

    records key: (absolute video path, target frame index). Each record has
    gt_chw_u8_sha256 from the ACTUAL pinned training decoder and three views.
    Unknown/misaligned metadata raises; no GT exclusion or silent resampling.
    """
    def __init__(self, base, records, sides=None):
        self.base, self.sides = base, sides
        self.offscreen_targets = 0
        self.records = {}
        for row in records:
            key = (str(Path(row['video_path']).resolve()), int(row['target_frame_index']))
            if key in self.records:
                raise ValueError(f'Duplicate ROI target: {key}')
            if row.get('alignment_verified') is not True or not row.get('alignment_evidence'):
                raise ValueError('Requires audited trajectory/frame alignment evidence')
            self.records[key] = row

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        key = (str(Path(item['__url__']).resolve()), int(item['target_frame_index']))
        row = self.records[key]
        if image_digest(item['images'][1]) != row['gt_chw_u8_sha256']:
            raise ValueError(f'ROI/decoded GT mismatch: {key}')
        mask, _ = rectangular_canvas(row['views'], self.sides)
        if not mask.any():
            # Valid geometry can put both TCPs outside all views. Keep the
            # sample with ordinary full-image loss; never invent an on-image
            # point or discard GT. Missing records/calibration still fail above.
            self.offscreen_targets += 1
        item['_loss_roi_pair'] = torch.stack([torch.zeros_like(mask), mask])
        return item
