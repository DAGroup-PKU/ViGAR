"""CPU-only raw expert projection audit; NOT a training-alignment receipt."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from fixed_ee_roi import VIEWS, ee_point, project, rectangular_canvas


def read_rgb(dataset, frame):
    raw = dataset[frame]
    if np.asarray(raw).ndim == 3:
        raise ValueError('Unencoded RGB layout needs explicit channel convention')
    rgb = cv2.imdecode(np.frombuffer(bytes(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError('Cannot decode expert image')
    # RoboTwin pkl2hdf5.images_encoding passes SAPIEN RGB directly to
    # cv2.imencode without an RGB->BGR conversion. The inverse imdecode
    # therefore restores the original RGB array; another channel flip is wrong.
    return rgb


def export(path, frame, out, offset):
    with h5py.File(path, 'r') as f:
        count = len(f['endpose/left_endpose'])
        frame = frame if frame >= 0 else count + frame
        if not 0 <= frame < count:
            raise ValueError('Frame outside episode')
        poses = {a: f[f'endpose/{a}_endpose'][frame] for a in ('left', 'right')}
        points = {a: ee_point(p, offset) for a, p in poses.items()}
        views, rgb_views = {}, {}
        for name, (camera, *_layout) in VIEWS.items():
            group = f[f'observation/{camera}']
            rgb = read_rgb(group['rgb'], frame)
            e, k = group['extrinsic_cv'][frame], group['intrinsic_cv'][frame]
            views[name] = dict(source_hw=list(rgb.shape[:2]),
                ee_pixels={a: project(p, e, k) for a, p in points.items()},
                extrinsic_cv=e.tolist(), intrinsic_cv=k.tolist())
            rgb_views[name] = rgb
        # Same two-stage resize as the actual concat converter.
        head = rgb_views['head']; h, w = head.shape[:2]
        bottom = cv2.hconcat([cv2.resize(rgb_views[n], (w//2, h//2), interpolation=cv2.INTER_LINEAR)
                            for n in ('left_wrist', 'right_wrist')])
        canvas = cv2.resize(cv2.vconcat([head, bottom]), (320, 384), interpolation=cv2.INTER_LINEAR)
        mask, boxes = rectangular_canvas(views)
    out.mkdir(parents=True, exist_ok=False)
    preview = canvas.copy()
    for b in boxes:
        x0, y0, x1, y1 = b['xyxy']
        color = (255, 70, 40) if b['arm'] == 'left' else (40, 220, 255)
        cv2.rectangle(preview, (x0, y0), (x1-1, y1-1), color, 1)
        cx, cy = [int(round(v)) for v in b['center']]
        cv2.drawMarker(preview, (cx, cy), color, cv2.MARKER_CROSS, 9, 1)
    for filename, image in [('gt.png', canvas), ('overlay.png', preview)]:
        if not cv2.imwrite(str(out/filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise IOError(filename)
    cv2.imwrite(str(out/'mask.png'), (mask.numpy()*255).astype(np.uint8))
    receipt = dict(source_hdf5=str(path), source_frame=frame, frame_count=count,
        tcp_offset_m=offset, pose_convention='world XYZ + WXYZ',
        camera_convention='target-frame extrinsic_cv, +Z forward',
        raw_image_codec='RoboTwin RGB-direct-imencode; imdecode without extra channel swap',
        views=views, boxes=boxes, roi_fraction=float(mask.mean()),
        raw_concat_sha256=hashlib.sha256(canvas.tobytes()).hexdigest(),
        alignment_verified=False,
        note='Raw expert preview only; training video identity/decoder binding still required.')
    (out/'projection.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('source_frame','roi_fraction','alignment_verified')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hdf5', type=Path, required=True)
    p.add_argument('--frame', type=int, default=-1)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tcp-offset-m', type=float, default=.12)
    a = p.parse_args()
    export(a.hdf5, a.frame, a.output, a.tcp_offset_m)
