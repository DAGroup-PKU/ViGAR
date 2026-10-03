"""Training preparation only: loss-side expert ROIs, never model conditioning.

The optional adapter delegates timestep/masking/reduction to the pinned Cosmos
loss. It changes only the velocity residual's spatial weighting. No trainer,
GPU allocation, or live monkey-patching occurs on import.
"""
import hashlib

import torch
import torch.nn.functional as F

CANVAS_HW = (384, 320)
VIEWS = {'head': (0, 0, 256, 320), 'left_wrist': (256, 0, 128, 160),
         'right_wrist': (256, 160, 128, 160)}


def project_world_point(point, world_to_camera_opencv, intrinsics):
    """Explicit OpenCV camera convention: +Z forward; never guess SAPIEN axes."""
    p = torch.as_tensor(point, dtype=torch.float64)
    transform = torch.as_tensor(world_to_camera_opencv, dtype=torch.float64)
    k = torch.as_tensor(intrinsics, dtype=torch.float64)
    if p.shape != (3,) or transform.shape != (4, 4) or k.shape != (3, 3):
        raise ValueError('Expected point[3], calibrated transform[4,4], K[3,3]')
    camera = transform @ torch.cat([p, p.new_ones(1)])
    if not torch.isfinite(camera).all() or camera[2] <= 0:
        return None
    q = k @ camera[:3]
    return (q[:2] / q[2]).float()


def ee_canvas(views):
    """Gaussian windows in each source view, resized with half-pixel mapping.

Each view: source_hw, ee_pixels (expert points from target/contact window),
sigma_fraction (default .075 of min output dimension). Invisible/offscreen
points are omitted, not clamped onto a border. Windows cannot cross view seams.
"""
    canvas = torch.zeros(CANVAS_HW, dtype=torch.float32)
    if set(views) != set(VIEWS):
        raise ValueError('Three calibrated views required')
    for name, (oy, ox, h, w) in VIEWS.items():
        meta = views[name]
        sh, sw = meta['source_hw']
        if min(sh, sw) <= 0:
            raise ValueError('Invalid source dimensions')
        sigma = float(meta.get('sigma_fraction', .075)) * min(h, w)
        if sigma <= 0:
            raise ValueError('Positive sigma required')
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        tile = torch.zeros(h, w)
        for point in meta['ee_pixels']:
            x, y = map(float, point)
            if not (0 <= x < sw and 0 <= y < sh):
                continue
            cx, cy = (x+.5)*w/sw-.5, (y+.5)*h/sh-.5
            tile = torch.maximum(tile, torch.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma**2)))
        canvas[oy:oy+h, ox:ox+w] = tile
    return canvas


def union_roi(ee, object_input, object_target, target_area):
    """Expert visible masks cover both old/new object locations and landing area."""
    masks = [torch.as_tensor(m, dtype=torch.float32) for m in
             (ee, object_input, object_target, target_area)]
    if any(m.shape != CANVAS_HW or not torch.isfinite(m).all() or
           (m < 0).any() or (m > 1).any() for m in masks):
        raise ValueError('Expected finite [384,320] masks in [0,1]')
    return torch.stack(masks).amax(0)


def latent_weights(roi, pred, strength=3.0):
    """Area pooling, no temporal interpolation; per-frame mean weight is one.

    ROI[T,H,W] must follow exactly the packed vision item order. The pinned
    loader flattens each source/target pair into TWO separate T=1 vision items.
    Temporal-compressed video
requires a separate verified mapping and is deliberately not supported here.
"""
    # The network's unpatchify returns [1,C,T,H,W], while unit-level
    # callers may provide [C,T,H,W]. Spatial weights broadcast over both.
    if pred.ndim not in (4, 5) or (pred.ndim == 5 and pred.shape[0] != 1) or strength < 0:
        raise ValueError('Vision residual [C,T,H,W] or [1,C,T,H,W] required')
    roi = torch.as_tensor(roi, device=pred.device, dtype=torch.float32).detach()
    if roi.ndim != 3 or roi.shape[0] != pred.shape[-3]:
        raise ValueError('ROI frame order/length must match latent vision')
    if not torch.isfinite(roi).all() or (roi < 0).any() or (roi > 1).any():
        raise ValueError('ROI must be finite and in [0,1]')
    mask = F.interpolate(roi[:, None], size=pred.shape[-2:], mode='area')[:, 0]
    w = 1 + strength * mask
    return w / w.mean(dim=(-2, -1), keepdim=True)


def weighted_flow_loss(base_loss, *, roi_masks=None, strength=3.0, **kwargs):
    """Use the original loss implementation for all other behavior and modalities.

Squaring sqrt(w)*(pred-target) gives w*sqerr; the upstream function still applies
its original noisy mask, RF time weights, active normalization and reductions.
The baseline path passes through unchanged, bit-for-bit.
"""
    if roi_masks is None or strength == 0 or not kwargs['has_valid_tokens']:
        return base_loss(**kwargs)
    pred, target = kwargs['pred'], kwargs['target']
    if len(roi_masks) != len(pred) or len(pred) != len(target):
        raise ValueError('One aligned ROI per sample required')
    if kwargs.get('raw_action_dim') is not None:
        raise ValueError('ROI weighting is vision-only')
    weighted = [(p-t) * latent_weights(m, p, strength).sqrt().to(p.dtype)
                for p, t, m in zip(pred, target, roi_masks)]
    return base_loss(**dict(kwargs, pred=weighted, target=[torch.zeros_like(p) for p in weighted]))


def target_roi_for_item(item, sidecar, mode):
    """Bind loss metadata to the exact decoded input/GT, never replace either.

This adapter returns a side-channel; it does not add expert metadata to captions,
images or sequence plans. Collator/packing integration is an explicit next gate.
"""
    if mode not in ('ee_window', 'ee_object_target'):
        raise ValueError(mode)
    for key in ('__key__', '__url__', 'input_frame_index', 'target_frame_index'):
        if item[key] != sidecar[key]:
            raise ValueError(f'Sidecar does not match sample: {key}')
    images = item['images']
    if len(images) != 2 or any(tuple(x.shape) != (3, 384, 320) for x in images):
        raise ValueError('Only pinned two-image three-view contract is supported')
    for image, key in zip(images, ('source_chw_u8_sha256', 'gt_chw_u8_sha256')):
        if image.dtype != torch.uint8:
            raise ValueError('Hash binding must occur before image normalization')
        actual = hashlib.sha256(image.contiguous().cpu().numpy().tobytes()).hexdigest()
        if actual != sidecar[key]:
            raise ValueError(f'Replayed GT/source differs: {key}')
    ee = ee_canvas(sidecar['views'])
    target = ee if mode == 'ee_window' else union_roi(ee, sidecar['object_input_mask'],
                      sidecar['object_target_mask'], sidecar['target_area_mask'])
    if not (target > .01).any():
        raise ValueError('Empty supervision ROI: requires metadata repair, not GT exclusion')
    return torch.stack([torch.zeros_like(target), target])


def collate_roi_samples(base_collator, samples):
    """Loss-only transport, matching upstream sample-major images flattening.

Attach `_loss_roi_pair` after the dataset chooses the actual input/target. Strip
it before the normal collator sees samples; carry only the resulting masks to
the loss caller, which must pop `_loss_roi_items` BEFORE model preprocessing.
"""
    clean, roi_items = [], []
    for sample in samples:
        pair = sample['_loss_roi_pair']
        if pair.shape != (2, *CANVAS_HW) or len(sample['images']) != 2:
            raise ValueError('Expected fixed I2I pair, no video/mixed-modality packing')
        clean.append({k: v for k, v in sample.items() if k != '_loss_roi_pair'})
        roi_items.extend([pair[0:1].detach(), pair[1:2].detach()])
    batch = base_collator.collate(clean)
    batch['_loss_roi_items'] = roi_items
    return batch


class ROICollator:
    """Configuration target for the prepared private training release only."""
    def __init__(self):
        from cosmos_framework.data.vfm.dataflow import VFMListCollator
        self.base = VFMListCollator()

    def collate(self, samples):
        return collate_roi_samples(self.base, samples)
