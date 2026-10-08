import torch
import torch.nn.functional as F

CANVAS_HW = (384, 320)
VIEWS = {
    "head": (0, 0, 256, 320),
    "left_wrist": (256, 0, 128, 160),
    "right_wrist": (256, 160, 128, 160),
}


def latent_weights(roi, pred, strength=3.0):
    if (
        pred.ndim not in (4, 5)
        or (pred.ndim == 5 and pred.shape[0] != 1)
        or strength < 0
    ):
        raise ValueError("Vision residual [C,T,H,W] or [1,C,T,H,W] required")
    roi = torch.as_tensor(roi, device=pred.device, dtype=torch.float32).detach()
    if roi.ndim != 3 or roi.shape[0] != pred.shape[-3]:
        raise ValueError("ROI frame order/length must match latent vision")
    if not torch.isfinite(roi).all() or (roi < 0).any() or (roi > 1).any():
        raise ValueError("ROI must be finite and in [0,1]")
    mask = F.interpolate(roi[:, None], size=pred.shape[-2:], mode="area")[:, 0]
    w = 1 + strength * mask
    return w / w.mean(dim=(-2, -1), keepdim=True)


def weighted_flow_loss(base_loss, *, roi_masks=None, strength=3.0, **kwargs):
    if roi_masks is None or strength == 0 or (not kwargs["has_valid_tokens"]):
        return base_loss(**kwargs)
    (pred, target) = (kwargs["pred"], kwargs["target"])
    if len(roi_masks) != len(pred) or len(pred) != len(target):
        raise ValueError("One aligned ROI per sample required")
    if kwargs.get("raw_action_dim") is not None:
        raise ValueError("ROI weighting is vision-only")
    weighted = [
        (p - t) * latent_weights(m, p, strength).sqrt().to(p.dtype)
        for (p, t, m) in zip(pred, target, roi_masks)
    ]
    return base_loss(
        **dict(kwargs, pred=weighted, target=[torch.zeros_like(p) for p in weighted])
    )


def collate_roi_samples(base_collator, samples):
    (clean, roi_items) = ([], [])
    for sample in samples:
        pair = sample["_loss_roi_pair"]
        if pair.shape != (2, *CANVAS_HW) or len(sample["images"]) != 2:
            raise ValueError("Expected fixed I2I pair, no video/mixed-modality packing")
        clean.append({k: v for (k, v) in sample.items() if k != "_loss_roi_pair"})
        roi_items.extend([pair[0:1].detach(), pair[1:2].detach()])
    batch = base_collator.collate(clean)
    batch["_loss_roi_items"] = roi_items
    return batch


class ROICollator:
    def __init__(self):
        from cosmos_framework.data.vfm.dataflow import VFMListCollator

        self.base = VFMListCollator()

    def collate(self, samples):
        return collate_roi_samples(self.base, samples)
