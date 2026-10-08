"""Causal planner interface, shared by offline training preparation and serving.

Call ``generate_goal(model, current_rgb, instruction, seed=...)`` whenever the
caller requests a new fixed goal. No annotation, phase ID, future image, action,
or simulator state is accepted by this interface.
"""

import numpy as np
import torch

from cosmos_framework.data.vfm.dataflow import VFMListCollator
from cosmos_framework.data.vfm.sequence_packing import SequencePlan
from cosmos_framework.utils import misc


def conditioning_batch(current_rgb, instruction):
    rgb = np.asarray(current_rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (384, 320, 3):
        raise ValueError("Planner needs an RGB uint8 three-camera canvas [384,320,3]")
    source = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
    size = torch.tensor([384, 320, 384, 320], dtype=torch.float32)
    # The second image specifies output shape only. It is deliberately blank:
    # the sampler replaces all its tokens with noise, so no teacher target can leak.
    return VFMListCollator().collate([{
        "images": [source, torch.zeros_like(source)],
        "image_size": [size, size.clone()], "ai_caption": str(instruction),
        "selected_caption_type": "editing_instruction", "num_frames": 2,
        "fps": 30.0, "conditioning_fps": 30.0,
        "sequence_plan": SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[]),
        "height": 384, "width": 320, "original_height": 384, "original_width": 320,
        "aspect_ratio": "3,4", "resize_resolution": "256",
    }])


@torch.no_grad()
def generate_goal(model, current_rgb, instruction, *, seed, num_steps=35, guidance=2.5):
    batch = misc.to(conditioning_batch(current_rgb, instruction), device="cuda")
    result = model.generate_samples_from_batch(
        batch, n_sample=1, seed=[int(seed)], num_steps=num_steps,
        guidance=guidance, shift=5.0, sigma_max=80.0,
    )
    image = model.decode(result["vision"][0]).detach().float().cpu()
    if image.ndim == 5:
        image = image[0, :, -1]
    elif image.ndim == 4:
        image = image[0] if image.shape[0] == 1 else image[:, -1]
    if tuple(image.shape) != (3, 384, 320):
        raise RuntimeError(f"Unexpected planner output shape: {tuple(image.shape)}")
    return ((image + 1) * 127.5).clamp(0, 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
