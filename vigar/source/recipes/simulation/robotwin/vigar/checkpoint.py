"""Load one native DCP weight tree into the inference network."""

from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict


def load_serving_weights(model, checkpoint, weights):
    """Match native EMA-to-network casting without constructing an EMA replica."""
    if weights not in {"regular", "ema"}:
        raise ValueError("Serving weights must be regular or ema")
    checkpoint = Path(checkpoint)
    if not (checkpoint / "complete.json").is_file():
        raise ValueError(f"Incomplete checkpoint: {checkpoint}")
    state = get_model_state_dict(model)
    if not state or any(not name.startswith("net.") for name in state):
        raise ValueError("Serving requires only the native net weight tree")
    prefix = "net_ema." if weights == "ema" else "net."
    selected = {prefix + name.removeprefix("net."): tensor for name, tensor in state.items()}
    metadata = dcp.FileSystemReader(checkpoint / "model").read_metadata().state_dict_metadata
    source_names = {name for name in metadata if name.startswith(prefix)}
    if source_names != set(selected):
        raise ValueError("Selected checkpoint tensor names do not exactly match the inference network")
    for name, tensor in selected.items():
        stored = metadata[name]
        # Native EMA is FP32; copy_to converts it to the regular network dtype.
        ema_cast = (
            weights == "ema"
            and stored.properties.dtype == torch.float32
            and tensor.dtype in {torch.float16, torch.bfloat16}
        )
        if tensor.shape != stored.size or (tensor.dtype != stored.properties.dtype and not ema_cast):
            raise ValueError(f"Checkpoint tensor shape/dtype mismatch: {name}")
    dcp.load(selected, checkpoint_id=checkpoint / "model")
    incompatible = set_model_state_dict(model, state, options=StateDictOptions(strict=False))
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Checkpoint model mapping is incomplete: {incompatible}")
    return dict(weights=weights, source_prefix=prefix, tensors=len(selected), ema_replica=False)
