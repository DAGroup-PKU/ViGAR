"""Validated inference overrides shared by the simulator and model server."""

import math


def resolve_sampling(overrides=None, train=None):
    train = train or {}
    values = dict(
        num_steps=train.get("gen_num_steps", 5),
        guidance=train.get("gen_guidance", 3.0),
        shift=train.get("gen_shift", 5.0),
    )
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or set(overrides) - values.keys():
        raise ValueError("sampling accepts num_steps, guidance and shift only")
    values.update(overrides)
    if type(values["num_steps"]) is not int or values["num_steps"] < 1:
        raise ValueError("sampling.num_steps must be a positive integer")
    for key in ("guidance", "shift"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"sampling.{key} must be finite")
        if value < 0 or (key == "shift" and value == 0):
            raise ValueError("sampling.guidance must be nonnegative and sampling.shift positive")
        # Native UniPC requires float shift even when YAML spells it as `1`.
        values[key] = float(value)
    return values
