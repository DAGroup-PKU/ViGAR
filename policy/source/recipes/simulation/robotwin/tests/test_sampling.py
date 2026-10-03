"""Inference tuning preserves checkpoint defaults and rejects invalid requests."""

import pytest

from recipes.simulation.robotwin.goalwam.sampling import resolve_sampling


def test_sampling_overrides_preserve_unspecified_checkpoint_settings():
    train = dict(gen_num_steps=10, gen_guidance=3.0, gen_shift=5.0)
    assert resolve_sampling(None, train) == dict(num_steps=10, guidance=3.0, shift=5.0)
    assert resolve_sampling(dict(guidance=1.0), train) == dict(num_steps=10, guidance=1.0, shift=5.0)
    assert train == dict(gen_num_steps=10, gen_guidance=3.0, gen_shift=5.0)


def test_integer_yaml_shift_is_converted_for_native_unipc():
    sampling = resolve_sampling(dict(guidance=3, shift=1))
    assert type(sampling["shift"]) is float and sampling["shift"] == 1.0
    assert type(sampling["guidance"]) is float and sampling["guidance"] == 3.0


@pytest.mark.parametrize(
    "value",
    [
        dict(num_steps=0),
        dict(num_steps=True),
        dict(guidance=-1),
        dict(shift=0),
        dict(shift=float("nan")),
        dict(guidance=float("inf")),
        dict(unknown=2),
        [],
    ],
)
def test_invalid_sampling_rejected(value):
    with pytest.raises(ValueError):
        resolve_sampling(value)
