"""Selected-weight loading must preserve native EMA casting and reject bad maps."""

import pytest
import torch
import torch.distributed.checkpoint as dcp

from recipes.simulation.robotwin.goalwam.checkpoint import load_serving_weights


def make_model():
    model = torch.nn.Module()
    model.net = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
    return model


def save_bundle(path, values):
    dcp.save(values, checkpoint_id=path / "model")
    (path / "complete.json").write_text("{}")


@pytest.mark.parametrize("weights", ["regular", "ema"])
def test_selected_tree_matches_native_copy(tmp_path, weights):
    model = make_model()
    regular = {name: torch.full_like(value, -8) for name, value in model.state_dict().items()}
    ema = {
        name.replace("net.", "net_ema.", 1): torch.linspace(0.01, 1.13, value.numel()).reshape(value.shape)
        for name, value in regular.items()
    }
    save_bundle(tmp_path, regular | ema)
    loading = load_serving_weights(model, tmp_path, weights)
    for name, value in model.state_dict().items():
        source = ema[name.replace("net.", "net_ema.", 1)] if weights == "ema" else regular[name]
        expected = torch.empty_like(value)
        expected.copy_(source)
        assert torch.equal(value, expected)
    assert loading["source_prefix"] == ("net_ema." if weights == "ema" else "net.")
    assert not hasattr(model, "net_ema")


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype"])
def test_invalid_selected_tree_rejected_before_loading(tmp_path, corruption):
    model = make_model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    values = {name.replace("net.", "net_ema.", 1): value.float() for name, value in before.items()}
    if corruption == "missing":
        del values["net_ema.bias"]
    elif corruption == "extra":
        values["net_ema.extra"] = torch.ones(1)
    elif corruption == "shape":
        values["net_ema.weight"] = torch.ones(1)
    else:
        values["net_ema.weight"] = values["net_ema.weight"].to(torch.int64)
    save_bundle(tmp_path, values)
    with pytest.raises(ValueError, match="tensor"):
        load_serving_weights(model, tmp_path, "ema")
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
