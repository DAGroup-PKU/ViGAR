"""Registry, serialized contract, asset relocation and lifecycle boundaries."""

from pathlib import Path

import pytest
import torch

from veomni.models.loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY, get_model_class, get_model_config
from veomni.models.transformers.goalwam.configuration_goalwam import GoalWAMConfig


def test_registered_config_round_trip_and_relocated_assets(tmp_path):
    assert "goalwam" in MODEL_CONFIG_REGISTRY and "goalwam" in MODELING_REGISTRY
    path = Path(__file__).resolve().parents[1] / "configs/migration/initialization_config.json"
    config = get_model_config(str(path))
    assert get_model_class(config).__name__ == "GoalWAMPolicy"
    config.save_pretrained(tmp_path)
    restored = get_model_config(str(tmp_path))
    assert restored.cosmos == config.cosmos
    native = restored.runtime_config(
        base_checkpoint=tmp_path / "base",
        vae_path=tmp_path / "vae.pth",
        tokenizer_path=tmp_path / "tokens",
        shard_degree=4,
    )
    assert native.model.config.tokenizer.vae_path == str(tmp_path / "vae.pth")
    assert native.model.config.vlm_config.tokenizer.pretrained_model_name == str(tmp_path / "tokens")
    architecture = Path(native.model.config.vlm_config.model_instance.config.base_config.json_file)
    runtime_root = Path(__file__).resolve().parents[3] / "third_party/goalwam"
    assert architecture.is_relative_to(runtime_root) and architecture.is_file()
    assert native.model.config.max_action_dim == 64
    assert native.model.config.rectified_flow_training_config.action_loss_weight == 10


def test_contract_rejects_accidental_architecture_extension():
    with pytest.raises(ValueError, match="frozen"):
        GoalWAMConfig(observation_history=2)


def test_wrapper_defers_native_sharding_and_does_not_reinitialize(monkeypatch):
    from cosmos_framework.model.vfm import omni_mot_model

    from veomni.models.transformers.goalwam.modeling_goalwam import GoalWAMPolicy

    class Core(torch.nn.Module):
        def __init__(self, config, *, defer_network_init):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((2,), 7.0))
            self.deferred = defer_network_init
            self.initialized = False

        def initialize_network(self, parallel_dims):
            if self.initialized:
                raise RuntimeError("already initialized")
            self.initialized = True

        def training_step(self, batch, iteration):
            return {"iteration": iteration}, self.weight.sum() * batch

    monkeypatch.setattr(omni_mot_model, "OmniMoTModel", Core)
    model = GoalWAMPolicy(GoalWAMConfig(cosmos={"model": {"config": {}}}), defer_network_init=True)
    assert model.core.deferred and not model.core.initialized
    assert model.core.weight.tolist() == [7.0, 7.0]
    model.initialize_network()
    with pytest.raises(RuntimeError, match="already"):
        model.initialize_network()
    result = model(torch.tensor(2.0), iteration=3)
    assert result.native_outputs == {"iteration": 3}
    assert result.loss.item() == 28.0
