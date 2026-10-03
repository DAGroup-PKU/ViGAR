"""A checkpoint owns its configuration and dependencies, even after relocation."""

import json
from pathlib import Path

import pytest

from recipes.GoalWAM.trainer.arguments import GoalWAMModelArguments
from recipes.GoalWAM.trainer.checkpoint_bundle import (
    BUNDLE_ASSETS,
    finish_bundle,
    read_bundle,
    save_portable_config,
)
from veomni.models.transformers.goalwam.configuration_goalwam import GoalWAMConfig


PRESET = Path(__file__).resolve().parents[1] / "configs/migration/initialization_config.json"


def make_bundle(path):
    path.mkdir()
    config = GoalWAMConfig.from_json_file(PRESET)
    save_portable_config(config, path)
    for relative in (
        "model/.metadata",
        BUNDLE_ASSETS["vae"],
        BUNDLE_ASSETS["backbone_config"],
        BUNDLE_ASSETS["tokenizer"] + "/tokenizer.json",
    ):
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture asset")
    finish_bundle(path, kind="initial", iteration=0, world_size=4)
    return path


def test_bundle_relocation_and_authoritative_configuration(tmp_path):
    old = make_bundle(tmp_path / "original")
    relocated = tmp_path / "relocated"
    old.rename(relocated)
    args = GoalWAMModelArguments(model_path=str(relocated))
    assert args.config_path == args.native_checkpoint == str(relocated)
    assert args.vae_path == str(relocated / BUNDLE_ASSETS["vae"])
    assert args.tokenizer_path == str(relocated / BUNDLE_ASSETS["tokenizer"])
    config = GoalWAMConfig.from_json_file(relocated / "config.json")
    runtime = config.runtime_config(
        base_checkpoint=relocated, vae_path=None, tokenizer_path=None, shard_degree=4, checkpoint_dir=relocated
    )
    assert runtime.model.config.tokenizer.vae_path == args.vae_path
    assert runtime.model.config.vlm_config.tokenizer.pretrained_model_name == args.tokenizer_path
    assert runtime.model.config.vlm_config.model_instance.config.base_config.json_file == str(
        relocated / BUNDLE_ASSETS["backbone_config"]
    )
    assert config.cosmos["checkpoint"]["load_path"] == "."


@pytest.mark.parametrize(
    "field", ["config_path", "native_checkpoint", "vae_path", "tokenizer_path", "base_checkpoint"]
)
def test_conflicting_assets_rejected(tmp_path, field):
    bundle = make_bundle(tmp_path / "bundle")
    with pytest.raises(ValueError, match="conflicting|selected checkpoint"):
        GoalWAMModelArguments(model_path=str(bundle), **{field: str(tmp_path / "other")})


def test_missing_and_corrupted_bundle_rejected(tmp_path):
    bundle = make_bundle(tmp_path / "bundle")
    config = bundle / "config.json"
    content = config.read_bytes()
    config.write_bytes(content.replace(b'"action_horizon": 48', b'"action_horizon": 47'))
    with pytest.raises(ValueError, match="checksum"):
        read_bundle(bundle)
    config.write_bytes(content)
    (bundle / BUNDLE_ASSETS["vae"]).unlink()
    with pytest.raises(ValueError, match="Missing or truncated"):
        read_bundle(bundle)
    (bundle / "complete.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        read_bundle(bundle)


def test_completion_requires_all_assets_and_rejects_external_symlinks(tmp_path):
    bundle = make_bundle(tmp_path / "bundle")
    (bundle / "complete.json").unlink()
    vae = bundle / BUNDLE_ASSETS["vae"]
    vae.unlink()
    with pytest.raises(ValueError, match="missing asset"):
        finish_bundle(bundle, kind="initial", iteration=0, world_size=4)
    assert not (bundle / "complete.json").exists()
    external = tmp_path / "external.pth"
    external.write_bytes(b"fixture asset")
    vae.symlink_to(external)
    with pytest.raises(ValueError, match="inside"):
        finish_bundle(bundle, kind="initial", iteration=0, world_size=4)


def test_manifest_completion_binding(tmp_path):
    bundle = make_bundle(tmp_path / "bundle")
    manifest_path = bundle / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["kind"] = "training"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completion marker"):
        read_bundle(bundle)
