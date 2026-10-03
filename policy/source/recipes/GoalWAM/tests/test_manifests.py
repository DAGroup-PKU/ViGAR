"""Named manifest populations, explicit legacy holdout and per-entry weights."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from recipes.GoalWAM.data.dataset import LeRobotPolicyDataset, load_manifests, manifest_paths
from recipes.GoalWAM.tests.test_checkpoint_bundle import make_bundle
from recipes.GoalWAM.trainer.arguments import VeOmniGoalWAMArguments
from veomni.arguments.parser import _add_arguments_recursive, _instantiate_recursive


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_mapping_arguments_and_explicit_evaluation(tmp_path):
    values = yaml.safe_load((CONFIGS / "robotwin.yaml").read_text())
    values["model"]["model_path"] = str(make_bundle(tmp_path / "bundle"))
    parser = argparse.ArgumentParser()
    _add_arguments_recursive(parser, VeOmniGoalWAMArguments)
    options = vars(parser.parse_args(["--data.eval_path", '{"held_out": "test.yaml"}']))
    values["data"]["eval_path"] = options["data.eval_path"]
    args = _instantiate_recursive(VeOmniGoalWAMArguments, values)
    assert args.data.eval_path == {"held_out": "test.yaml"}
    assert args.data.split == args.data.eval_split == "all"
    values["data"].pop("eval_path")
    with pytest.raises(ValueError, match="Set data.eval_path explicitly"):
        _instantiate_recursive(VeOmniGoalWAMArguments, values)
    values["train"]["eval_loss_per_rank"] = values["train"]["generation_per_rank"] = 0
    assert _instantiate_recursive(VeOmniGoalWAMArguments, values).data.eval_path is None
    for config in ("parity.yaml", "eval.yaml"):
        legacy = _instantiate_recursive(
            VeOmniGoalWAMArguments, yaml.safe_load((CONFIGS / "migration" / config).read_text())
        )
        assert legacy.data.split == "train" and legacy.data.eval_split == "eval"
        assert legacy.data.train_path == legacy.data.eval_path


@pytest.mark.parametrize("value", [{}, [], {"": "a.yaml"}, {"robot": None}, {"robot": " "}])
def test_invalid_manifest_mapping_rejected(value):
    with pytest.raises(ValueError, match="[Mm]anifest"):
        manifest_paths(value)


def test_namespaced_entries_and_manifest_provenance(tmp_path):
    first, second = tmp_path / "a.yaml", tmp_path / "b.yaml"
    first.write_text(yaml.safe_dump({"shared_task": {"robot_type": "robot_a", "data_path": "/a"}}))
    second.write_text(yaml.safe_dump({"shared_task": {"robot_type": "robot_b", "data_path": "/b"}}))
    sources = {"a": str(first), "b": str(second)}
    paths, hashes, entries = load_manifests(sources)
    assert paths == sources and set(hashes) == {"a", "b"} and hashes["a"] != hashes["b"]
    assert list(entries) == ["a/shared_task", "b/shared_task"]
    assert entries["a/shared_task"]["robot_type"] == "robot_a"
    assert entries["b/shared_task"]["robot_type"] == "robot_b"
    # A single path preserves the original dataset names and digest schema.
    assert load_manifests(first)[1] == hashes["a"]
    assert list(load_manifests(first)[2]) == ["shared_task"]


def write_population(tmp_path):
    root = tmp_path / "data"
    metadata = root / "meta"
    episodes = metadata / "episodes/chunk-000"
    episodes.mkdir(parents=True)
    features = {
        "action": {"shape": [49]},
        "observation.state": {"shape": [49]},
        **{
            f"observation.images.{camera}": {"shape": [240, 320, 3]}
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        },
    }
    (metadata / "info.json").write_text(
        json.dumps(
            dict(
                codebase_version="v3.0",
                fps=30,
                eef_gripper_frame="astribot_s1",
                action_dim_mask=[True] * 49,
                features=features,
            )
        )
    )
    annotation = json.dumps(
        dict(task={"command": {"en": "Move the block"}}, fps=30, duration=64 / 30, resolution=[240, 320])
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(episode_index=index, length=64, annotation=annotation, tasks=["Move the block"])
                for index in range(8)
            ]
        ),
        episodes / "file-000.parquet",
    )
    pq.write_table(pa.Table.from_pylist([dict(task="Move the block")]), metadata / "tasks.parquet")
    stats = {key: dict(mean=[0] * 49, std=[1] * 49) for key in ("observation.state", "action")}
    norm = tmp_path / "norm.json"
    norm.write_text(json.dumps(dict(norm_stats=stats, metadata={"state_arm_eef_coordinate": "head_camera"})))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "task": dict(
                    data_path=str(root),
                    robot_type="robotwin_aloha_agilex",
                    dataset_type="lerobot",
                    ignore_episodes=[7],
                    sampling_rate=2,
                )
            }
        )
    )
    return manifest, {"robotwin_aloha_agilex": str(norm)}


def test_all_episodes_by_default_and_weights_independent_of_holdout(tmp_path):
    manifest, norms = write_population(tmp_path)
    manifests = {"robot": str(manifest)}
    training = LeRobotPolicyDataset(manifests, norms, training=True)
    evaluation = LeRobotPolicyDataset(manifests, norms, training=False)
    assert len(training.episodes) == len(evaluation.episodes) == 7
    assert {ep["metadata"]["episode_index"] for ep in training.episodes} == set(range(7))
    assert len(training) == 7 * 16 * 2 and len(evaluation) == 7 * 16
    assert all(ep["name"] == "robot/task" for ep in training.episodes)
    # Only explicit legacy holdout excludes five episodes, reproducibly.
    legacy_train = LeRobotPolicyDataset(manifest, norms, split="train")
    legacy_eval = LeRobotPolicyDataset(manifest, norms, split="eval")
    train_ids = {ep["metadata"]["episode_index"] for ep in legacy_train.episodes}
    eval_ids = {ep["metadata"]["episode_index"] for ep in legacy_eval.episodes}
    assert len(train_ids) == 2 and len(eval_ids) == 5 and not train_ids & eval_ids
    assert train_ids | eval_ids == set(range(7))
    assert len(legacy_train) == 2 * 16 * 2 and len(legacy_eval) == 5 * 16
