"""Serving preserves the multi-view training goal layout and tolerates unavailable wrist RGB."""

import json

import numpy as np
import pytest
import torch
import yaml

from recipes.GoalWAM.data.images import camera_layout
from recipes.GoalWAM.tests.test_goal_sampling import dataset
from recipes.simulation.robotwin.common.geometry import ROBOT, encode_observation
from recipes.simulation.robotwin.goalwam import policy
from recipes.simulation.robotwin.tests.test_bridge import observation as observation


@pytest.mark.parametrize("missing", ["absent", "none", "empty", "wrong_dtype"])
def test_serving_goal_composition(tmp_path, observation, missing):
    raw = dataset(tmp_path)
    data = dict(
        img_size=[32, 32],
        enable_cameras=["head", "left", "right"],
        state_arm_eef_coordinate="head_camera",
        goal_image_composition="multi_view",
    )
    processor = policy.ObservationProcessor(data, raw.normalizers[ROBOT])
    request = dict(encode_observation(observation), instruction="Move the object.", robot_type=ROBOT)
    request["goal_images"] = dict(request["images"])
    complete, _, _ = processor.raw_sample(request)
    for name, (y, x, h, w) in camera_layout((32, 32), data["enable_cameras"])[1].items():
        assert complete["video"][:, :1, y : y + h, x : x + w].any()
    for name in ("left", "right"):
        if missing == "absent":
            request["goal_images"].pop(name)
        else:
            request["goal_images"][name] = dict(
                none=None, empty=np.empty((0, 0, 3), dtype=np.uint8), wrong_dtype=np.ones((24, 32, 3))
            )[missing]
    sample, _, _ = processor.raw_sample(request)
    assert torch.equal(sample["video"], complete["video"])
    assert sample["goal_frame"].shape == (3, 1, 64, 32)
    for name, (y, x, h, w) in camera_layout((32, 32), data["enable_cameras"])[1].items():
        if name != "head":
            assert not sample["goal_frame"][..., y : y + h, x : x + w].any()
    request["goal_images"].pop("head")
    with pytest.raises(ValueError, match="RGB"):
        processor.raw_sample(request)
    raw.close()


@pytest.mark.parametrize("saved_mode", [None, "multi_view"])
def test_saved_composition_defaults_and_override_contract(tmp_path, monkeypatch, saved_mode):
    monkeypatch.setattr(policy, "read_bundle", lambda path: {})
    monkeypatch.setattr(policy, "ContractNormalizer", lambda *args: object())
    data = dict(img_size=[240, 320], norm_stat_files={ROBOT: "unused.json"})
    if saved_mode:
        data["goal_image_composition"] = saved_mode
    (tmp_path / "recipe_config.json").write_text(json.dumps(dict(data=data)))
    settings, _, _ = policy.load_settings(tmp_path)
    assert settings["data"]["goal_image_composition"] == (saved_mode or "multi_view")
    data["goal_image_composition"] = "unsupported"
    supplied = tmp_path / "override.yaml"
    supplied.write_text(yaml.safe_dump(dict(data=data)))
    with pytest.raises(ValueError, match="data.goal_image_composition"):
        policy.load_settings(tmp_path, supplied)


@pytest.mark.parametrize(
    "mode,cameras",
    [
        ("multi_view", ["head", "left", "right"]),
        ("multi_view", ["head"]),
    ],
)
def test_bucketed_serving_matches_training(tmp_path, observation, mode, cameras):
    from recipes.GoalWAM.data.images import CAMERA_KEYS
    from recipes.GoalWAM.tests.test_resolution_buckets import BUCKETS, variable_dataset

    raw = variable_dataset(tmp_path, mode, cameras)
    processor = policy.ObservationProcessor(
        dict(
            img_size=[224, 288],
            img_size_buckets=BUCKETS,
            enable_cameras=cameras,
            state_arm_eef_coordinate="head_camera",
            goal_image_composition=mode,
        ),
        raw.normalizers[ROBOT],
    )
    for i, size in enumerate(BUCKETS):
        request = dict(encode_observation(observation), instruction="Move the object.", robot_type=ROBOT)
        request["source_hw"] = size
        request["images"] = {name: raw._video(raw.episodes[i], CAMERA_KEYS[name], [0])[0] for name in cameras}
        request["goal_images"] = dict(request["images"])
        sample, _, _ = processor.raw_sample(request)
        training = raw[i * 64]
        assert sample["image_bucket_id"] == i
        assert torch.equal(sample["target_hw"], training["target_hw"])
        assert torch.equal(sample["video"][:, :1], training["video"][:, :1])
        assert torch.equal(sample["goal_frame"], training["goal_frame"])
        prepared, _, _ = processor.prepare(request)
        assert prepared["image_size"].tolist() == [*training["video"].shape[-2:]] * 2
    request.pop("source_hw")
    # Raw head aspect (24x32) chooses the landscape 224x288 bucket.
    assert processor.raw_sample(request)[0]["image_bucket_id"] == 1


def test_serving_rejects_bucket_override(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "read_bundle", lambda path: {})
    data = dict(img_size=[224, 288], img_size_buckets=[[224, 288]], norm_stat_files={ROBOT: "unused.json"})
    (tmp_path / "recipe_config.json").write_text(json.dumps(dict(data=data, image_preprocessing_version=2)))
    data["img_size_buckets"].append([256, 256])
    supplied = tmp_path / "override.yaml"
    supplied.write_text(yaml.safe_dump(dict(data=data)))
    with pytest.raises(ValueError, match="img_size_buckets"):
        policy.load_settings(tmp_path, supplied)
