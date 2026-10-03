"""Multi-view goal layouts, unavailable wrists and native conditioning."""

import json

import pytest
import torch

from recipes.GoalWAM.data.dataset import LeRobotPolicyDataset
from recipes.GoalWAM.data.images import CAMERA_KEYS, camera_layout, compose_goal_image
from recipes.GoalWAM.tests.test_goal_sampling import dataset
from recipes.GoalWAM.trainer.arguments import GoalWAMDataArguments
from recipes.GoalWAM.trainer.metrics import image_metrics


def views():
    return {
        name: torch.full((1, 3, 24, 32), color, dtype=torch.uint8)
        for name, color in (("head", 50), ("left", 100), ("right", 200))
    }


@pytest.mark.parametrize("missing", [("left",), ("right",), ("left", "right")])
@pytest.mark.parametrize("invalid", ["absent", "none", "empty", "nan", "bad_shape"])
def test_missing_or_invalid_wrists_leave_black_reserved_cells(missing, invalid):
    images = views()
    for name in missing:
        if invalid == "absent":
            images.pop(name)
        else:
            images[name] = {
                "none": None,
                "empty": torch.empty(1, 3, 0, 32),
                "nan": torch.full((1, 3, 24, 32), float("nan")),
                "bad_shape": torch.zeros(1, 1, 24, 32),
            }[invalid]
    canvas, mask, boxes = compose_goal_image(images, (32, 32), ["head", "left", "right"])
    assert canvas.shape == (1, 3, 64, 32)
    assert boxes == camera_layout((32, 32), ["head", "left", "right"])[1]
    for name, (y, x, h, w) in boxes.items():
        cell = canvas[:, :, y : y + h, x : x + w]
        pixels = mask[y : y + h, x : x + w]
        if name in missing:
            assert not cell.any() and not pixels.any()
        else:
            assert pixels.any() and cell[..., pixels].eq({"head": 50, "left": 100, "right": 200}[name]).all()


def test_black_rgb_is_valid_and_required_head_is_not_fabricated():
    images = views()
    images["left"].zero_()
    _, mask, boxes = compose_goal_image(images, (32, 32), list(images))
    y, x, h, w = boxes["left"]
    assert mask[y : y + h, x : x + w].any()
    images["head"] = None
    with pytest.raises(ValueError, match="head"):
        compose_goal_image(images, (32, 32), list(images))


def test_legacy_resize_profile_composes_multi_view_goal():
    images = views()
    images.pop("left")
    goal, mask, boxes = compose_goal_image(images, None, ["head", "left", "right"], resolution="384x320")
    assert goal.shape == (1, 3, 384, 320)
    assert not goal[..., ~mask].any()
    assert not goal[:, :, 256:, :160].any()
    assert goal[:, :, :256].eq(50).all()
    assert goal[:, :, 256:, 160:].eq(200).all()


def rebuild(raw, tmp_path, mode="multi_view"):
    result = LeRobotPolicyDataset(
        raw.manifest,
        {"robotwin_aloha_agilex": str(tmp_path / "norm.json")},
        training=True,
        include_tail_windows=raw.include_tail_windows,
        img_size=[32, 32],
        goal_image_composition=mode,
    )
    result._episode_rows, result._video = raw._episode_rows, raw._video
    return result


@pytest.mark.parametrize("missing", [("left",), ("left", "right")])
def test_dataset_without_wrist_features_loads_black_cells(tmp_path, missing):
    raw = dataset(tmp_path)
    info_path = tmp_path / "data/meta/info.json"
    info = json.loads(info_path.read_text())
    for name in missing:
        info["features"].pop(CAMERA_KEYS[name])
    info_path.write_text(json.dumps(info))
    raw = rebuild(raw, tmp_path)
    original = raw._video

    def read(ep, camera, indices):
        assert camera not in [CAMERA_KEYS[name] for name in missing]
        return original(ep, camera, indices)

    raw._video = read
    sample = raw[0]
    for name in missing:
        for key, mask_key, boxes_key in (
            ("goal_frame", "goal_pixel_mask", "goal_camera_boxes"),
            ("video", "video_pixel_mask", "camera_boxes"),
        ):
            y, x, h, w = sample[boxes_key][name]
            assert not sample[key][..., y : y + h, x : x + w].any()
            assert not sample[mask_key][y : y + h, x : x + w].any()
    metrics = image_metrics(sample["video"], sample["video"], sample["video_pixel_mask"], sample["camera_boxes"])
    assert all(f"video_{name}_mae" not in metrics for name in missing)
    assert "image_composition" in raw.selection_record()


@pytest.mark.parametrize("error_type", [ValueError, KeyError, FileNotFoundError, TypeError])
def test_unavailable_goal_wrist_retains_valid_rollout_and_head_errors_fail(tmp_path, error_type):
    raw = dataset(tmp_path)
    reference = raw[0]
    original = raw._video

    def read(ep, camera, indices):
        if camera == CAMERA_KEYS["left"] and ep["end"] - 1 in indices:
            raise error_type("Unavailable wrist goal frames or metadata")
        return original(ep, camera, indices)

    raw._video = read
    with pytest.warns(UserWarning, match="unavailable wrist"):
        sample = raw[0]
    torch.testing.assert_close(sample["video"], reference["video"], rtol=0, atol=0)
    y, x, h, w = sample["goal_camera_boxes"]["left"]
    assert not sample["goal_frame"][..., y : y + h, x : x + w].any()
    assert not sample["goal_pixel_mask"][y : y + h, x : x + w].any()

    def fail_head(ep, camera, indices):
        raise ValueError("Head decoder failed")

    raw._video = fail_head
    with pytest.raises(ValueError, match="Head decoder failed"):
        raw[0]


def test_composition_config_validation():
    assert GoalWAMDataArguments(train_path="manifest.yaml").goal_image_composition == "multi_view"
    with pytest.raises(ValueError, match="goal_image_composition"):
        GoalWAMDataArguments(train_path="manifest.yaml", goal_image_composition="unknown")
