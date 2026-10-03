"""Current-segment instructions, unchanged targets and resumable text contracts."""

import argparse
import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from recipes.GoalWAM.data.data_loader import StatefulWindowLoader
from recipes.GoalWAM.data.dataset import LeRobotPolicyDataset, LeRobotPolicySFTDataset
from recipes.GoalWAM.data.text_conditioning import segment_text_intervals, select_text
from recipes.GoalWAM.tests.test_goal_sampling import only, segment
from recipes.GoalWAM.tests.test_tail_windows import population
from recipes.GoalWAM.trainer.arguments import GoalWAMDataArguments, VeOmniGoalWAMArguments
from veomni.arguments.parser import _add_arguments_recursive, _instantiate_recursive


def annotated_population(tmp_path, *, rate=1):
    base, _ = population(tmp_path, rate=rate)
    path = tmp_path / "data/meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(path).to_pylist()
    for ep in episodes:
        annotation = json.loads(ep["annotation"])
        first, second = segment(0, 8 / 30), segment(10 / 30, 60 / 30)
        first["caption"]["overall"]["en"] = "Pick up the block"
        second["caption"]["overall"]["en"] = "Place the block"
        annotation["segments"] = [first, second]
        ep["annotation"] = json.dumps(annotation)
    pq.write_table(pa.Table.from_pylist(episodes), path)
    return base


def configured(base, tmp_path, mode="episode", *, training=True, tails=True):
    raw = LeRobotPolicyDataset(
        base.manifest,
        {"robotwin_aloha_agilex": str(tmp_path / "norm.json")},
        training=training,
        include_tail_windows=tails,
        img_size=[32, 32],
        goal_sampling=only("segment"),
        text_conditioning=mode,
    )
    raw._episode_rows, raw._video = base._episode_rows, base._video
    return raw


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("mode", ["episode", "segment", "episode_segment"])
def test_current_segment_text_and_unchanged_goals_targets(tmp_path, training, mode):
    base = annotated_population(tmp_path)
    raw = configured(base, tmp_path, mode, training=training, tails=training)
    legacy = configured(base, tmp_path, training=training, tails=training)
    for index, subtask in [(0, "Pick up the block"), (7, "Pick up the block"), (8, None), (10, "Place the block")]:
        sample, reference = raw[(index, 123)], legacy[(index, 123)]
        expected = reference["ai_caption"]
        if subtask and mode != "episode":
            expected = subtask if mode == "segment" else f"{expected}\nCurrent subtask: {subtask}"
        assert sample["ai_caption"] == expected
        for key in ("action", "action_valid_mask", "video", "goal_frame"):
            torch.testing.assert_close(sample[key], reference[key], rtol=0, atol=0)
        assert sample["goal_frame_index"] == reference["goal_frame_index"]
        if not training:
            assert sample["goal_frame_index"] == 63
    # At frame zero, the visual goal is in a later segment, but the caption is current.
    if training:
        assert raw[0]["goal_segment_index"] == 1
        assert raw[63]["ai_caption"] == "Move the block"
    raw.include_robot_type_text_context = True
    assert raw[0]["ai_caption"].startswith("<robot_type>robotwin_aloha_agilex</robot_type> ")


def test_acceleration_does_not_rescale_annotation_time(tmp_path):
    raw = configured(annotated_population(tmp_path, rate=3), tmp_path, "segment")
    assert raw[7]["ai_caption"] == "Pick up the block"
    assert raw[8]["ai_caption"] == "Move the block"
    assert raw[10]["ai_caption"] == "Place the block"


def test_effective_start_does_not_shift_segment_timeline(tmp_path):
    base = annotated_population(tmp_path)
    path = tmp_path / "data/meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(path).to_pylist()
    for ep in episodes:
        annotation = json.loads(ep["annotation"])
        annotation["effective_start_time"] = 10 / 30
        ep["annotation"] = json.dumps(annotation)
    pq.write_table(pa.Table.from_pylist(episodes), path)
    raw = configured(base, tmp_path, "segment")
    assert raw[0]["window_start_frame"] == 10
    assert raw[0]["ai_caption"] == "Place the block"


def test_missing_segments_and_fractional_boundaries():
    for value in (None, [], "", "  "):
        assert segment_text_intervals({"segments": value}, 30) == []
    assert segment_text_intervals({}, 30) == []
    intervals = segment_text_intervals({"segments": [segment("0.05", "0.15")]}, 30)
    assert [select_text("episode", intervals, frame, "segment") for frame in range(6)] == [
        "episode",
        "episode",
        "move",
        "move",
        "move",
        "episode",
    ]


def test_text_resume_contract_includes_captions_timing_and_mode(tmp_path):
    base = annotated_population(tmp_path)
    raw = configured(base, tmp_path, "segment", tails=False)
    # Text must be protected even without mixture goals, tail windows or buckets.
    raw.random_goal_sampling = False

    def loader():
        return StatefulWindowLoader(LeRobotPolicySFTDataset(raw))

    saved = loader().state_dict()
    assert "text_conditioning" in saved and "tail_windows" not in saved and "goal_sampling" not in saved
    assert raw.selection_record()["text_conditioning"] == saved["text_conditioning"]
    loader().load_state_dict(saved)
    original = copy.deepcopy(raw.episodes[0])
    for intervals in ([(0, 8, "Changed caption")], [(1, 8, "Pick up the block")]):
        raw.episodes[0]["text_segments"] = intervals
        with pytest.raises(ValueError, match="Text conditioning changed"):
            loader().load_state_dict(saved)
    raw.episodes[0] = original
    raw.text_conditioning = "episode_segment"
    with pytest.raises(ValueError, match="Text conditioning changed"):
        loader().load_state_dict(saved)
    raw.text_conditioning = "episode"
    legacy = loader().state_dict()
    assert "text_conditioning" not in legacy and "text_conditioning" not in raw.selection_record()
    loader().load_state_dict(legacy)
    with pytest.raises(ValueError, match="Text conditioning changed"):
        loader().load_state_dict(saved)
    raw.text_conditioning = "segment"
    with pytest.raises(ValueError, match="Text conditioning changed"):
        loader().load_state_dict(legacy)


def test_config_and_cli_text_modes():
    values = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/pretrain.yaml").read_text())["data"]
    assert _instantiate_recursive(GoalWAMDataArguments, values).text_conditioning == "episode"
    parser = argparse.ArgumentParser()
    _add_arguments_recursive(parser, VeOmniGoalWAMArguments)
    for mode in ("episode", "segment", "episode_segment"):
        cli = vars(parser.parse_args(["--data.text_conditioning", mode]))
        assert cli["data.text_conditioning"] == mode
        assert GoalWAMDataArguments(train_path="unused.yaml", text_conditioning=mode).text_conditioning == mode
    with pytest.raises(ValueError, match="text_conditioning"):
        GoalWAMDataArguments(train_path="unused.yaml", text_conditioning="goal_segment")
