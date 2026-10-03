# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmos_framework.data.vfm.local_datasets import episode_image_edit_dataset as dataset_module
from cosmos_framework.data.vfm.local_datasets.episode_image_edit_dataset import (
    EpisodeImageEditDataset,
    _compose_three_camera_frame,
    _Episode,
    _segment_target_index,
    _subgoal_target_segment_index,
    _three_camera_concat_spec,
)


def _segments(*ranges: tuple[int, int]) -> tuple[tuple[int, int, bool, str, str], ...]:
    return tuple((start, end, False, f"subtask-{index}", f"cot-{index}") for index, (start, end) in enumerate(ranges))


def test_last_fifteen_percent_targets_next_segment() -> None:
    segments = _segments((0, 20), (20, 40), (40, 60))

    assert _segment_target_index(segments, 16, 60) == (0, 19)
    assert _segment_target_index(segments, 17, 60) == (1, 39)
    assert _segment_target_index(segments, 19, 60) == (1, 39)
    assert _segment_target_index(segments, 36, 60) == (1, 39)
    assert _segment_target_index(segments, 37, 60) == (2, 59)


def test_tail_rounds_up_to_whole_frames() -> None:
    segments = _segments((0, 7), (7, 14))

    assert _subgoal_target_segment_index(segments, 4) == 0
    assert _subgoal_target_segment_index(segments, 5) == 1
    assert _subgoal_target_segment_index(segments, 6) == 1


def test_tail_fraction_is_configurable_and_zero_disables_it() -> None:
    segments = _segments((0, 20), (20, 40))

    assert _segment_target_index(segments, 14, 40, 0.25) == (0, 19)
    assert _segment_target_index(segments, 15, 40, 0.25) == (1, 39)
    assert _segment_target_index(segments, 19, 40, 0.0) == (0, 19)
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        _segment_target_index(segments, 0, 40, 1.01)


def test_three_camera_composition_matches_inverted_t_layout() -> None:
    head = np.full((4, 8, 3), 10, dtype=np.uint8)
    left = np.full((8, 16, 3), 20, dtype=np.uint8)
    right = np.full((8, 16, 3), 30, dtype=np.uint8)

    composed = _compose_three_camera_frame(head, left, right, target_hw=(6, 8))

    assert composed.shape == (6, 8, 3)
    # Before the final resize, head occupies the full top 4x8 and hands occupy 2x4 each below it.
    assert np.all(composed[:4, :, :] == 10) or np.all(composed[:3, :, :] == 10)
    assert composed[-1, 0, 0] < composed[-1, -1, 0]


def test_robotwin_three_camera_uses_portrait_offline_stream() -> None:
    output_key, target_hw, aspect_ratio = _three_camera_concat_spec("observation.images.cam_high")

    assert output_key == "observation.images.concat_view_384x320"
    assert target_hw == (384, 320)
    assert aspect_ratio == "3,4"


def test_legacy_action_annotations_are_an_explicit_and_auto_fallback(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    # The fixed-goal wrapper is intentionally not the legacy action_steps schema.
    (meta / "annotations.json").write_text(
        json.dumps({"schema": "fixedgoal/v1", "episodes": {"0": {"segments": []}}}),
        encoding="utf-8",
    )
    (meta / "legacy_action_annotations.json").write_text(
        json.dumps(
            {
                "0": {
                    "action_steps": [
                        {
                            "start_frame": 0,
                            "end_frame": 10,
                            "action_text": "pick up the object",
                            "is_mistake": False,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    expected = {0: ((0, 10, False, "pick up the object", ""),)}
    assert dataset_module._load_lerobot_segments(meta, "legacy_action_annotations") == expected
    assert dataset_module._load_lerobot_segments(meta, "auto") == expected


def test_fixed_goal_annotations_segments_are_supported(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "annotations.json").write_text(
        json.dumps(
            {
                "schema": "robotwin-seg1-clean-fixedgoal/v1",
                "episodes": {
                    "42": {
                        "segments": [
                            {
                                "start_frame": 0,
                                "end_frame_exclusive": 115,
                                "goal_frame": 114,
                                "stage_text": "Raise the laptop lid.",
                            },
                            {
                                "start_frame": 115,
                                "end_frame_exclusive": 262,
                                "goal_frame": 261,
                                "stage_text": "Open the laptop fully.",
                            },
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert dataset_module._load_lerobot_segments(meta, "annotations") == {
        42: (
            (0, 115, False, "Raise the laptop lid.", ""),
            (115, 262, False, "Open the laptop fully.", ""),
        )
    }


def test_final_segment_tail_targets_its_own_final_frame() -> None:
    segments = _segments((0, 20), (20, 40))

    assert _segment_target_index(segments, 36, 40) == (1, 39)
    assert _segment_target_index(segments, 37, 40) == (1, 39)
    assert _segment_target_index(segments, 39, 40) == (1, 39)


def test_generator_only_sample_uses_main_task_not_subtask(monkeypatch) -> None:
    segments = _segments((0, 20), (20, 40))
    episode = _Episode(
        episode_id=3,
        video_path=Path("episode3.mp4"),
        instruction_path=Path("episodes.jsonl"),
        width=2,
        height=2,
        fps=30.0,
        total_frames=40,
        instructions=("Assemble the object",),
        aspect_ratio="1,1",
        target_width=2,
        target_height=2,
        segments=segments,
        segment_ends=(20, 40),
    )
    dataset = EpisodeImageEditDataset.__new__(EpisodeImageEditDataset)
    dataset.samples = [(0, 17)]
    dataset.episodes = [episode]
    dataset.seed = 42
    dataset.target_mode = "segment_final"
    dataset.reasoner_subtask_target = False
    dataset.next_subgoal_tail_fraction = 0.15
    dataset.cfg_dropout_rate = 0.0
    dataset.dataset_name = "test"
    dataset.resize_resolution = "2"
    dataset._tokenize_caption = lambda caption: torch.tensor([len(caption)])
    monkeypatch.setattr(
        dataset_module,
        "_decode_two_frames",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2, 3), dtype=np.uint8),
        ),
    )

    sample = dataset[0]

    assert sample["ai_caption"] == "Assemble the object"
    assert sample["reasoner_prompt"] == "Assemble the object"
    assert "subtask-1" not in sample["ai_caption"]
    assert sample["target_segment_index"] == 1
    assert sample["target_frame_index"] == 39


def test_episode_final_mode_always_targets_last_frame(monkeypatch) -> None:
    episode = _Episode(
        episode_id=3,
        video_path=Path("episode3.mp4"),
        instruction_path=Path("episodes.jsonl"),
        width=2,
        height=2,
        fps=30.0,
        total_frames=10,
        instructions=("Assemble the object",),
        aspect_ratio="1,1",
        target_width=2,
        target_height=2,
    )
    dataset = EpisodeImageEditDataset.__new__(EpisodeImageEditDataset)
    dataset.samples = [(0, 4)]
    dataset.episodes = [episode]
    dataset.seed = 42
    dataset.target_mode = "episode_final"
    dataset.reasoner_subtask_target = False
    dataset.cfg_dropout_rate = 0.0
    dataset.dataset_name = "test"
    dataset.resize_resolution = "2"
    dataset._tokenize_caption = lambda caption: torch.tensor([len(caption)])
    decoded = {}

    def fake_decode(*args, **kwargs):
        decoded.update(kwargs)
        return (
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2, 3), dtype=np.uint8),
        )

    monkeypatch.setattr(dataset_module, "_decode_two_frames", fake_decode)

    sample = dataset[0]

    assert decoded["frame_idx"] == 4
    assert decoded["final_idx"] == 9
    assert sample["target_frame_index"] == 9
    assert sample["num_segments"] == 1


def test_reasoner_sample_keeps_subtask_out_of_ai_caption(monkeypatch) -> None:
    segments = _segments((0, 20), (20, 40))
    episode = _Episode(
        episode_id=3,
        video_path=Path("episode3.mp4"),
        instruction_path=Path("episodes.jsonl"),
        width=2,
        height=2,
        fps=30.0,
        total_frames=40,
        instructions=("Assemble the object",),
        aspect_ratio="1,1",
        target_width=2,
        target_height=2,
        segments=segments,
        segment_ends=(20, 40),
    )
    dataset = EpisodeImageEditDataset.__new__(EpisodeImageEditDataset)
    dataset.samples = [(0, 17)]
    dataset.episodes = [episode]
    dataset.seed = 42
    dataset.target_mode = "segment_final"
    dataset.reasoner_subtask_target = True
    dataset.reasoner_target_format = "cot_and_subtask"
    dataset.next_subgoal_tail_fraction = 0.15
    dataset.cfg_dropout_rate = 0.0
    dataset.dataset_name = "test"
    dataset.resize_resolution = "2"
    dataset._tokenize_prompt_and_target = lambda prompt, cot, subtask: (
        torch.tensor([1, 2, 3]),
        1,
        f"{cot} -> {subtask}",
    )
    monkeypatch.setattr(
        dataset_module,
        "_decode_two_frames",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2, 3), dtype=np.uint8),
        ),
    )

    sample = dataset[0]

    assert sample["ai_caption"] == "Assemble the object"
    assert sample["reasoner_prompt"] == "Assemble the object"
    assert sample["reasoner_target_text"] == "cot-1 -> subtask-1"
    assert sample["source_segment_index"] == 0
    assert sample["target_segment_index"] == 1
    assert sample["target_frame_index"] == 39
    assert sample["source_segment_start_frame"] == 0
    assert sample["source_segment_end_frame_exclusive"] == 20
    assert sample["target_segment_start_frame"] == 20
    assert sample["target_segment_end_frame_exclusive"] == 40
    assert sample["num_segments"] == 2


def test_subtask_only_reasoner_target_does_not_require_cot() -> None:
    dataset = EpisodeImageEditDataset.__new__(EpisodeImageEditDataset)
    dataset.reasoner_target_format = "subtask"
    dataset.max_caption_tokens = 32
    dataset._tokenizer = type(
        "Tokenizer",
        (),
        {"encode": staticmethod(lambda text, add_special_tokens=False: [len(text)])},
    )()
    original = dataset_module.tokenize_caption
    dataset_module.tokenize_caption = lambda *args, **kwargs: [1, 2]
    try:
        token_ids, prompt_tokens, target_text = dataset._tokenize_prompt_and_target(
            "Do the task", "", "Pick up the cup"
        )
    finally:
        dataset_module.tokenize_caption = original

    assert token_ids.tolist() == [1, 2, 15]
    assert prompt_tokens == 2
    assert target_text == "Pick up the cup"


def test_selects_one_episode_per_canonical_task() -> None:
    dataset = EpisodeImageEditDataset.__new__(EpisodeImageEditDataset)
    dataset.episodes_per_task = 1
    dataset.episodes_per_dataset = None
    dataset.max_episode_index = 2
    dataset.seed = 4242
    base = dict(
        video_path=Path("episode.mp4"),
        instruction_path=Path("episodes.jsonl"),
        width=2,
        height=2,
        fps=30.0,
        total_frames=2,
        instructions=("task",),
        aspect_ratio="1,1",
        target_width=2,
        target_height=2,
    )
    episodes = [
        _Episode(episode_id=0, task_name="task-a", **base),
        _Episode(episode_id=1, task_name="task-a", **base),
        _Episode(episode_id=2, task_name="task-b", **base),
        _Episode(episode_id=3, task_name="task-b", **base),
    ]

    selected = dataset._select_eval_episodes(episodes)

    assert len(selected) == 2
    assert {episode.task_name for episode in selected} == {"task-a", "task-b"}
    assert all(episode.episode_id <= 2 for episode in selected)
