"""Goal diversity, uncertain annotations, target integrity and exact stream resume."""

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from torch.utils.data import Dataset

from recipes.GoalWAM.data.data_loader import StatefulWindowLoader
from recipes.GoalWAM.data.dataset import LeRobotPolicyDataset, LeRobotPolicySFTDataset
from recipes.GoalWAM.data.goal_sampling import (
    GoalSamplingConfig,
    episode_goal_timing,
    resolve_goal_sampling,
    select_goal,
)
from recipes.GoalWAM.tests.test_tail_windows import population
from recipes.GoalWAM.trainer.arguments import GoalWAMDataArguments, VeOmniGoalWAMArguments
from recipes.GoalWAM.trainer.evaluator import inference_sample
from veomni.arguments.parser import _add_arguments_recursive, _instantiate_recursive


def draw(config=None, occurrence=0, **kwargs):
    args = dict(start=0, end=600, fps=30, rate=1, horizon=48, endpoints=[], seed=42, training=True)
    args.update(kwargs)
    return select_goal(resolve_goal_sampling(config or {"mode": "mixture"}), occurrence=occurrence, **args)


def only(source, **kwargs):
    return dict(
        mode="mixture", weights={key: float(key == source) for key in ("terminal", "segment", "future")}, **kwargs
    )


def dataset(tmp_path, config=None, *, training=True, tails=True):
    base, _ = population(tmp_path, tails=tails)
    raw = LeRobotPolicyDataset(
        base.manifest,
        {"robotwin_aloha_agilex": str(tmp_path / "norm.json")},
        training=training,
        include_tail_windows=tails,
        img_size=[32, 32],
        goal_sampling=config,
    )
    raw._episode_rows, raw._video = base._episode_rows, base._video
    return raw


@pytest.mark.parametrize("mode", ["terminal", "mixture"])
def test_recipe_config_defaults_and_nested_cli(mode):
    path = Path(__file__).resolve().parents[1] / "configs/robotwin.yaml"
    values = yaml.safe_load(path.read_text())["data"]
    # Exercise both supported modes without depending on the active experiment.
    values["goal_sampling"]["mode"] = mode
    args = _instantiate_recursive(GoalWAMDataArguments, values)
    assert args.goal_sampling.mode == mode
    assert args.goal_sampling.weights == dict(terminal=0.30, segment=0.25, future=0.45)
    assert args.goal_sampling.segment.boundary_jitter_seconds == 0.5
    assert GoalWAMDataArguments(train_path="unused.yaml").goal_sampling.mode == "terminal"
    parser = argparse.ArgumentParser()
    _add_arguments_recursive(parser, VeOmniGoalWAMArguments)
    cli = vars(parser.parse_args(["--data.goal_sampling.future.max_offset_seconds", "6.0"]))
    assert cli["data.goal_sampling.future.max_offset_seconds"] == 6.0


@pytest.mark.parametrize(
    "value",
    [
        {"mode": "random"},
        {"weights": {"terminal": 1}},
        {"weights": dict(terminal=0.5, segment=0.5, future=0.5)},
        {"weights": dict(terminal=-0.1, segment=0.5, future=0.6)},
        {"weights": dict(terminal=float("nan"), segment=0.5, future=0.5)},
        {"weights": dict(terminal=True, segment=0, future=0)},
        {"future": {"max_offset_seconds": float("inf")}},
        {"future": {"min_horizon_multiple": 0.5}},
        {"segment": {"boundary_jitter_seconds": -1}},
        {"segment": {"max_future_endpoints": 1.5}},
        {"segment": {"max_future_endpoints": True}},
        {"segment": {"endpoint_rank_decay": 0}},
        {"segment": {"endpoint_rank_decay": 2}},
        {"fallback": dict(segment="future", future="segment")},
        {"min_offset": "current"},
        {"eval_mode": "mixture"},
        {"future": {"typo": 1}},
        {"segment": None},
        {"typo": 1},
    ],
)
def test_invalid_goal_configs(value):
    with pytest.raises(ValueError, match="goal_sampling"):
        resolve_goal_sampling(value)


def test_terminal_and_eval_are_exact_and_do_not_use_global_rng():
    state = random.getstate()
    for i in range(100):
        assert draw({"mode": "terminal"}, i)["goal_index"] == 599
        assert draw(only("future"), i, training=False)["goal_index"] == 599
    assert random.getstate() == state


def test_terminal_weights_bypass_goal_rng(monkeypatch):
    def unexpected_rng(*args, **kwargs):
        raise AssertionError("Terminal-only weights must not instantiate a goal RNG")

    monkeypatch.setattr("recipes.GoalWAM.data.goal_sampling.random.Random", unexpected_rng)
    for occurrence in (0, 17, 10000):
        assert draw(only("terminal"), occurrence) == draw({"mode": "terminal"}, occurrence)


def assert_same_values(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same_values(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_same_values(a, e)
    else:
        assert actual == expected


@pytest.mark.parametrize("tails", [False, True])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_terminal_weights_match_legacy_samples_stream_and_resume(tmp_path, monkeypatch, tails, dropout):
    from cosmos_framework.data.vfm.augmentors import text_tokenizer

    from recipes.GoalWAM.tests.test_training_conditioning import TinyTokenizer

    monkeypatch.setattr(text_tokenizer, "lazy_instantiate", lambda _: TinyTokenizer())
    actual = dataset(tmp_path, only("terminal"), tails=tails)
    legacy = LeRobotPolicyDataset(
        actual.manifest,
        {"robotwin_aloha_agilex": str(tmp_path / "norm.json")},
        training=True,
        include_tail_windows=tails,
        img_size=[32, 32],
    )
    legacy._episode_rows, legacy._video = actual._episode_rows, actual._video
    assert actual.goal_sampling.mode == "mixture"
    assert not actual.random_goal_sampling
    assert actual.goal_sampling_record() is None
    assert actual.selection_record() == legacy.selection_record()
    count = actual.episodes[0]["count"]
    for index in (0, count // 2, count - 1):
        assert_same_values(actual[index], legacy[index])
        ep, _ = actual.locate(index)
        assert actual[index]["goal_frame_index"] == ep["end"] - 1

    loaders = [
        StatefulWindowLoader(
            LeRobotPolicySFTDataset(raw, tokenizer_config={}, cfg_dropout_rate=dropout),
            batch_size=2,
        )
        for raw in (actual, legacy)
    ]
    changed, previous = loaders
    assert changed.state_dict() == previous.state_dict()
    assert changed.sampler.include_position == previous.sampler.include_position == bool(dropout)
    if dropout:
        assert not changed.loader.dataset.include_goal_position
    else:
        assert changed.loader.dataset is changed.dataset
    # Resume in both directions from a consumed cursor, not the worker request count.
    previous.sampler.cursor = 7
    changed.load_state_dict(previous.state_dict())
    previous.load_state_dict(changed.state_dict())
    before = random.getstate()
    streams = [iter(loader) for loader in loaders]
    try:
        for _ in range(16):
            a, e = [next(iterator) for iterator in streams]
            for key in (
                "video",
                "action",
                "action_valid_mask",
                "ai_caption",
                "text_token_ids",
                "sequence_plan",
                "goal_frame_index",
                "domain_id",
                "fps",
                "action_fps",
            ):
                assert_same_values(a[key], e[key])
    finally:
        for iterator in streams:
            iterator.close()
    assert random.getstate() == before
    assert changed.state_dict() == previous.state_dict()


def test_mixture_fallback_probabilities_and_uniform_future_support():
    samples = [draw(occurrence=i) for i in range(8000)]
    requested = Counter(s["goal_requested_source"] for s in samples)
    resolved = Counter(s["goal_source"] for s in samples)
    for source, probability in dict(terminal=0.3, segment=0.25, future=0.45).items():
        assert requested[source] / len(samples) == pytest.approx(probability, abs=0.02)
    assert resolved["future"] == requested["future"] + requested["segment"]
    future = [s["goal_index"] for s in samples if s["goal_source"] == "future"]
    assert set(future) == set(range(48, 121))
    assert np.mean(future) == pytest.approx(84, abs=1)
    assert all(
        s["goal_fallback_reason"] == "no_eligible_segment" for s in samples if s["goal_requested_source"] == "segment"
    )
    fallback = draw(only("segment", fallback=dict(segment="terminal", future="terminal")))
    assert fallback["goal_source"] == "terminal" and fallback["goal_index"] == 599


def test_segment_jitter_intersections_and_endpoint_rank_sampling():
    # Endpoint 10 has no legal support. Endpoint 48 is clipped to [48,53],
    # endpoint 70 has [65,75]; endpoint 90 is beyond the first two eligible.
    config = only("segment")
    samples = [draw(config, i, fps=10, endpoints=[10, 48, 70, 90]) for i in range(5000)]
    by_endpoint = {j: [s["goal_index"] for s in samples if s["goal_segment_index"] == j] for j in (1, 2)}
    assert set(by_endpoint[1]) == set(range(48, 54))
    assert set(by_endpoint[2]) == set(range(65, 76))
    assert len(by_endpoint[1]) / len(samples) == pytest.approx(2 / 3, abs=0.03)
    # Clipping must not accumulate out-of-range probability on row 48.
    assert by_endpoint[1].count(48) / len(by_endpoint[1]) == pytest.approx(1 / 6, abs=0.03)
    exact = draw(only("segment", segment={"boundary_jitter_seconds": 0}), endpoints=[60])
    assert exact["goal_index"] == 60
    # An endpoint outside effective time is usable only if its jitter overlaps.
    clipped = draw(config, end=80, fps=10, endpoints=[84])
    assert clipped["goal_index"] == 79 and clipped["goal_source"] == "segment"
    assert draw(config, end=80, fps=10, endpoints=[100])["goal_source"] == "future"


def test_acceleration_and_tail_rules():
    samples = [draw(only("future"), i, rate=3) for i in range(1000)]
    assert min(s["goal_index"] for s in samples) == 144
    assert max(s["goal_index"] for s in samples) == 288
    assert max(s["goal_delay_seconds"] for s in samples) == 9.6
    for start in (20, 48, 63):
        for source in ("terminal", "future", "segment"):
            result = draw(only(source), start=start, end=64, endpoints=[63])
            assert result["goal_index"] == 63
            assert result["goal_delay_seconds"] == (63 - start) / 30


def segment(lo, hi):
    return dict(start_time=lo, end_time=hi, caption={"overall": {"en": "move", "zh": "移动"}})


@pytest.mark.parametrize("missing", [None, "", "  "])
def test_optional_annotation_values(missing):
    annotation = dict(duration=10, effective_start_time=missing, effective_end_time=missing, segments=missing)
    assert episode_goal_timing(annotation, 300, 30, context="test") == (0, 300, [])
    annotation.update(effective_start_time=1.1, segments=[])
    assert episode_goal_timing(annotation, 300, 30, context="test") == (33, 300, [])
    annotation.update(effective_start_time=missing, effective_end_time=9.1)
    assert episode_goal_timing(annotation, 300, 30, context="test") == (0, 273, [])


def test_segment_timebase_and_gaps():
    annotation = dict(duration=10, segments=[segment(0, 2.1), segment(4, 6.5)])
    assert episode_goal_timing(annotation, 300, 30, context="test") == (0, 300, [62, 194])
    # At t=3s the anchor lies in an annotation gap; next eligible end still works.
    goal = draw(only("segment", segment={"boundary_jitter_seconds": 0}), start=90, end=300, endpoints=[62, 194])
    assert goal["goal_index"] == 194


@pytest.mark.parametrize(
    "update",
    [
        {"effective_start_time": []},
        {"effective_start_time": False},
        {"effective_end_time": float("nan")},
        {"effective_end_time": 11},
        {"effective_start_time": 5, "effective_end_time": 4},
        {"segments": {}},
        {"segments": [None]},
        {"segments": [segment(2, 1)]},
        {"segments": [segment(0, 3), segment(2, 4)]},
        {"segments": [segment(0, float("inf"))]},
        {"segments": [dict(start_time=0, end_time=2)]},
    ],
)
def test_malformed_nonempty_annotations_fail(update):
    with pytest.raises(ValueError):
        episode_goal_timing(dict(duration=10, **update), 300, 30, context="test")


def test_decoder_targets_and_inference_conditions(tmp_path):
    raw = dataset(tmp_path, only("future"))
    reference = raw.physical_sample(0)
    goals = set()
    calls = []
    video = raw._video

    def read(ep, camera, indices):
        calls.append((camera, indices))
        return video(ep, camera, indices)

    raw._video = read
    sft = LeRobotPolicySFTDataset(raw)
    for occurrence in range(20):
        key = (0, occurrence)
        sample = raw.physical_sample(key)
        for name in ("action", "action_valid_mask", "action_indices", "video_indices", "action_time_valid"):
            torch.testing.assert_close(sample[name], reference[name], rtol=0, atol=0)
        prepared = sft[key]
        goal = prepared["goal_frame_index"]
        goals.add(goal)
        assert all(indices[-1] == goal for _, indices in calls[-3:])
        pixels = prepared["video_pixel_mask"]
        assert prepared["video"][0][:, 0][:, pixels].eq(goal).all()
        assert not prepared["video"][0][:, 0][:, ~pixels].any()
        assert prepared["sequence_plan"].vision_item_roles == ["goal", "default"]
        assert prepared["sequence_plan"].condition_frame_indexes_action == [0]
        inference = inference_sample(prepared)
        torch.testing.assert_close(inference["video"][0], prepared["video"][0])
        assert not inference["video"][-1][:, 1:].any()
        assert not inference["action"][1:].any()
    assert len(goals) > 5
    # Physical and decoded integer previews always select the same goal.
    assert raw[0]["goal_frame_index"] == reference["goal_index"]


def test_dataset_accepts_null_optional_annotations_and_eval_stays_terminal(tmp_path):
    raw = dataset(tmp_path)
    path = tmp_path / "data/meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(path).to_pylist()
    for ep in episodes:
        annotation = json.loads(ep["annotation"])
        annotation.update(segments=None, effective_start_time="", effective_end_time=None)
        ep["annotation"] = json.dumps(annotation)
    pq.write_table(pa.Table.from_pylist(episodes), path)
    evaluated = LeRobotPolicyDataset(
        raw.manifest,
        {"robotwin_aloha_agilex": str(tmp_path / "norm.json")},
        training=False,
        goal_sampling=only("future"),
    )
    evaluated._episode_rows = raw._episode_rows
    assert evaluated.physical_sample((0, 15))["goal_index"] == 63
    assert not evaluated.random_goal_sampling and evaluated.goal_sampling_record() is None


def test_resume_contract_covers_annotations_without_tail_windows(tmp_path):
    raw = dataset(tmp_path, {"mode": "mixture"}, tails=False)
    loader = StatefulWindowLoader(LeRobotPolicySFTDataset(raw))
    saved = loader.state_dict()
    assert "goal_sampling" in saved and "tail_windows" not in saved
    assert "goal_sampling" in raw.selection_record()
    restored = StatefulWindowLoader(LeRobotPolicySFTDataset(raw))
    restored.load_state_dict(saved)
    manifest_digest = raw.manifest_sha256
    raw.manifest_sha256 = "changed-manifest"
    with pytest.raises(ValueError, match="Goal sampling changed"):
        StatefulWindowLoader(LeRobotPolicySFTDataset(raw)).load_state_dict(saved)
    raw.manifest_sha256 = manifest_digest
    raw.episodes[0]["goal_endpoints"] = [60]
    with pytest.raises(ValueError, match="Goal sampling changed"):
        StatefulWindowLoader(LeRobotPolicySFTDataset(raw)).load_state_dict(saved)
    raw.episodes[0]["goal_endpoints"] = []
    raw.goal_sampling.future.max_offset_seconds = 5.0
    with pytest.raises(ValueError, match="Goal sampling changed"):
        StatefulWindowLoader(LeRobotPolicySFTDataset(raw)).load_state_dict(saved)
    raw.random_goal_sampling = False
    raw.goal_sampling = GoalSamplingConfig()
    legacy = StatefulWindowLoader(LeRobotPolicySFTDataset(raw))
    assert "goal_sampling" not in legacy.state_dict()
    assert "goal_sampling" not in raw.selection_record()
    legacy.load_state_dict(legacy.state_dict())
    with pytest.raises(ValueError, match="Goal sampling changed"):
        legacy.load_state_dict(saved)
    with pytest.raises(ValueError, match="Goal sampling changed"):
        restored.load_state_dict(legacy.state_dict())


class GoalStreamDataset(Dataset):
    """Small spawn-picklable dataset exercising production occurrence plumbing."""

    def __init__(self, dropout):
        self.cfg_dropout_rate = dropout
        self._dataset = self
        self.random_goal_sampling = True

    def __len__(self):
        return 11

    def goal_sampling_record(self):
        return {"version": 1, "config": {"mode": "mixture"}}

    def __getitem__(self, key):
        index, occurrence = key
        goal = draw(occurrence=occurrence)
        return dict(index=index, occurrence=occurrence, dropped=random.random() < self.cfg_dropout_rate, **goal)


def stream(workers, dropout, count, state=None, *, prefetch_factor=2):
    loader = StatefulWindowLoader(
        GoalStreamDataset(dropout), batch_size=2, num_workers=workers, prefetch_factor=prefetch_factor
    )
    if workers:
        loader.loader.multiprocessing_context = "spawn"
    if state is not None:
        loader.load_state_dict(state)
    iterator = iter(loader)
    samples = []
    try:
        for _ in range(count):
            batch = next(iterator)
            samples.extend(
                {
                    key: batch[key][i].item() if isinstance(batch[key][i], torch.Tensor) else batch[key][i]
                    for key in batch
                }
                for i in range(2)
            )
        saved = copy.deepcopy(loader.state_dict())
    finally:
        iterator.close()
        if workers:
            loader.loader._iterator._shutdown_workers()
    return samples, saved


def test_occurrences_reproduce_across_workers_resume_and_dropout():
    random_state = random.getstate()
    reference, _ = stream(0, 0.1, 40)
    parallel, _ = stream(2, 0.1, 40)
    assert parallel == reference
    prefix, saved = stream(2, 0.1, 7, prefetch_factor=1)
    resumed, _ = stream(2, 0.1, 33, saved, prefetch_factor=4)
    assert prefix + resumed == reference
    no_dropout, _ = stream(0, 0.0, 40)
    assert random.getstate() == random_state
    for expected, actual in zip(reference, no_dropout, strict=True):
        assert {k: v for k, v in expected.items() if k != "dropped"} == {
            k: v for k, v in actual.items() if k != "dropped"
        }
        digest = hashlib.sha256(f"goalwam-caption-v1:42:{expected['occurrence']}".encode()).digest()
        assert expected["dropped"] == (random.Random(digest).random() < 0.1)
    assert any(len({s["goal_index"] for s in reference if s["index"] == index}) > 2 for index in range(11))
