"""Episode-end anchors, real video prefixes and masked native action learning."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from cosmos_framework.data.vfm.action.transforms import build_sequence_plan_from_mode
from cosmos_framework.model.vfm.algorithm.loss.flow_matching import compute_flow_matching_loss

from recipes.GoalWAM.data.data_loader import StatefulWindowLoader, collate_samples
from recipes.GoalWAM.data.dataset import LeRobotPolicyDataset, LeRobotPolicySFTDataset
from recipes.GoalWAM.tests.test_manifests import write_population
from recipes.GoalWAM.trainer.evaluator import inference_sample
from recipes.GoalWAM.trainer.metrics import image_metrics


def population(tmp_path, *, rate=1, start=0, end=64, tails=True):
    manifest, norms = write_population(tmp_path)
    config = yaml.safe_load(manifest.read_text())
    config["task"].update(acceleration_rate=rate, sampling_rate=1)
    manifest.write_text(yaml.safe_dump(config))
    path = tmp_path / "data/meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(path).to_pylist()
    for episode in episodes:
        annotation = json.loads(episode["annotation"])
        annotation.update(effective_start_time=start / 30, effective_end_time=end / 30)
        episode["annotation"] = json.dumps(annotation)
    pq.write_table(pa.Table.from_pylist(episodes), path)
    raw = LeRobotPolicyDataset(manifest, norms, training=True, include_tail_windows=tails, img_size=[32, 32])
    state = torch.zeros(64, 49)
    state[:, [32, 40, 48]] = 1
    state[:, 0] = torch.arange(64) / 100
    actions = state.clone()
    actions[:, 0] += 0.005
    actions[:, [7, 15, 33, 41]] = torch.arange(64, dtype=torch.float32)[:, None]
    raw._episode_rows = lambda _: {"observation.state": state, "action": actions}

    def video(ep, camera, indices):
        assert all(ep["start"] <= i < ep["end"] for i in indices)
        return np.broadcast_to(
            np.array(indices, dtype=np.uint8)[:, None, None, None], (len(indices), 32, 32, 3)
        ).copy()

    raw._video = video
    return raw, actions


@pytest.mark.parametrize("rate", [1, 3])
@pytest.mark.parametrize(("start", "end"), [(0, 64), (5, 43), (20, 21)])
def test_every_effective_anchor_and_no_fabricated_targets(tmp_path, rate, start, end):
    raw, actions = population(tmp_path, rate=rate, start=start, end=end)
    assert len(raw) == 7 * (end - start)
    assert {raw.locate(i)[1] for i in range(end - start)} == set(range(start, end))
    for index in range(end - start):
        sample = raw.physical_sample(index)
        valid_indices = list(range(start + index, end, rate))[:48]
        count = len(valid_indices)
        assert sample["action_time_valid"].sum() == count
        torch.testing.assert_close(sample["absolute_actions"][:count], actions[valid_indices])
        assert not sample["action_valid_mask"][1 + count :].any()
        assert not sample["action"][1 + count :].any()
        assert not sample["relative_actions"][count:].any()
        frames = sample["video_indices"].tolist()
        assert frames[0] == start + index and frames[-1] < end
        assert len(frames) in (1, 5, 9, 13)
        assert all(b - a == 4 * rate for a, b in zip(frames, frames[1:]))
    final = raw[end - start - 1]
    assert final["video"].shape[1] == 1
    assert final["action_valid_mask"][1, 0]
    assert final["action"][1, 0].item() == pytest.approx(0.005, abs=1e-6)
    assert final["action"][1, 7].item() == pytest.approx(end - 1, rel=2e-6)
    assert not final["action_valid_mask"][2:].any()


def test_full_windows_unchanged_and_partial_frames_never_reach_vae(tmp_path):
    raw, _ = population(tmp_path)
    old = LeRobotPolicyDataset(raw.manifest, {"robotwin_aloha_agilex": str(tmp_path / "norm.json")}, img_size=[32, 32])
    old._episode_rows, old._video = raw._episode_rows, raw._video
    for index in range(16):
        for key in ("action", "action_valid_mask", "video", "goal_frame"):
            torch.testing.assert_close(raw[index][key], old[index][key], rtol=0, atol=0)
    sft = LeRobotPolicySFTDataset(raw)
    samples = [sft[index] for index in (0, 16, 32, 48, 63)]
    assert [s["video"][-1].shape[1] for s in samples] == [13, 9, 5, 1, 1]
    for sample in samples:
        plan = sample["sequence_plan"]
        assert plan.condition_frame_indexes_action == plan.condition_frame_indexes_vision == [0]
        assert plan.vision_item_roles == ["goal", "default"]
        assert sample["action"].shape == (49, 64)
        inference = inference_sample(sample)
        assert not inference["action"][1:].any()
        assert not inference["video"][-1][:, 1:].any()
        torch.testing.assert_close(inference["video"][0], sample["video"][0])
    batch = collate_samples(samples)
    assert [items[-1].shape[1] for items in batch["video"]] == [13, 9, 5, 1, 1]


def test_native_loss_learns_last_command_and_ignores_padded_rows(tmp_path):
    raw, _ = population(tmp_path)
    sample = LeRobotPolicySFTDataset(raw)[63]
    prediction = torch.ones_like(sample["action"], requires_grad=True)
    target = torch.zeros_like(prediction)
    valid = sample["action_valid_mask"]
    target[~valid] = float("nan")
    condition = torch.zeros(49, 1)
    condition[0] = 1
    rf = SimpleNamespace(train_time_weight=lambda ts, _: torch.ones_like(ts))
    loss, _ = compute_flow_matching_loss(
        [prediction],
        [target],
        [condition],
        torch.ones(1, 1),
        True,
        rf,
        {},
        raw_action_dim=[49],
        action_valid_mask=[valid],
    )
    assert loss.item() == 1
    loss.backward()
    assert prediction.grad[1, valid[1]].abs().min() > 0
    assert not prediction.grad[0].any() and not prediction.grad[2:].any()
    assert not prediction.grad[~valid].any()
    # A goal + current-only rollout has no noisy vision target.
    vision = [torch.ones(2, 1, 2, 2, requires_grad=True) for _ in range(2)]
    video_loss, _ = compute_flow_matching_loss(
        vision,
        vision,
        [torch.ones(1, 1, 1)] * 2,
        torch.ones(2, 1),
        False,
        rf,
        {},
        exclude_fully_conditioned_items=True,
    )
    video_loss.backward()
    assert video_loss.item() == 0 and all(not item.grad.any() for item in vision)


def test_resume_rejects_tail_contract_change_even_with_same_population_size(tmp_path):
    raw, _ = population(tmp_path)
    loader = StatefulWindowLoader(LeRobotPolicySFTDataset(raw))
    state = loader.state_dict()
    state["cursor"] = 9
    restored = StatefulWindowLoader(LeRobotPolicySFTDataset(raw))
    restored.load_state_dict(state)
    assert restored.sampler.cursor == 9
    missing = copy.deepcopy(state)
    missing.pop("tail_windows")
    with pytest.raises(ValueError, match="Tail-window"):
        restored.load_state_dict(missing)
    raw.include_tail_windows = False
    legacy = StatefulWindowLoader(LeRobotPolicySFTDataset(raw))
    with pytest.raises(ValueError, match="Tail-window"):
        legacy.load_state_dict(state)
    raw.include_tail_windows = True
    raw.episodes[0]["end"] -= 1
    with pytest.raises(ValueError, match="Tail-window"):
        StatefulWindowLoader(LeRobotPolicySFTDataset(raw)).load_state_dict(state)


@pytest.mark.parametrize("mode", ["inverse_dynamics", "forward_dynamics", "image2video"])
def test_partial_video_rejects_other_conditioning_modes(mode):
    with pytest.raises(ValueError, match="Partial video"):
        build_sequence_plan_from_mode(mode, 1, 49, action_video_downsample_factor=4, allow_partial_video=True)


def test_current_only_video_has_no_fabricated_future_metrics():
    video = torch.zeros(3, 1, 8, 8, dtype=torch.uint8)
    metrics = image_metrics(video, video)
    assert metrics["current_frame_mae"] == 0
    assert metrics["video_future_frames"] == 0
    assert "video_mae" not in metrics and "video_psnr" not in metrics
    assert "video_temporal_mae" not in metrics


def test_tail_validity_intersects_sparse_channels_and_frame_gaps(tmp_path):
    raw, _ = population(tmp_path)
    ep = raw.episodes[0]
    entry = raw.entries[ep["name"]]
    entry["action_mask"][1] = False
    rows = raw._episode_rows(ep)
    rows["action"][:, 1] = float("nan")
    rows["action_valid_mask"] = entry["action_mask"].expand(64, -1).clone()
    rows["action_valid_mask"][62] = False
    raw._episode_rows = lambda _: rows
    sample = raw.physical_sample(61)
    valid = sample["action_valid_mask"]
    assert valid[1, 0] and valid[3, 0]
    assert not valid[2].any() and not valid[4:].any()
    assert not valid[1:, 1].any()
    assert torch.isfinite(sample["action"]).all()
    assert not sample["action"][~valid].any()
