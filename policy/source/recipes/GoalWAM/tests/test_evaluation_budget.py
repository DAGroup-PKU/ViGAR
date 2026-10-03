"""Independent loss/generation work, metric denominators and global media limits."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recipes.GoalWAM.data.action_layout import resolve_action_layout
from recipes.GoalWAM.data.data_loader import EvaluationWindows, evaluation_loader
from recipes.GoalWAM.data.dataset import ACTION_LAYOUT
from recipes.GoalWAM.trainer import evaluator, visualization
from recipes.GoalWAM.trainer.evaluation_budget import resolve_evaluation_budget


class TestWindows:
    """Small picklable dataset exercising the actual evaluation loader."""

    __test__ = False

    def __init__(self, size=37):
        self.size = size
        self._dataset = self

    def __len__(self):
        return self.size

    def physical_sample(self, index):
        return dict(robot_type="robot", start=index)

    def __getitem__(self, index):
        return dict(video=torch.tensor(index), ai_caption=str(index))

    def prepare_sample(self, decoded, robot):
        return decoded


def test_evaluation_shuffle_is_disjoint_reproducible_and_prefix_stable():
    dataset = TestWindows()
    ranks = [EvaluationWindows(dataset, 24, rank, 3, 42) for rank in range(3)]
    rows = sorted([item for windows in ranks for item in windows], key=lambda item: item["sample_id"])
    indices = [row["index"] for row in rows]
    assert len(indices) == len(set(indices)) == 24
    assert indices != list(range(24))
    assert any(index not in (0, len(dataset) // 2, len(dataset) - 1) for index in indices)
    for rank, windows in enumerate(ranks):
        assert len(windows) == 8
        assert [item["index"] for item in windows] == indices[rank::3]
        shorter = EvaluationWindows(dataset, 12, rank, 3, 42)
        assert [item["index"] for item in shorter] == [item["index"] for item in windows][:4]
    assert EvaluationWindows(dataset, 24, 0, 3, 42).indices == indices
    assert EvaluationWindows(dataset, 24, 0, 3, 43).indices != indices


@pytest.mark.parametrize("workers", [0, 2])
def test_evaluation_shuffle_ignores_worker_count_and_global_rng(workers):
    np.random.seed(99)
    expected_rng = np.random.get_state()
    windows = EvaluationWindows(TestWindows(), 12, 1, 3, 42)
    loader = evaluation_loader(TestWindows(), 12, 1, 3, seed=42, num_workers=workers)
    assert [(item["sample_id"], item["index"]) for item in loader] == [
        (item["sample_id"], item["index"]) for item in windows
    ]
    actual_rng = np.random.get_state()
    np.testing.assert_array_equal(actual_rng[1], expected_rng[1])
    assert actual_rng[2:] == expected_rng[2:]


def test_evaluation_exhaustion_keeps_ranks_equal_without_duplicates():
    ranks = [EvaluationWindows(TestWindows(11), 100, rank, 4, 42) for rank in range(4)]
    assert [len(windows) for windows in ranks] == [2] * 4
    indices = [item["index"] for windows in ranks for item in windows]
    assert len(indices) == len(set(indices)) == 8
    assert len(EvaluationWindows(TestWindows(0), 0, 0, 4, 42)) == 0
    with pytest.raises(ValueError, match="at least one distinct window per rank"):
        EvaluationWindows(TestWindows(3), 4, 0, 4, 42)


def test_legacy_and_explicit_budgets():
    old = resolve_evaluation_budget(8, count=1024, visual_count=32)
    assert (old.loss_per_rank, old.generation_per_rank, old.visual_max) == (128, 128, 32)
    new = resolve_evaluation_budget(
        8, count=3, eval_loss_per_rank=512, generation_per_rank=128, generation_wandb_max=128
    )
    assert (new.windows_per_rank, new.generation_per_rank, new.visual_max) == (512, 128, 128)
    partial = resolve_evaluation_budget(8, count=32, eval_loss_per_rank=0, generation_wandb_max=100)
    assert (partial.loss_per_rank, partial.generation_per_rank, partial.visual_max) == (0, 4, 32)
    with pytest.raises(ValueError, match="divisible"):
        resolve_evaluation_budget(8, count=3)


@pytest.mark.parametrize("field", ["eval_loss_per_rank", "generation_per_rank", "generation_wandb_max"])
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_budget_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        resolve_evaluation_budget(1, **{field: value})


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_calls = self.generation_calls = self.decode_calls = 0
        self.tensor_kwargs = {}
        self.accum_video_sample_counter = 7
        self.latent = torch.zeros(1, 2, 4, 2, 2)

    def ema_scope(self, **kwargs):
        return nullcontext()

    def training_step(self, batch, iteration):
        self.loss_calls += 1
        self.accum_video_sample_counter += 1
        torch.rand(3)
        return {
            "flow_matching_loss_action": torch.tensor(float(self.loss_calls)),
            "flow_matching_loss_vision": torch.tensor(2.0),
            "x0": [self.latent],
        }, torch.tensor(10.0 * (self.loss_calls + 2))

    def denoise(self):
        pass

    def generate_samples_from_batch(self, batch, **kwargs):
        self.generation_calls += 1
        assert not batch["action"][1:].any(), "Generation received future action targets"
        assert not batch["video"][-1][:, 1:].any(), "Generation received future frames"
        action = torch.zeros(49, 49)
        action[:, [32, 40, 48]] = 1
        return {"action": [action], "vision": [self.latent]}

    def decode(self, latent):
        self.decode_calls += 1
        return torch.zeros(3, 13, 2, 2)


@pytest.mark.parametrize("loss_count,gen_count,visual_max", [(3, 1, 1), (1, 3, 2), (0, 2, 0), (2, 0, 5), (0, 0, 0)])
@pytest.mark.parametrize("rank", [0, 1])
def test_evaluator_independent_work_and_denominators(tmp_path, monkeypatch, loss_count, gen_count, visual_max, rank):
    from cosmos_framework.utils import misc

    world = 2
    monkeypatch.setattr(evaluator.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(evaluator.dist, "get_world_size", lambda: world)
    monkeypatch.setattr(evaluator.dist, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: None)
    monkeypatch.setattr(
        evaluator.dist, "all_gather_object", lambda gathered, rows: gathered.__setitem__(slice(None), [rows] * world)
    )
    monkeypatch.setattr(misc, "to", lambda value, **kwargs: value)
    monkeypatch.setattr(evaluator, "collate_samples", lambda samples: samples[0])
    visuals = []
    monkeypatch.setattr(
        visualization,
        "save_sample_visuals",
        lambda output, sample_id, **kwargs: visuals.append(sample_id) or {"image": "fixture"},
    )
    monkeypatch.setattr(visualization, "write_gallery", lambda *args: None)
    anchor = torch.zeros(49)
    anchor[[32, 40, 48]] = 1
    target = anchor.repeat(48, 1)
    raw = SimpleNamespace(
        layout=resolve_action_layout(ACTION_LAYOUT),
        height=2,
        width=2,
        normalizers={"robot": SimpleNamespace(denormalize_action=lambda value: value)},
        entries={"task": {"fps": 30, "rate": 1}},
        video_stride=4,
    )
    closed = []

    class Iterator:
        def __init__(self, count):
            self.items = iter(range(rank, count, world))

        def __iter__(self):
            return self

        def __next__(self):
            sample_id = next(self.items)
            return dict(
                sample_id=sample_id,
                index=sample_id,
                physical=dict(
                    robot_type="robot",
                    dataset="task",
                    episode_index=0,
                    start=sample_id,
                    relative_valid=torch.ones(48, 49, dtype=torch.bool),
                    relative_actions=target,
                    absolute_actions=target,
                    anchor_state=anchor,
                ),
                sample=dict(
                    action=torch.ones(49, 64),
                    video=[torch.ones(3, 1, 2, 2), torch.ones(3, 13, 2, 2)],
                    sequence_plan=SimpleNamespace(
                        condition_frame_indexes_action=[0], condition_frame_indexes_vision=[0]
                    ),
                ),
                target_video=torch.zeros(3, 13, 2, 2, dtype=torch.uint8),
                pixel_mask=None,
                camera_boxes={},
                caption="task",
            )

        def _shutdown_workers(self):
            closed.append(True)

    monkeypatch.setattr(evaluator, "evaluation_loader", lambda dataset, count, *args, **kwargs: Iterator(count))
    model = TinyModel()
    rng = torch.get_rng_state().clone()
    summary = evaluator.evaluate(
        model,
        SimpleNamespace(_dataset=raw),
        tmp_path,
        10,
        record_artifacts=False,
        eval_loss_per_rank=loss_count,
        generation_per_rank=gen_count,
        generation_wandb_max=visual_max,
    )
    assert model.loss_calls == loss_count
    assert model.generation_calls == model.decode_calls == gen_count
    assert model.training and model.accum_video_sample_counter == 7
    assert torch.equal(rng, torch.get_rng_state())
    assert closed == [True]
    assert visuals == [i for i in range(rank, gen_count * world, world) if i < visual_max]
    assert len(list(tmp_path.rglob("*_generation.pt"))) == gen_count
    assert not list(tmp_path.rglob("*_forward.pt"))
    assert summary["counts"].get("loss", 0) == loss_count * world
    assert summary["counts"].get("eef_pose_ade", 0) == gen_count * world
    assert summary["counts"].get("eef_rot_ade", 0) == gen_count * world
    assert summary["counts"].get("current_latent_max_abs", 0) == min(loss_count, gen_count) * world
    if loss_count:
        assert summary["mean"]["loss"] == 10 * ((loss_count + 1) / 2 + 2)
    if gen_count:
        assert summary["mean"]["eef_pose_ade"] == summary["mean"]["eef_rot_ade"] == 0
        assert not any(key.startswith(("left_", "right_")) for key in summary["mean"])
        assert "joint_ade" in summary["mean"] and "joint_mae_rad" not in summary["mean"]
        assert "gripper_mae_closedness" in summary["mean"]
