"""Geometry, sparse validity and normalization checks for the 0824 adapter."""

import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from recipes.GoalWAM.data.dataset import (
    ACTION_LAYOUT,
    ContractNormalizer,
    validate_mask,
)
from recipes.GoalWAM.data.relative_action import (
    check_anchor_head_quaternion,
    propagate_relative_action_validity,
    to_global_action,
    to_relative_action,
    transform_state_arm_eef_coordinate,
)


def poses():
    rng = np.random.default_rng(5)
    state = rng.normal(size=49).astype(np.float32)
    actions = rng.normal(size=(48, 49)).astype(np.float32)
    for group in (slice(29, 33), slice(37, 41), slice(45, 49)):
        state[group] = Rotation.random(random_state=rng).as_quat()
        actions[:, group] = Rotation.random(48, random_state=rng).as_quat()
    return torch.tensor(state), torch.tensor(actions)


def test_nonidentity_head_geometry_and_inverse():
    state, actions = poses()
    rel = to_relative_action(state, actions, ACTION_LAYOUT)
    restored = to_global_action(state, rel, ACTION_LAYOUT)
    rs = Rotation.from_quat(state[45:49].numpy())
    head_state = transform_state_arm_eef_coordinate(state, "head_camera", ACTION_LAYOUT)
    for group, quat in ((slice(26, 29), slice(29, 33)), (slice(34, 37), slice(37, 41))):
        expected = rs.inv().apply((actions[:, group] - state[group]).numpy())
        np.testing.assert_allclose(rel[:, group], expected, atol=1e-6)
        np.testing.assert_allclose(
            head_state[group],
            rs.inv().apply((state[group] - state[42:45]).numpy()),
            atol=1e-6,
        )
        actual_rot = Rotation.from_quat(rel[:, quat].numpy())
        expected_rot = Rotation.from_quat(state[quat].numpy()).inv() * Rotation.from_quat(actions[:, quat].numpy())
        assert np.max((actual_rot.inv() * expected_rot).magnitude()) < 1e-6
        assert torch.all(rel[:, quat.stop - 1] >= 0)
    scalar = list(range(29)) + [33, 34, 35, 36, 41, 42, 43, 44]
    torch.testing.assert_close(restored[:, scalar], actions[:, scalar], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(rel[:, [7, 15, 33, 41]], actions[:, [7, 15, 33, 41]])


def test_anchor_validity_is_atomic_but_grippers_are_absolute():
    sm = torch.ones(49, dtype=torch.bool)
    am = torch.ones(48, 49, dtype=torch.bool)
    sm[26] = False
    sm[7] = False
    am[3, 37] = False
    valid = propagate_relative_action_validity(am, sm, ACTION_LAYOUT)
    assert not valid[:, 26:29].any()
    assert valid[:, 7].all()
    assert not valid[3, 37:41].any()
    sm[45:49] = False
    valid = propagate_relative_action_validity(am, sm, ACTION_LAYOUT)
    assert not valid[:, 34:37].any()
    assert valid[:, 33].all()
    with pytest.raises(ValueError, match="atomic"):
        validate_mask(sm, context="test")


def test_declared_valid_zero_head_quaternion_is_rejected():
    state, _ = poses()
    state[45:49] = 0
    with pytest.raises(ValueError):
        check_anchor_head_quaternion(state, torch.ones(49, dtype=torch.bool), ACTION_LAYOUT)


def test_separate_scales_constant_identity_and_masked_nan(tmp_path):
    stats = {}
    for key, mean, std in (("observation.state", 10, 2), ("action", 1, 4)):
        stats[key] = {"mean": [mean] * 49, "std": [std] * 49}
        stats[key]["std"][0] = 0
    path = tmp_path / "stats.json"
    path.write_text(
        json.dumps(
            {
                "norm_stats": stats,
                "metadata": {
                    "action_horizon": 50,
                    "state_arm_eef_coordinate": "head_camera",
                },
            }
        )
    )
    before = path.read_bytes()
    n = ContractNormalizer(path)
    x = torch.full((48, 49), 5.0)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, 2] = False
    x[:, 2] = float("nan")
    action = n.normalize(x, "action", mask)
    state = n.normalize(x, "observation.state", mask)
    assert torch.isfinite(action).all() and not action[:, 2].any()
    assert torch.all(action[:, 0] == 5)
    assert not torch.allclose(action[:, 1], state[:, 1])
    torch.testing.assert_close(n.denormalize_action(action)[mask], x[mask])
    assert n.metadata["action_horizon"] == 50 and path.read_bytes() == before


def test_distributed_sampler_resume_ignores_prefetch():
    from itertools import islice

    from recipes.GoalWAM.data.data_loader import WindowSampler

    reference = WindowSampler(17, 1, 4, seed=8)
    expected = list(islice(iter(reference), 12))
    # A worker may request 8 items, although the trainer has consumed only 3.
    iterator = iter(reference)
    assert list(islice(iterator, 8)) == expected[:8]
    reference.cursor = 3
    restored = WindowSampler(17, 1, 4, seed=8)
    restored.load_state_dict(reference.state_dict())
    assert list(islice(iter(restored), 9)) == expected[3:]
    with pytest.raises(ValueError, match="topology"):
        WindowSampler(17, 0, 4, seed=8).load_state_dict(reference.state_dict())


def test_collated_masks_match_native_dense_actions():
    from cosmos_framework.data.vfm.action.validity import batch_action_validity

    from recipes.GoalWAM.data.data_loader import collate_samples

    samples = [
        dict(
            action=torch.randn(49, 64),
            action_valid_mask=torch.ones(49, 64, dtype=torch.bool),
        )
        for _ in range(2)
    ]
    samples[1]["action_valid_mask"][:, 49:] = False
    batch = collate_samples(samples)
    masks = batch_action_validity(batch, [item[0] for item in batch["action"]])
    for sample, mask in zip(samples, masks, strict=True):
        torch.testing.assert_close(mask, sample["action_valid_mask"])


def test_fixed_selection_respects_sampling_weights():
    from types import SimpleNamespace

    from recipes.GoalWAM.data.data_loader import fixed_indices

    raw = SimpleNamespace(
        episodes=[
            dict(name="a", weighted=100, count=10),
            dict(name="b", weighted=4, count=20),
        ]
    )
    indices = fixed_indices(raw, 6, seed=8)
    assert indices == [0, 102, 99, 100, 50, 103]


def test_inference_removes_future_labels_and_retains_goal():
    from types import SimpleNamespace

    from recipes.GoalWAM.trainer.evaluator import inference_sample

    goal = torch.rand(3, 1, 8, 8)
    rollout = torch.rand(3, 13, 8, 8)
    actions = torch.rand(49, 64)
    plan = SimpleNamespace(condition_frame_indexes_action=[0], condition_frame_indexes_vision=[0])
    sample = dict(action=actions, video=[goal, rollout], sequence_plan=plan)
    result = inference_sample(sample)
    assert torch.equal(result["action"][0], actions[0])
    assert torch.equal(result["video"][-1][:, 0], rollout[:, 0])
    assert result["video"][0] is goal
    assert not result["action"][1:].any() and not result["video"][-1][:, 1:].any()
    assert actions[1:].any() and rollout[:, 1:].any()
    plan.condition_frame_indexes_action = [0, 1]
    with pytest.raises(ValueError, match="current"):
        inference_sample(sample)


def test_master_checkpoint_mapping_preserves_unrounded_values():
    from types import SimpleNamespace

    from recipes.GoalWAM.trainer.evaluator import optimizer_master_state

    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    model.weight.data.fill_(1)
    master = torch.full((1, 2), 1.0001, dtype=torch.float32)
    opt = SimpleNamespace(
        master_weights=True,
        param_groups=[{"params": [model.weight]}],
        param_groups_master=[{"params": [master]}],
    )
    source = optimizer_master_state(model, SimpleNamespace(optimizers=[opt]))
    assert source["weight"] is master
    assert not torch.equal(source["weight"], model.weight.float())
    opt.param_groups_master = None
    restored = optimizer_master_state(model, SimpleNamespace(optimizers=[opt]), initialize=True)
    restored["weight"].copy_(source["weight"])
    assert torch.equal(opt.param_groups_master[0]["params"][0], master)
