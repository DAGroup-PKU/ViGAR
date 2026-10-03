"""Independent frame/control checks and online-versus-training input contracts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from recipes.simulation.robotwin.common.geometry import (
    ACTION_MASK,
    STATE_MASK,
    camera_transform,
    canonical_pose_to_world,
    controller_actions,
    encode_observation,
    world_pose_to_canonical,
)
from recipes.simulation.robotwin.common.transport import decode, dumps, loads


os.environ.setdefault("COSMOS_TRAINING", "1")
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def observation():
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.4, 0.7]).as_matrix()
    matrix[:3, 3] = [0.5, -0.2, 1.3]
    return dict(
        observation={
            camera: dict(cam2world_gl=matrix.copy(), rgb=np.full((240, 320, 3), color, dtype=np.uint8))
            for camera, color in [
                ("head_camera", [30, 60, 90]),
                ("left_camera", [2, 80, 200]),
                ("right_camera", [100, 0, 255]),
            ]
        },
        joint_action=dict(
            left_arm=np.arange(6) * 0.1, right_arm=np.arange(6) * -0.1, left_gripper=0.25, right_gripper=0.8
        ),
        endpose=dict(
            left_endpose=np.r_[[0.2, 0.1, 0.8], [1, 0, 0, 0]],
            right_endpose=np.r_[[0.5, -0.3, 0.7], [0, 0, 0, 1]],
            left_gripper=0.25,
            right_gripper=0.8,
        ),
    )


def test_geometry_matches_independent_rigid_transform(observation):
    current = encode_observation(observation)
    for side, lo in [("left", 26), ("right", 34)]:
        pose = observation["endpose"][f"{side}_endpose"]
        camera = current["cam2world_gl"].copy()
        axes = camera[:3, :3].copy()
        camera[:3, :3] = np.column_stack([-axes[:, 2], -axes[:, 0], axes[:, 1]])
        expected_xyz = (np.linalg.inv(camera) @ np.r_[pose[:3], 1])[:3]
        expected_rotation = (
            Rotation.from_matrix(camera[:3, :3]).inv()
            * Rotation.from_quat(pose[[4, 5, 6, 3]])
            * Rotation.from_euler("z", np.pi / 2)
        )
        np.testing.assert_allclose(current["state"][lo : lo + 3], expected_xyz, atol=1e-6)
        actual_rotation = Rotation.from_quat(current["state"][lo + 3 : lo + 7])
        assert (expected_rotation.inv() * actual_rotation).magnitude() < 1e-6
        inverse = canonical_pose_to_world(current["state"][lo : lo + 7], current["cam2world_gl"])
        np.testing.assert_allclose(inverse[:3], pose[:3], atol=1e-6)
        assert abs(np.dot(inverse[3:], pose[3:])) == pytest.approx(1, abs=1e-6)
    np.testing.assert_array_equal(current["state"][42:], [0, 0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(current["state"][[7, 15, 33, 41]], [75, 20, 75, 20])
    assert not current["state"][~STATE_MASK].any()
    np.testing.assert_array_equal(current["images"]["head"][0, 0], [30, 60, 90])  # No RGB/BGR swap


def test_known_camera_axes_and_tool_basis():
    camera = np.eye(4)
    camera[:3, 3] = [1, 2, 3]
    result = world_pose_to_canonical([3, 5, 7, 1, 0, 0, 0], camera)
    np.testing.assert_allclose(result[:3], [-4, -2, 3])  # (-GL.z,-GL.x,+GL.y)
    # Independently written axes: Astribot x=raw y, y=-raw x, z=raw z.
    expected = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    np.testing.assert_allclose(Rotation.from_quat(result[3:]).as_matrix(), expected, atol=1e-6)


@pytest.mark.parametrize("action_type", ["ee", "qpos"])
def test_executable_commands_match_observed_native_pose(observation, action_type):
    current = encode_observation(observation)
    absolute = np.broadcast_to(current["state"], (48, 49)).copy()
    absolute[:, 29:33] *= 2.5
    absolute[:, 37:41] *= -0.7
    absolute[:, [7, 33]] = -20
    absolute[:, [15, 41]] = 150
    valid = np.broadcast_to(ACTION_MASK, absolute.shape).copy()
    commands = controller_actions(absolute, valid, current["cam2world_gl"], action_type)
    if action_type == "qpos":
        np.testing.assert_array_equal(commands[0, :6], current["state"][:6])
        np.testing.assert_array_equal(commands[:, [6, 13]], np.tile([1, 0], (48, 1)))
    else:
        np.testing.assert_allclose(commands[0, :3], observation["endpose"]["left_endpose"][:3], atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(commands[:, 3:7], axis=-1), 1, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(commands[:, 11:15], axis=-1), 1, atol=1e-6)
        np.testing.assert_array_equal(commands[:, [7, 15]], np.tile([1, 0], (48, 1)))
    assert absolute[0, 7] == -20  # Preserve raw diagnostics


def test_invalid_control_cannot_fabricate_missing_joints(observation):
    current = encode_observation(observation)
    chunk = np.tile(current["state"], (48, 1))
    valid = np.tile(ACTION_MASK, (48, 1))
    valid[:, :23] = False
    with pytest.raises(ValueError, match="joint supervision"):
        controller_actions(chunk, valid, current["cam2world_gl"], "qpos")
    chunk[:, 29:33] = 0
    with pytest.raises(ValueError, match="zero-norm"):
        controller_actions(chunk, valid, current["cam2world_gl"], "ee")
    bad = current["cam2world_gl"].copy()
    bad[:3, :3] *= 2
    with pytest.raises(ValueError, match="rigid"):
        camera_transform(bad)


def test_protocol_roundtrip_and_reject_object_arrays(observation):
    current = encode_observation(observation)
    decoded = loads(dumps(current))
    for name in ["state", "state_valid_mask", "action_valid_mask", "cam2world_gl"]:
        np.testing.assert_array_equal(decoded[name], current[name])
    with pytest.raises(ValueError):
        dumps(np.array([object()]))
    with pytest.raises(ValueError):
        decode(dict(__array__="", dtype="O", shape=[0]))


@pytest.fixture
def processor():
    from recipes.simulation.robotwin.goalwam.policy import ObservationProcessor, load_settings

    checkpoint = Path("/path/to/vigar/assets")
    if not checkpoint.is_dir():
        pytest.skip("Local GoalWAM bundle unavailable")
    settings, normalizer, _ = load_settings(checkpoint, ROOT / "recipes/GoalWAM/configs/robotwin.yaml")
    return ObservationProcessor(settings["data"], normalizer)


def test_online_inputs_match_dataset_with_no_future_labels(processor):
    import torch

    from recipes.GoalWAM.data.dataset import LeRobot0824Dataset

    config = processor.data
    dataset = LeRobot0824Dataset(
        {key: str(ROOT / value) for key, value in config["eval_path"].items()},
        {key: str(ROOT / value) for key, value in config["norm_stat_files"].items()},
        norm_type=config["norm_type"],
        img_size=config["img_size"],
        img_size_buckets=config.get("img_size_buckets"),
        goal_image_composition=config.get("goal_image_composition", "multi_view"),
        enable_cameras=config["enable_cameras"],
        supervise_head_eef=config["supervise_head_eef"],
        supervise_arm_head_torso=config["supervise_arm_head_torso"],
        state_arm_eef_coordinate=config["state_arm_eef_coordinate"],
    )
    try:
        physical, item = dataset.physical_sample(0), dataset[0]
        episode = physical["episode"]
        cameras = [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
        images, goal_images = {}, {}
        for key, name in zip(cameras, ["head", "left", "right"], strict=True):
            frames = dataset._video(episode, key, [physical["start"], physical["goal_index"]])
            images[name], goal_images[name] = frames
        request = dict(
            state=physical["anchor_state"].numpy(),
            state_valid_mask=STATE_MASK,
            action_valid_mask=ACTION_MASK,
            images=images,
            goal_images=goal_images,
            instruction=item["ai_caption"],
            robot_type="robotwin_aloha_agilex",
        )
        sample, state, mask = processor.raw_sample(request)
        assert torch.equal(sample["action"][0], item["action"][0])
        assert torch.equal(sample["action_valid_mask"], item["action_valid_mask"])
        assert torch.equal(sample["video"][:, :1], item["video"][:, :1])
        assert torch.equal(sample["goal_frame"], item["goal_frame"])
        assert not sample["action"][1:].any() and not sample["video"][:, 1:].any()
        assert torch.equal(state, physical["anchor_state"])
        assert torch.equal(mask, physical["relative_valid"])
    finally:
        dataset.close()


def test_base_flag_does_not_recover_robot_base_and_stats_reject_it(processor, observation):
    from recipes.GoalWAM.data.dataset import ContractNormalizer
    from recipes.GoalWAM.data.relative_action import transform_state_arm_eef_coordinate

    state = encode_observation(observation)["state"]
    head = transform_state_arm_eef_coordinate(state, "head_camera", processor.layout)
    base = transform_state_arm_eef_coordinate(state, "base", processor.layout)
    np.testing.assert_array_equal(head, base)  # Stored head pose is identity.
    assert not np.allclose(base[26:29], observation["endpose"]["left_endpose"][:3])
    with pytest.raises(ValueError, match="statistics must describe base state"):
        ContractNormalizer(processor.normalizer.path, "bounds_99_clip3", "base")


def test_setup_dry_run_has_no_side_effects(tmp_path):
    workspace = tmp_path / "absent"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "recipes/simulation/robotwin/common/setup.py"),
            "--workspace",
            str(workspace),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["model_packages"] is False
    assert not workspace.exists()


def test_checkpoint_recipe_defaults_and_mismatch(processor, tmp_path, monkeypatch):
    import yaml

    from recipes.simulation.robotwin.goalwam import policy as goalwam_policy

    data = dict(processor.data)
    saved = dict(data=data, train={})
    (tmp_path / "recipe_config.json").write_text(json.dumps(saved))
    monkeypatch.setattr(goalwam_policy, "read_bundle", lambda _: {})
    supplied = dict(data={k: v for k, v in data.items() if k != "resolution"}, train={})
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(supplied))
    settings, _, _ = goalwam_policy.load_settings(tmp_path, recipe)
    assert settings["data"]["resolution"] == "384x320"
    supplied["data"]["supervise_arm_head_torso"] = False
    recipe.write_text(yaml.safe_dump(supplied))
    with pytest.raises(ValueError, match="disagrees.*supervise_arm_head_torso"):
        goalwam_policy.load_settings(tmp_path, recipe)


@pytest.mark.parametrize("replan_steps,step_limit", [(3, 5), (32, 35)])
@pytest.mark.parametrize("sampling", [None, {"num_steps": 20, "guidance": 1.0, "shift": 3.0}])
def test_closed_loop_replans_and_logs_runtime_errors(
    observation, tmp_path, monkeypatch, replan_steps, step_limit, sampling
):
    import copy
    from types import SimpleNamespace

    from recipes.simulation.robotwin.goalwam import rollout as run_goalwam

    events = []

    class FakeEnv:
        def __init__(self):
            self.observation = copy.deepcopy(observation)
            self.step_lim = step_limit
            self.take_action_cnt = 0
            self.eval_success = False
            self.plan_success = True

        def get_obs(self):
            return self.observation

        def _update_render(self):
            pass

        def play_once(self):
            self.observation["observation"]["head_camera"]["rgb"][:] = 241
            return {"info": {}}

        def check_success(self):
            return True

        def close_env(self, **kwargs):
            events.append("closed")

        def set_instruction(self, **kwargs):
            pass

        def take_action(self, command, action_type):
            assert action_type == "qpos" and command.shape == (14,)
            self.take_action_cnt += 1
            self.observation["joint_action"]["left_arm"] = command[:6].copy()
            self.eval_success = self.take_action_cnt == step_limit

    class FakeClient:
        def __init__(self):
            self.requests = []

        def call(self, path, request):
            self.requests.append(copy.deepcopy(request))
            assert set(request) == {
                "state",
                "state_valid_mask",
                "action_valid_mask",
                "images",
                "cam2world_gl",
                "robot_type",
                "instruction",
                "goal_images",
                "generation_seed",
                "artifact_context",
            } | ({"sampling"} if sampling is not None else set())
            assert request.get("sampling") == sampling
            assert request["goal_images"]["head"][0, 0, 0] == 241
            chunk = np.tile(request["state"], (48, 1))
            chunk[:, 0] += np.arange(1, 49) * 0.01
            return dict(absolute=chunk, valid=np.tile(ACTION_MASK, (48, 1)))

    def make_env(task, args, seed):
        if seed == 100000:
            raise run_goalwam.UnstableScene("unstable initial object")
        return FakeEnv()

    monkeypatch.setattr(run_goalwam, "make_environment", make_env)
    monkeypatch.setattr(run_goalwam, "task_arguments", lambda *a: {})
    monkeypatch.setitem(
        sys.modules,
        "generate_episode_instructions",
        SimpleNamespace(generate_episode_descriptions=lambda *a: [{"unseen": ["test instruction"]}]),
    )
    monkeypatch.chdir(tmp_path)
    config = dict(
        robotwin_root=str(tmp_path),
        output=str(tmp_path / "result"),
        tasks=["fake"],
        episodes=1,
        task_config="clean",
        start_seed=100000,
        max_seed_attempts=2,
        instruction_type="unseen",
        save_video=False,
        max_policy_steps=None,
        generation_seed=9000,
        action_type="qpos",
        replan_steps=replan_steps,
    )
    if sampling is not None:
        config["sampling"] = sampling
    client = FakeClient()
    run_goalwam.evaluate(config, client)
    assert len(client.requests) == 2 and len(events) == 2
    for query_index, request in enumerate(client.requests):
        assert request["artifact_context"] == dict(
            task="fake", episode_seed=100001, query_index=query_index, control_step=query_index * replan_steps
        )
    assert client.requests[1]["state"][0] == pytest.approx(replan_steps * 0.01)
    commands = np.load(tmp_path / "result/fake/seed_000100001/executed_commands.npy")
    np.testing.assert_allclose(commands[:, 0], np.arange(1, step_limit + 1) * 0.01, atol=1e-6)
    final = np.load(tmp_path / "result/fake/seed_000100001/final_state.npz")
    assert final["state"][0] == pytest.approx(step_limit * 0.01)
    summary = json.loads((tmp_path / "result/summary.json").read_text())
    assert summary["complete"] and summary["successes"] == 1
    assert summary["records"][0]["status"] == "expert_rejected"
    assert summary["records"][-1]["control_steps"] == step_limit
    # A matched evaluation reuses the exact expert goal/instruction and creates
    # only the policy environment, rather than rerunning the expert planner.
    cached = dict(config, output=str(tmp_path / "cached"), oracle_reference_roots=[str(tmp_path / "result")])
    cached_client = FakeClient()
    run_goalwam.evaluate(cached, cached_client)
    assert len(events) == 3 and len(cached_client.requests) == 2
    for original, repeated in zip(client.requests, cached_client.requests, strict=True):
        assert original["instruction"] == repeated["instruction"]
        for camera in original["goal_images"]:
            np.testing.assert_array_equal(original["goal_images"][camera], repeated["goal_images"][camera])
    cached_summary = json.loads((tmp_path / "cached/summary.json").read_text())
    assert cached_summary["complete"] and cached_summary["records"][0]["oracle_reference"]
    # A changed scene image must fail before sending any policy request.
    import imageio.v2 as imageio

    initial_path = tmp_path / "result/fake/seed_000100001/initial_head.png"
    changed = imageio.imread(initial_path)
    changed[0, 0, 0] ^= 1
    imageio.imwrite(initial_path, changed)
    mismatch_client = FakeClient()
    with pytest.raises(AssertionError):
        run_goalwam.evaluate(dict(cached, output=str(tmp_path / "mismatch")), mismatch_client)
    assert not mismatch_client.requests and len(events) == 4
    # An inference exception remains an error with no fabricated policy failure.
    config["output"] = str(tmp_path / "failed")

    def fail(*args):
        raise ConnectionError("policy disconnected")

    client.call = fail
    with pytest.raises(ConnectionError):
        run_goalwam.evaluate(config, client)
    summary = json.loads((tmp_path / "failed/summary.json").read_text())
    assert not summary["complete"] and summary["evaluated"] == 0 and summary["errors"]
    assert len(events) == 6
    # Fresh expert/policy resets must match too, including scene RGB rather
    # than only robot state. Reject clutter drift before the first inference.
    resets = 0

    def drifted_reset(task, args, seed):
        nonlocal resets
        env = make_env(task, args, seed)
        resets += 1
        if resets == 2:
            env.observation["observation"]["head_camera"]["rgb"][0, 0, 0] ^= 1
        return env

    monkeypatch.setattr(run_goalwam, "make_environment", drifted_reset)
    drift_client = FakeClient()
    with pytest.raises(AssertionError):
        run_goalwam.evaluate(dict(config, output=str(tmp_path / "fresh_drift")), drift_client)
    assert not drift_client.requests and len(events) == 8
