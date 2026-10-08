"""RoboTwin world/tool poses <-> 49D camera-NWU/Astribot-tool representation.

Only NumPy/SciPy are required, so this bridge also works with other policies.
RoboTwin get_arm_pose returns SAPIEN WORLD poses, not get_*_orig_endpose (base).
"""

import numpy as np
from scipy.spatial.transform import Rotation


ROBOT = "robotwin_aloha_agilex"
CAMERAS = {"head": "head_camera", "left": "left_camera", "right": "right_camera"}
CAMERA_NWU_TO_GL = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float64)
ASTRIBOT_TO_RAW_TOOL = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
ACTION_MASK = np.zeros(49, dtype=bool)
ACTION_MASK[np.r_[0:6, 7, 8:14, 15, 26:42]] = True
STATE_MASK = ACTION_MASK.copy()
STATE_MASK[42:49] = True


def camera_transform(cam2world_gl):
    matrix = np.asarray(cam2world_gl, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite 4x4 head cam2world_gl")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
    ):
        raise ValueError("Camera extrinsics must be a rigid camera-to-world transform")
    return rotation @ CAMERA_NWU_TO_GL, matrix[:3, 3]


def world_pose_to_canonical(pose, cam2world_gl):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > 1e-3:
        raise ValueError("RoboTwin EEF must be xyz + unit wxyz world/tool quaternion")
    rotation, origin = camera_transform(cam2world_gl)
    xyz = rotation.T @ (pose[:3] - origin)
    orientation = Rotation.from_matrix(
        rotation.T @ Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix() @ ASTRIBOT_TO_RAW_TOOL
    ).as_quat()
    return np.r_[xyz, orientation].astype(np.float32)


def canonical_pose_to_world(pose, cam2world_gl):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape[-1] != 7 or not np.isfinite(pose).all():
        raise ValueError("Expected finite canonical xyz + xyzw pose")
    if np.any(np.linalg.norm(pose[..., 3:], axis=-1) < 1e-8):
        raise ValueError("Cannot execute a zero-norm EEF quaternion")
    rotation, origin = camera_transform(cam2world_gl)
    xyz = pose[..., :3] @ rotation.T + origin
    # SciPy explicitly projects nonzero predictions to unit quaternions.
    orientation = Rotation.from_matrix(
        rotation @ Rotation.from_quat(pose[..., 3:]).as_matrix() @ ASTRIBOT_TO_RAW_TOOL.T
    ).as_quat()
    return np.concatenate([xyz, orientation[..., [3, 0, 1, 2]]], axis=-1).astype(np.float32)


def encode_observation(observation):
    """Match the recorded-data converter, including availability and closedness."""
    matrix = np.asarray(observation["observation"]["head_camera"]["cam2world_gl"])
    camera_transform(matrix)
    state = np.zeros(49, dtype=np.float32)
    for side, joint, grip, pose in [("left", 0, 7, 26), ("right", 8, 15, 34)]:
        joints = np.asarray(observation["joint_action"][f"{side}_arm"], dtype=np.float32)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("This bridge supports six-joint aloha-agilex arms")
        state[joint : joint + 6] = joints
        for source, target in [("joint_action", grip), ("endpose", pose + 7)]:
            openness = float(observation[source][f"{side}_gripper"])
            if not np.isfinite(openness) or not -1e-4 <= openness <= 1.0001:
                raise ValueError("RoboTwin gripper openness must be in [0,1]")
            state[target] = (1 - np.clip(openness, 0, 1)) * 100
        state[pose : pose + 7] = world_pose_to_canonical(observation["endpose"][f"{side}_endpose"], matrix)
    state[48] = 1  # Stored common frame is the head camera itself.
    return dict(
        state=state,
        state_valid_mask=STATE_MASK.copy(),
        action_valid_mask=ACTION_MASK.copy(),
        images=extract_images(observation),
        cam2world_gl=matrix.copy(),
        robot_type=ROBOT,
    )


def extract_images(observation):
    result = {}
    for name, key in CAMERAS.items():
        value = np.asarray(observation["observation"][key]["rgb"])
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ValueError(f"{key} must contain decoded HWC uint8 RGB")
        result[name] = np.ascontiguousarray(value)
    return result


def controller_actions(absolute, valid, cam2world_gl, action_type="qpos"):
    """Convert one fixed-anchor chunk to native commands BEFORE executing any row."""
    absolute = np.asarray(absolute, dtype=np.float32)
    valid = np.asarray(valid)
    if (
        absolute.ndim != 2
        or absolute.shape[0] < 1
        or absolute.shape[1] != 49
        or valid.shape != absolute.shape
        or valid.dtype != bool
    ):
        raise ValueError("Expected a nonempty Tx49 chunk and boolean mask")
    if not np.isfinite(absolute[valid]).all():
        raise ValueError("Nonfinite valid command")
    if action_type == "qpos":
        # The policy server has already added the measured joint anchor to
        # predicted joint deltas. Camera-frame conversion applies to EEF state
        # conditioning; these absolute joint angles need no camera rotation/IK.
        dims = np.r_[0:6, 7, 8:14, 15]
        if not valid[:, dims].all():
            raise ValueError("qpos control requires joint supervision; select ee for task-space-only checkpoints")
        result = absolute[:, dims].copy()
        result[:, [6, 13]] = 1 - np.clip(result[:, [6, 13]], 0, 100) / 100
        return result
    if action_type != "ee":
        raise ValueError("action_type must be ee or qpos")
    if not valid[:, 26:42].all():
        raise ValueError("EEF control requires valid left/right poses and EEF grippers")
    left = canonical_pose_to_world(absolute[:, 26:33], cam2world_gl)
    right = canonical_pose_to_world(absolute[:, 34:41], cam2world_gl)
    grip = 1 - np.clip(absolute[:, [33, 41]], 0, 100) / 100
    return np.concatenate([left, grip[:, :1], right, grip[:, 1:]], axis=-1)
