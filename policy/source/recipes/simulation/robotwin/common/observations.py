"""Native camera artifacts and expert/policy reset checks."""

import numpy as np
from PIL import Image


CAMERAS = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


def save_observation(directory, prefix, observation):
    for name, camera in CAMERAS.items():
        Image.fromarray(observation["observation"][camera]["rgb"]).save(directory / f"{prefix}_{name}.png")
    np.savez_compressed(
        directory / f"{prefix}_state.npz",
        joint_state=observation["joint_action"]["vector"],
        cam2world_gl=observation["observation"]["head_camera"]["cam2world_gl"],
    )


def assert_same_scene(expert, policy):
    np.testing.assert_allclose(expert["joint_action"]["vector"], policy["joint_action"]["vector"], atol=1e-5, rtol=0)
    for camera in CAMERAS.values():
        np.testing.assert_array_equal(expert["observation"][camera]["rgb"], policy["observation"][camera]["rgb"])
    np.testing.assert_allclose(
        expert["observation"]["head_camera"]["cam2world_gl"],
        policy["observation"]["head_camera"]["cam2world_gl"],
        atol=1e-6,
        rtol=0,
    )
