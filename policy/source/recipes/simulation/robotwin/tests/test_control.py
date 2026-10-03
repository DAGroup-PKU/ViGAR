import numpy as np
import pytest

from recipes.simulation.robotwin.common.control import smooth_joint_commands, validate_joint_smoothing


def test_smoothing_preserves_motion_and_grippers_without_mutating_predictions():
    time = np.arange(48, dtype=np.float32)
    motion = np.tile((0.0001 * time**2 + 0.001 * time)[:, None], (1, 14))
    commands = motion.copy()
    joints = np.r_[0:6, 7:13]
    commands[:, joints] += (0.02 * (-1.0) ** time)[:, None]
    commands[:, [6, 13]] = (time % 3 / 2)[:, None]
    original = commands.copy()
    filtered = smooth_joint_commands(commands, 7)
    assert np.abs(filtered[3:-3, joints] - motion[3:-3, joints]).mean() < 0.01
    np.testing.assert_array_equal(filtered[:, [6, 13]], commands[:, [6, 13]])
    np.testing.assert_array_equal(commands, original)
    np.testing.assert_array_equal(smooth_joint_commands(commands, 1), commands)
    np.testing.assert_allclose(smooth_joint_commands(motion, 7), motion, atol=1e-7)


@pytest.mark.parametrize("window", [0, 4, 49, True, 7.0])
def test_invalid_smoothing_window(window):
    with pytest.raises(ValueError, match="joint_smoothing_window"):
        validate_joint_smoothing(window, "qpos")


def test_joint_filter_cannot_be_applied_to_eef_quaternions():
    with pytest.raises(ValueError, match="qpos"):
        validate_joint_smoothing(7, "ee")
