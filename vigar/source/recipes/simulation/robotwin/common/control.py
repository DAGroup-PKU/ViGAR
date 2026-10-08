"""Optional execution filtering; raw model predictions remain unchanged."""

import numpy as np


def validate_joint_smoothing(window, action_type):
    if type(window) is not int or window < 1 or window > 47 or window % 2 != 1:
        raise ValueError("joint_smoothing_window must be an odd integer in [1,47]")
    if window > 1 and action_type != "qpos":
        raise ValueError("Joint smoothing requires qpos control")


def smooth_joint_commands(commands, window):
    validate_joint_smoothing(window, "qpos")
    commands = np.asarray(commands, dtype=np.float32)
    if commands.shape != (48, 14) or not np.isfinite(commands).all():
        raise ValueError("Expected finite native 48x14 joint commands")
    result = commands.copy()
    if window > 1:
        from scipy.signal import savgol_filter

        joints = np.r_[0:6, 7:13]
        result[:, joints] = savgol_filter(commands[:, joints], window, 2, axis=0)
    return result
