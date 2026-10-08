"""Authoritative monotonic task metrics shared by switching and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


PUT_BOTTLES_DUSTBIN_COUNT_KEY = "__put_bottles_dustbin_count__"
_DUSTBIN_CENTER_XY = np.asarray([-0.45, 0.0], dtype=np.float64)
_DUSTBIN_HALF_EXTENT_XY = np.asarray([0.221, 0.325], dtype=np.float64)
_DUSTBIN_Z_MIN = 0.2
_DUSTBIN_Z_MAX = 0.7


def count_put_bottles_in_dustbin(task_env: Any) -> int:
    """Return the count used by RoboTwin's official task success check.

    Keep the strict inequalities here: this function is the single source of
    truth for both online subgoal switching and atomic-stage evaluation.
    """

    bottles = getattr(task_env, "bottles", None)
    if bottles is None:
        raise ValueError("put_bottles_dustbin environment has no bottles")
    count = 0
    for bottle in bottles:
        position = np.asarray(bottle.get_pose().p, dtype=np.float64)
        inside_xy = np.all(
            np.abs(position[:2] - _DUSTBIN_CENTER_XY)
            < _DUSTBIN_HALF_EXTENT_XY
        )
        if inside_xy and _DUSTBIN_Z_MIN < position[2] < _DUSTBIN_Z_MAX:
            count += 1
    return count
