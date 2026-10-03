import sys
from types import SimpleNamespace

import numpy as np
import pytest

from recipes.simulation.robotwin.common.task_state import initialize_reset_hooks, initialize_task_state


@pytest.mark.parametrize("face,expected", [(1, "left"), (-1, "right"), (0, "right")])
def test_laptop_arm_uses_initial_orientation(monkeypatch, face, expected):
    quaternion = np.array([1, 0, 0, 0])

    def get_face_prod(q, source, target):
        np.testing.assert_array_equal(q, quaternion)
        assert source == target == [1, 0, 0]
        return face

    monkeypatch.setitem(sys.modules, "envs.open_laptop", SimpleNamespace(ArmTag=str, get_face_prod=get_face_prod))
    env = SimpleNamespace(laptop=SimpleNamespace(get_pose=lambda: SimpleNamespace(q=quaternion)))
    initialize_task_state(env, "open_laptop")
    assert env.arm_tag == expected


@pytest.mark.parametrize("task", ["place_object_scale", "put_object_cabinet"])
@pytest.mark.parametrize("x,expected", [(-0.2, "left"), (0, "left"), (0.2, "right")])
def test_object_tasks_use_initial_side_and_height(monkeypatch, task, x, expected):
    monkeypatch.setitem(sys.modules, f"envs.{task}", SimpleNamespace(ArmTag=str))
    pose = SimpleNamespace(p=np.array([x, -0.1, 0.783]))
    env = SimpleNamespace(object=SimpleNamespace(get_pose=lambda: pose))
    initialize_task_state(env, task)
    assert env.arm_tag == expected
    assert getattr(env, "origin_z", None) == (0.783 if task == "put_object_cabinet" else None)


def test_other_tasks_keep_their_setup_state():
    env = SimpleNamespace(stage_success_tag=False)
    initialize_task_state(env, "click_bell")
    assert vars(env) == {"stage_success_tag": False}


@pytest.mark.parametrize("task", ["dump_bin_bigbin", "put_bottles_dustbin"])
def test_offset_table_bounds_do_not_accumulate_across_resets(task):
    original_defaults = [-0.59, 0.59]
    seen = []

    def upstream(cluttered_numbers=10, xlim=original_defaults, ylim=None, zlim=None):
        xlim[0] += 0.3
        xlim[1] += 0.3
        seen.append((cluttered_numbers, xlim.copy(), ylim, zlim))

    env = SimpleNamespace(get_cluttered_table=upstream)
    initialize_reset_hooks(env, task)
    env.get_cluttered_table()
    env.get_cluttered_table()
    assert seen[0] == seen[1] == (10, [-0.29, 0.8899999999999999], [-0.34, 0.34], [0.741])
    assert original_defaults == [-0.59, 0.59]
    explicit = [-0.1, 0.1]
    env.get_cluttered_table(3, xlim=explicit)
    assert explicit == [-0.1, 0.1]
    assert seen[-1][0] == 3


def test_reset_hooks_leave_zero_offset_tasks_unchanged():
    env = SimpleNamespace()
    initialize_reset_hooks(env, "click_bell")
    assert vars(env) == {}
