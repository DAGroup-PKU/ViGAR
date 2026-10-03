from types import SimpleNamespace

import pytest

from recipes.simulation.robotwin.common.expert import run_expert_rollout


@pytest.mark.parametrize("plan,goal", [(True, True), (True, False), (False, False)])
def test_normal_expert_outcomes(plan, goal):
    info = {"info": {"object": "bottle"}}

    def check_success():
        assert plan, "A failed plan must not query the terminal goal"
        return goal

    expert = SimpleNamespace(plan_success=plan, play_once=lambda: info, check_success=check_success)
    record = {}
    assert run_expert_rollout(expert, record) is info
    assert record == dict(expert_plan_success=plan, expert_goal_reached=goal if plan else None)


@pytest.mark.parametrize(
    "error,plan,reject",
    [
        (AssertionError("target_pose cannot be None for move action."), True, True),
        (IndexError("list index out of range"), False, True),
        (IndexError("list index out of range"), True, False),
        (AssertionError("unexpected invariant"), False, False),
        (RuntimeError("CUDA out of memory"), False, False),
    ],
)
def test_only_recognized_expert_failures_are_rejected(error, plan, reject):
    def play_once():
        raise error

    expert = SimpleNamespace(plan_success=plan, play_once=play_once)
    record = {}
    if not reject:
        with pytest.raises(type(error)) as raised:
            run_expert_rollout(expert, record)
        assert raised.value is error
        assert not record
    else:
        assert run_expert_rollout(expert, record) is None
        assert record["expert_plan_success"] is False
        assert record["expert_goal_reached"] is None
        assert record["expert_reported_plan_success"] is plan
        assert str(error) in record["expert_exception"]
        assert "play_once" in record["expert_traceback"]


def test_goal_check_errors_are_not_expert_seed_rejections():
    def check_success():
        raise IndexError("bad goal state")

    expert = SimpleNamespace(plan_success=True, play_once=lambda: {}, check_success=check_success)
    with pytest.raises(IndexError, match="bad goal state"):
        run_expert_rollout(expert, {})
