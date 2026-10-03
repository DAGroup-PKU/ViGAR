"""Keep expected expert planning failures separate from policy/runtime errors."""

import traceback


def run_expert_rollout(expert, record):
    """Run the oracle expert; reject only recognized failed-planning exceptions."""
    try:
        info = expert.play_once()
    except (AssertionError, IndexError) as error:
        plan_success = bool(expert.plan_success)
        missing_grasp = isinstance(error, AssertionError) and str(error) == (
            "target_pose cannot be None for move action."
        )
        # RoboTwin grasp_actor returns (None, []) after planning fails. Some
        # task scripts still index that empty action list in play_once.
        failed_plan_index = isinstance(error, IndexError) and not plan_success
        if not (missing_grasp or failed_plan_index):
            raise
        record.update(
            expert_plan_success=False,
            expert_goal_reached=None,
            expert_reported_plan_success=plan_success,
            expert_exception=f"{type(error).__name__}: {error}",
            expert_traceback=traceback.format_exc(),
        )
        return None
    record["expert_plan_success"] = bool(expert.plan_success)
    record["expert_goal_reached"] = bool(expert.check_success()) if expert.plan_success else None
    return info


def expected_expert_failure(error, plan_success=True):
    # A grasp search can return None, which upstream turns into this assertion.
    # The official evaluator rejects expert exceptions before policy evaluation.
    return (isinstance(error, AssertionError) and str(error) == "target_pose cannot be None for move action.") or (
        isinstance(error, IndexError) and plan_success is False
    )
