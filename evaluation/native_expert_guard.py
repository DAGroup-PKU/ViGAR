"""Reject seeds whose pre-policy expert rollout raises, as the RoboTwin evaluator does."""
import traceback


def guarded_rollout(expert, record, original):
    try:
        return original(expert, record)
    except Exception as error:
        frame = traceback.extract_tb(error.__traceback__)[-1]
        known = (
            isinstance(error, TypeError)
            and record.get('task') == 'click_alarmclock'
            and str(error) == "'NoneType' object is not subscriptable"
            and frame.filename.endswith('/envs/click_alarmclock.py')
            and frame.name == 'play_once'
            and 'self.get_grasp_pose(self.alarm,' in (frame.line or '')
            and ')[:3]' in (frame.line or '')
        )
        record.update(
            expert_plan_success=False,
            expert_goal_reached=None,
            expert_reported_plan_success=bool(getattr(expert, 'plan_success', False)),
            expert_exception=f'{type(error).__name__}: {error}',
            expert_traceback=traceback.format_exc(),
            expert_rejection_rule='click_alarmclock_missing_grasp_v1' if known else 'expert_exception',
        )
        return None


def run_expert_rollout(expert, record):
    from recipes.simulation.robotwin.common.expert import run_expert_rollout as original
    return guarded_rollout(expert, record, original)
