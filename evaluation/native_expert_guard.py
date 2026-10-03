"""Reject one verified pre-policy expert failure; propagate all other errors."""
import traceback


def guarded_rollout(expert, record, original):
    try:
        return original(expert, record)
    except TypeError as error:
        frames = traceback.extract_tb(error.__traceback__)
        frame = frames[-1]
        expected = (
            record.get('task') == 'click_alarmclock'
            and str(error) == "'NoneType' object is not subscriptable"
            and frame.filename.endswith('/envs/click_alarmclock.py')
            and frame.name == 'play_once'
            and 'self.get_grasp_pose(self.alarm,' in (frame.line or '')
            and ')[:3]' in (frame.line or '')
        )
        if not expected:
            raise
        record.update(
            expert_plan_success=False,
            expert_goal_reached=None,
            expert_reported_plan_success=bool(expert.plan_success),
            expert_exception=f'{type(error).__name__}: {error}',
            expert_traceback=traceback.format_exc(),
            expert_rejection_rule='click_alarmclock_missing_grasp_v1',
        )
        return None


def run_expert_rollout(expert, record):
    from recipes.simulation.robotwin.common.expert import run_expert_rollout as original
    return guarded_rollout(expert, record, original)
