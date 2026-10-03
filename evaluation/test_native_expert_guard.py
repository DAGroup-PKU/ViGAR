import linecache
import unittest
from types import SimpleNamespace

from native_expert_guard import guarded_rollout


class GuardTests(unittest.TestCase):
    def expert(self, filename='/fake/envs/click_alarmclock.py', expression=None):
        expression = expression or 'self.get_grasp_pose(self.alarm, pre_dis=0.1)[:3]'
        source = 'def play_once(self):\n    return '+expression+'\n'
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        scope = {}
        exec(compile(source, filename, 'exec'), scope)
        return type('Expert', (), dict(play_once=scope['play_once'], plan_success=True,
            alarm=None, get_grasp_pose=lambda *a, **kw: None))()

    def test_exact_invalid_grasp(self):
        row = dict(task='click_alarmclock', seed=400006)
        self.assertIsNone(guarded_rollout(self.expert(), row, lambda e, r: e.play_once()))
        self.assertFalse(row['expert_plan_success'])
        self.assertTrue(row['expert_reported_plan_success'])
        self.assertEqual(row['expert_rejection_rule'], 'click_alarmclock_missing_grasp_v1')

    def test_other_task_is_not_hidden(self):
        with self.assertRaises(TypeError):
            guarded_rollout(self.expert(), dict(task='other'), lambda e, r: e.play_once())

    def test_other_callsite_is_not_hidden(self):
        with self.assertRaises(TypeError):
            guarded_rollout(self.expert('/fake/envs/other.py'), dict(task='click_alarmclock'), lambda e, r: e.play_once())

    def test_other_typeerror_is_not_hidden(self):
        with self.assertRaises(TypeError):
            guarded_rollout(self.expert(expression='None + 3'), dict(task='click_alarmclock'), lambda e, r: e.play_once())

    def test_success_is_unchanged(self):
        value = dict(info='untouched')
        self.assertIs(guarded_rollout(SimpleNamespace(), {}, lambda e, r: value), value)


if __name__ == '__main__':
    unittest.main()
