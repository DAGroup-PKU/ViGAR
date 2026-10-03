import unittest
import numpy as np
from sync_subgoal import SyncSubgoalProducer


class Planner:
    def __init__(self):self.calls=[];self.fail=False
    def call(self,r):
        self.calls.append(r)
        if self.fail:raise TimeoutError('Injected failure')
        return dict(goal_image=r['image'].copy())


class SyncTests(unittest.TestCase):
    def setUp(self):self.p=Planner();self.s=SyncSubgoalProducer(self.p)
    def submit(self,n):return self.s.submit(np.full((384,320,3),n,np.uint8),'task',42)
    def test_every_observation_has_its_own_goal(self):
        for n in range(1,6):
            q=self.submit(n);g=self.s.snapshot_or_wait(q,timeout=1)
            self.assertEqual(g.source_observation_version,q.observation_version)
            self.assertTrue((g.image==n).all())
        self.assertEqual(len(self.p.calls),5)
    def test_error_never_returns_old_cache(self):
        self.s.snapshot_or_wait(self.submit(1),timeout=1);self.p.fail=True
        with self.assertRaises(TimeoutError):self.s.snapshot_or_wait(self.submit(2),timeout=1)
    def test_reset_invalidates_old_submission(self):
        q=self.submit(1);self.s.reset()
        with self.assertRaises(RuntimeError):self.s.snapshot_or_wait(q,timeout=1)
    def test_observation_is_copied(self):
        image=np.full((384,320,3),7,np.uint8);q=self.s.submit(image,'task',42);image[:]=19
        self.assertTrue((self.s.snapshot_or_wait(q,timeout=1).image==7).all())


if __name__=='__main__':unittest.main(verbosity=2)
