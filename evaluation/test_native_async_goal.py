import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from native_async_goal import NativeAsyncGoalClient, generated_goal_views


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        canvas = np.zeros((384, 320, 3), np.uint8)
        canvas[:256, :, 0] = 255
        canvas[256:, :160, 1] = 255
        canvas[256:, 160:, 2] = 255
        self.canvas = canvas
        self.seen = []
        producer = SimpleNamespace(reset=lambda: None,
            submit=lambda *args: SimpleNamespace(epoch=2, observation_version=3),
            snapshot_or_wait=lambda *args, **kw: SimpleNamespace(
                epoch=2, image=canvas, version=1, source_observation_version=1,
                generated_at_unix=0, generation_ms=100))
        native = SimpleNamespace(call=lambda path, req: self.seen.append(req) or {'ok': True})
        self.client = NativeAsyncGoalClient(native, producer, lambda views: canvas,
                                            self.tmp.name+'/trace.jsonl',policy_obs_order='legacy_bgr')
        self.req = dict(state=np.zeros(49), images={k: np.zeros((240,320,3),np.uint8)
                         for k in ('head','left','right')}, instruction='Open the laptop.',
                        artifact_context=dict(task='open_laptop',episode_seed=42,
                                              query_index=0,control_step=0))

    def test_rgb_and_camera_order(self):
        views = generated_goal_views(self.canvas)
        for name, channel in [('head',0),('left',1),('right',2)]:
            self.assertEqual(views[name].shape, (240,320,3))
            self.assertTrue((views[name][...,channel] == 255).all())
            self.assertTrue((views[name].sum(-1) == 255).all())

    def test_only_goal_is_added(self):
        self.client.reset('open_laptop',42)
        self.client.call('/infer',self.req)
        self.assertEqual(set(self.seen[0])-set(self.req), {'goal_images'})
        self.assertIs(self.seen[0]['state'],self.req['state'])
        for k in self.req['images']:
            np.testing.assert_array_equal(self.seen[0]['images'][k],self.req['images'][k][...,::-1])

    def test_policy_only_colour_swap(self):
        self.req['images']={k:np.full((240,320,3),(7,29,231),np.uint8) for k in ('head','left','right')}
        self.client.reset('open_laptop',42);self.client.call('/infer',self.req)
        for k in self.req['images']:
            np.testing.assert_array_equal(self.req['images'][k][0,0],[7,29,231])
            np.testing.assert_array_equal(self.seen[0]['images'][k][0,0],[231,29,7])
        self.assertTrue((self.seen[0]['goal_images']['head'][...,0]==255).all())

    def test_reject_legacy_state_and_oracle(self):
        self.client.reset('open_laptop',42)
        for extra in ({'state':np.zeros(14)}, {'goal_images':{}}, {'stage_id':1}):
            with self.assertRaises(ValueError): self.client.call('/infer',dict(self.req,**extra))

    def test_episode_mismatch(self):
        self.client.reset('open_laptop',43)
        with self.assertRaises(ValueError): self.client.call('/infer',self.req)


    def test_strict_sync_rejects_old_observation_goal(self):
        self.client.goal_refresh_mode='sync_current'
        self.client.reset('open_laptop',42)
        with self.assertRaisesRegex(RuntimeError,'does not match current observation'):
            self.client.call('/infer',self.req)


if __name__ == '__main__': unittest.main()
