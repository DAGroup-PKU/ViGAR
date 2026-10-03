"""Goal-source adapter for native49D serving; not a complete eval launcher.

The caller must reset once per episode and supply only measured observations.
Reuse AsyncSubgoalProducer for latest-only/epoch behavior. Keep simulator state
conversion, policy normalization, and action reconstruction in their native owners.
"""
import json
import time
from pathlib import Path

import numpy as np


def generated_goal_views(canvas):
    """Match paired_dataset.goal_views exactly, without an intermediate PNG."""
    import torch
    import torch.nn.functional as F

    if canvas.shape != (384, 320, 3) or canvas.dtype != np.uint8:
        raise ValueError('Expected canonical RGB uint8 I2I canvas 384x320')
    t = torch.from_numpy(np.array(canvas, copy=True)).permute(2, 0, 1)[None]
    tiles = {'head': t[:, :, :256, :], 'left': t[:, :, 256:, :160],
             'right': t[:, :, 256:, 160:]}
    # Native ObservationProcessor recomposes these views into the policy's goal canvas.
    return {k: F.interpolate(v.float(), size=(240, 320), mode='bilinear',
                             align_corners=False).round().clamp(0, 255).byte()[0]
               .permute(1, 2, 0).numpy() for k, v in tiles.items()}


class NativeAsyncGoalClient:
    def __init__(self, native_client, producer, compose_i2i_input, trace_path, *, policy_obs_order, goal_refresh_mode='async_latest'):
        if policy_obs_order not in ('rgb','legacy_bgr'):
            raise ValueError('Explicit audited policy observation channel order required')
        self.policy_obs_order=policy_obs_order
        if goal_refresh_mode not in ('async_latest','sync_current'):raise ValueError('Unknown goal refresh mode')
        self.goal_refresh_mode=goal_refresh_mode
        self.client = native_client
        self.producer = producer
        self.compose = compose_i2i_input
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.episode = None

    def reset(self, task, seed):
        self.producer.reset()
        self.episode = (str(task), int(seed))

    def call(self, path, request=None):
        if path != '/infer':
            return self.client.call(path, request)
        if self.episode is None:
            raise RuntimeError('Reset required before the first episode')
        allowed = {'state', 'state_valid_mask', 'action_valid_mask', 'images',
                   'cam2world_gl', 'instruction', 'generation_seed', 'robot_type',
                   'artifact_context', 'sampling'}
        if set(request) - allowed:
            raise ValueError('Unexpected conditioning fields; no expert goals or stage IDs')
        state = np.asarray(request['state'])
        if state.shape != (49,) or not np.isfinite(state).all():
            raise ValueError('Expected measured native49D state, not legacy14D qpos')
        if set(request['images']) != {'head', 'left', 'right'}:
            raise ValueError('All three measured observation views are required')
        context = request.get('artifact_context', {})
        if set(context) - {'task', 'episode_seed', 'query_index', 'control_step'}:
            raise ValueError('Artifact context must not carry expert or stage provenance')
        if (context.get('task'), context.get('episode_seed')) != self.episode:
            raise ValueError('Episode context mismatch')
        started = time.monotonic()
        submission = self.producer.submit(self.compose(request['images']),
                                          str(request['instruction']), self.episode[1])
        snapshot = self.producer.snapshot_or_wait(submission, timeout=600)
        waited = time.monotonic() - started
        if snapshot.epoch != submission.epoch:
            raise RuntimeError('Stale goal from a previous episode')
        if self.goal_refresh_mode=='sync_current' and snapshot.source_observation_version!=submission.observation_version:
            raise RuntimeError('Sync goal does not match current observation')
        policy_images=({k:np.ascontiguousarray(v[...,::-1]) for k,v in request['images'].items()}
                       if self.policy_obs_order=='legacy_bgr' else request['images'])
        forwarded = dict(request, images=policy_images, goal_images=generated_goal_views(snapshot.image))
        result = self.client.call('/infer', forwarded)
        trace = dict(time=time.time(), **context, goal_version=snapshot.version, goal_refresh_mode=self.goal_refresh_mode,
                     source_observation_version=snapshot.source_observation_version,
                     current_observation_version=submission.observation_version,
                     goal_age_seconds=time.time()-snapshot.generated_at_unix,
                     goal_wait_seconds=waited, i2i_ms=snapshot.generation_ms,
                     action_rpc_seconds=time.monotonic()-started-waited,
                     expert_goals=False, oracle_stage_switching=False, planner_input_order='rgb',
                     policy_obs_order=self.policy_obs_order, policy_goal_order='rgb')
        with self.trace_path.open('a') as f:
            f.write(json.dumps(trace)+'\n')
        return result
