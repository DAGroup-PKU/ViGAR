"""Strict current-observation goal generation; no asynchronous cached fallback."""
import threading
import time
import numpy as np
from async_subgoal import _SubgoalRequest, _SubgoalSubmission, _SubgoalSnapshot, TARGET_HW


class SyncSubgoalProducer:
    def __init__(self, i2i_client):
        self.client=i2i_client
        self.lock=threading.RLock()
        self.epoch=0;self.observation_version=0;self.version=0
        self.pending=None;self.current=None;self.prompt=None;self.stopped=False

    def submit(self, observation, prompt, seed):
        with self.lock:
            if self.stopped:raise RuntimeError('Producer stopped')
            changed=self.prompt is not None and self.prompt!=prompt
            if changed:self.epoch+=1;self.current=None
            self.prompt=prompt;self.observation_version+=1
            self.pending=_SubgoalRequest(self.epoch,self.observation_version,
                np.ascontiguousarray(observation).copy(),prompt,seed)
            return _SubgoalSubmission(self.epoch,self.observation_version,changed)

    def snapshot_or_wait(self, submission, *, timeout):
        with self.lock:
            if self.stopped:raise RuntimeError('Producer stopped')
            r=self.pending
            if r is None or (r.epoch,r.observation_version)!=(submission.epoch,submission.observation_version):
                raise RuntimeError('Sync submission no longer current')
            # The RPC client's timeout is configured by the caller. Fail rather
            # than return an older image on any RPC/validation error.
            started=time.monotonic()
            response=self.client.call(dict(cmd='infer',image=r.observation,prompt=r.prompt,seed=r.seed))
            elapsed=time.monotonic()-started
            if timeout is not None and elapsed>timeout:raise TimeoutError('Sync generation deadline exceeded')
            image=np.asarray(response['goal_image'],dtype=np.uint8)
            if image.shape!=(*TARGET_HW,3):raise ValueError('Invalid generated goal shape')
            image=np.ascontiguousarray(image).copy();image.setflags(write=False)
            self.version+=1
            self.current=_SubgoalSnapshot(r.epoch,self.version,r.observation_version,image,r.prompt,r.seed,
                elapsed*1000,time.time(),response.get('server_timing'))
            assert self.current.source_observation_version==submission.observation_version
            self.pending=None
            return self.current

    def reset(self):
        with self.lock:
            self.epoch+=1;self.pending=None;self.current=None;self.prompt=None

    def stop(self):
        with self.lock:self.stopped=True

    def status(self):
        return dict(mode='sync_current',epoch=self.epoch,last_error=None,
                    latest_observation_version=self.observation_version,
                    cached_source_observation_version=None if self.current is None else self.current.source_observation_version)
