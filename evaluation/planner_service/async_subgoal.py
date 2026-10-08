#!/usr/bin/env python3
"""Gateway combining asynchronous subgoal planner goals with ViGAR actions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import socket
import threading
import time
from typing import Any

import numpy as np

# Adapted from the ViGAR deployment gateway.
# Latest-only, epoch reset and immutable snapshot behavior are unchanged.
# Only the RoboTwin portrait canvas differs from the AgiBot deployment.
TARGET_HW = (384, 320)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SubgoalRequest:
    epoch: int
    observation_version: int
    observation: np.ndarray
    prompt: str
    seed: int | None


@dataclass(frozen=True)
class _SubgoalSubmission:
    epoch: int
    observation_version: int
    prompt_changed: bool


@dataclass(frozen=True)
class _SubgoalSnapshot:
    epoch: int
    version: int
    source_observation_version: int
    image: np.ndarray
    prompt: str
    seed: int | None
    generation_ms: float
    generated_at_unix: float
    server_timing: Any


class AsyncSubgoalProducer:
    """Continuously turn the latest submitted observation into a cached goal.

    There is at most one subgoal planner request in flight and one pending request. A new
    submission replaces the pending request, so observations never build up in
    a stale FIFO queue. Completed subgoals atomically replace the current cache.
    """

    def __init__(self, planner_client: Any) -> None:
        self._planner_client = planner_client
        self._condition = threading.Condition()
        self._epoch = 0
        self._observation_version = 0
        self._subgoal_version = 0
        self._active_prompt: str | None = None
        self._pending: _SubgoalRequest | None = None
        self._in_flight: _SubgoalRequest | None = None
        self._current: _SubgoalSnapshot | None = None
        self._last_error: dict[str, Any] | None = None
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            name="vigar-subgoal-producer",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        observation: np.ndarray,
        prompt: str,
        seed: int | None,
    ) -> _SubgoalSubmission:
        observation_copy = np.ascontiguousarray(observation).copy()
        with self._condition:
            if self._stopped:
                raise RuntimeError("The asynchronous subgoal producer is stopped")

            prompt_changed = (
                self._active_prompt is not None and prompt != self._active_prompt
            )
            if prompt_changed:
                # An in-flight result from the previous prompt may still arrive.
                # Advancing the epoch prevents it from being published.
                self._epoch += 1
                self._pending = None
                self._current = None
                self._last_error = None
            self._active_prompt = prompt

            self._observation_version += 1
            request = _SubgoalRequest(
                epoch=self._epoch,
                observation_version=self._observation_version,
                observation=observation_copy,
                prompt=prompt,
                seed=seed,
            )
            # Latest-only mailbox: overwrite any observation that has not yet
            # started subgoal planner inference.
            self._pending = request
            self._condition.notify_all()
            return _SubgoalSubmission(
                epoch=request.epoch,
                observation_version=request.observation_version,
                prompt_changed=prompt_changed,
            )

    def snapshot_or_wait(
        self,
        submission: _SubgoalSubmission,
        *,
        timeout: float | None,
    ) -> _SubgoalSnapshot:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                current = self._current
                if current is not None and current.epoch == submission.epoch:
                    return current
                if self._stopped:
                    raise RuntimeError("The asynchronous subgoal producer stopped")

                error = self._last_error
                has_more_work = self._pending is not None or self._in_flight is not None
                if (
                    error is not None
                    and error["epoch"] == submission.epoch
                    and not has_more_work
                ):
                    raise RuntimeError(
                        "subgoal planner failed before the first subgoal was available: "
                        f"{error['message']}"
                    )

                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting for the first subgoal planner subgoal"
                    )
                self._condition.wait(remaining)

    def reset(self) -> None:
        with self._condition:
            self._epoch += 1
            self._active_prompt = None
            self._pending = None
            self._current = None
            self._last_error = None
            self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._condition:
            current = self._current
            return {
                "mode": "async_latest_only",
                "epoch": self._epoch,
                "latest_observation_version": self._observation_version,
                "cached_subgoal_version": (
                    None if current is None else current.version
                ),
                "cached_source_observation_version": (
                    None if current is None else current.source_observation_version
                ),
                "cached_generated_at_unix": (
                    None if current is None else current.generated_at_unix
                ),
                "in_flight_observation_version": (
                    None
                    if self._in_flight is None
                    else self._in_flight.observation_version
                ),
                "pending_observation_version": (
                    None if self._pending is None else self._pending.observation_version
                ),
                "last_error": None if self._last_error is None else dict(self._last_error),
            }

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopped:
                    self._condition.wait()
                if self._stopped:
                    return
                request = self._pending
                self._pending = None
                self._in_flight = request

            assert request is not None
            started = time.monotonic()
            try:
                planner_request: dict[str, Any] = {
                    "cmd": "infer",
                    "image": request.observation,
                    "prompt": request.prompt,
                }
                if request.seed is not None:
                    planner_request["seed"] = request.seed
                response = self._planner_client.call(planner_request)
                generation_ms = (time.monotonic() - started) * 1000.0
                subgoal = np.asarray(response["goal_image"], dtype=np.uint8)
                if subgoal.shape != (*TARGET_HW, 3):
                    raise ValueError(
                        f"subgoal planner worker returned invalid image shape: {subgoal.shape}"
                    )
                subgoal = np.ascontiguousarray(subgoal).copy()
                subgoal.setflags(write=False)

                with self._condition:
                    if request.epoch == self._epoch and not self._stopped:
                        self._subgoal_version += 1
                        self._current = _SubgoalSnapshot(
                            epoch=request.epoch,
                            version=self._subgoal_version,
                            source_observation_version=request.observation_version,
                            image=subgoal,
                            prompt=request.prompt,
                            seed=response.get("seed"),
                            generation_ms=generation_ms,
                            generated_at_unix=time.time(),
                            server_timing=response.get("server_timing"),
                        )
                        self._last_error = None
                        logger.info(
                            "Published subgoal version=%d from observation=%d in %.1f ms",
                            self._subgoal_version,
                            request.observation_version,
                            generation_ms,
                        )
                    self._in_flight = None
                    self._condition.notify_all()
            except Exception as exc:
                logger.exception(
                    "subgoal planner generation failed for observation version=%d",
                    request.observation_version,
                )
                with self._condition:
                    if request.epoch == self._epoch and not self._stopped:
                        self._last_error = {
                            "epoch": request.epoch,
                            "observation_version": request.observation_version,
                            "message": f"{type(exc).__name__}: {exc}",
                            "failed_at_unix": time.time(),
                        }
                    self._in_flight = None
                    self._condition.notify_all()
