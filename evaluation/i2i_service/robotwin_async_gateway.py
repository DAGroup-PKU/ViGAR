"""Real-robot latest-only asynchronous goal cache adapted to RoboTwin TCP."""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from async_subgoal import AsyncSubgoalProducer
from robotwin_wire import Client, serve


def concat_views(request):
    # Exactly the validated policy server's CPU bilinear preprocessing.
    import torch
    import torch.nn.functional as F

    def resize(rgb, hw):
        tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1)[None].float()
        return F.interpolate(tensor, size=hw, mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy().astype(np.uint8)

    head = np.asarray(request["head"], dtype=np.uint8)
    height, width = head.shape[:2]
    left = resize(request["left"], (height // 2, width // 2))
    right = resize(request["right"], (height // 2, width // 2))
    canvas = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
    return resize(canvas, (384, 320)) if canvas.shape[:2] != (384, 320) else canvas


class Gateway:
    def __init__(self, planner, policy, trace_path, concat=concat_views):
        self.producer = AsyncSubgoalProducer(planner)
        self.policy = policy
        self.concat = concat
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.episode_seed = 0
        self.policy_contract = self.policy.call({"cmd": "ping"})
        if self.policy_contract.get("subtask_text_conditioning", True):
            raise ValueError("Expected the v7 global-task-text executor, not oracle subtask text")

    def __call__(self, request):
        with self.lock:
            command = request.get("cmd", "infer")
            if command == "ping":
                return {"ok": True, "goal_mode": "i2i_async_latest_only", "expert_goals": False}
            if command == "reset":
                self.episode_seed = int(request.get("seed", 0))
                self.producer.reset()
                self.policy.call({"cmd": "reset"})
                return {"ok": True}
            if command != "infer":
                raise ValueError(f"Unsupported gateway command: {command}")
            if set(request) - {"cmd", "head", "left", "right", "state", "task_name", "prompt"}:
                raise ValueError("Gateway must not receive expert goals, subtask IDs or simulator state")
            prompt = str(request["prompt"])
            started = time.monotonic()
            submission = self.producer.submit(self.concat(request), prompt, self.episode_seed)
            # Only cold start / reset / prompt changes block on the planner.
            snapshot = self.producer.snapshot_or_wait(submission, timeout=600)
            goal_wait = time.monotonic() - started
            policy_request = dict(request, goal_concat=snapshot.image)
            if self.policy_contract.get("canonical_task_text_conditioning"):
                task = str(request["task_name"])
                policy_request["prompt"] = f'Complete the RoboTwin task "{" ".join(task.split("_"))}".'
            response = self.policy.call(policy_request)
            trace = {"time": time.time(), "task": request["task_name"], "seed": self.episode_seed,
                     "epoch": snapshot.epoch, "goal_version": snapshot.version,
                     "source_observation_version": snapshot.source_observation_version,
                     "current_observation_version": submission.observation_version,
                     "goal_age_seconds": time.time() - snapshot.generated_at_unix,
                     "goal_wait_seconds": goal_wait, "i2i_ms": snapshot.generation_ms,
                     "action_rpc_seconds": time.monotonic() - started - goal_wait,
                     "planner_error": self.producer.status()["last_error"]}
            with self.trace_path.open("a") as output:
                output.write(json.dumps(trace) + "\n")
            response["_async_goal"] = trace
            return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i2i-port", type=int, required=True)
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--trace", required=True)
    args = parser.parse_args()
    planner, policy = Client(args.i2i_port), Client(args.policy_port)
    gateway = Gateway(planner, policy, args.trace)
    print("I2I_GATEWAY_READY latest_only=1 cold_start_wait=1 oracle_goals=0", flush=True)
    try:
        serve(gateway, args.port)
    finally:
        gateway.producer.stop()
        planner.close()
        policy.close()


if __name__ == "__main__":
    main()
