#!/usr/bin/env python3
"""Replay official RoboTwin trajectories and emit v6 stage boundary frames.

The source HDF5/video files are never copied.  Episode ``i`` is reconstructed
with ``seed.txt[i]`` and ``_traj_data/episode{i}.pkl``; the resulting boundary
manifest points back to the existing frame stream.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import h5py
import yaml

sys.path.append("./")
sys.path.append("./policy")

from envs import CONFIGS_PATH

from cosmos_policy.stage_success import (
    STAGE_CONTRACT_VERSION,
    SUBGOAL_TASKS,
    stage_count_for_task,
    stage_texts_for_task,
    task_stage_success,
)
from cosmos_policy.subtask_oracle import ExpertSubtaskCapture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-recorded-terminal-on-replay-failure",
        action="store_true",
        help=(
            "accept an official-clean episode whose action replay reaches every "
            "internal v6 stage and preserves the exact recorded frame count but "
            "misses terminal check_success because of physics replay drift"
        ),
    )
    parser.add_argument("--clear-cache-freq", type=int, default=5)
    return parser.parse_args()


def _task_instance(task_name: str) -> Any:
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def _embodiment_config(robot_file: str) -> dict[str, Any]:
    with open(os.path.join(robot_file, "config.yml"), encoding="utf-8") as stream:
        return yaml.load(stream.read(), Loader=yaml.FullLoader)


def _load_base_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        args = yaml.load(stream.read(), Loader=yaml.FullLoader)
    with open(
        os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8"
    ) as stream:
        embodiments = yaml.load(stream.read(), Loader=yaml.FullLoader)
    selected = args["embodiment"]
    if len(selected) != 1:
        raise ValueError("v6 extraction currently requires one dual-arm embodiment")
    robot_file = embodiments[selected[0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["left_embodiment_config"] = _embodiment_config(robot_file)
    args["right_embodiment_config"] = _embodiment_config(robot_file)
    args["dual_arm_embodied"] = True
    args["embodiment_name"] = str(selected[0])
    args["need_plan"] = False
    args["save_data"] = False
    args["collect_data"] = False
    args["eval_mode"] = False
    args["render_freq"] = 0
    return args


def _source_dir(raw_root: Path, task_name: str) -> Path:
    candidates = sorted((raw_root / task_name).glob("demo_clean*"))
    candidates = [
        path
        for path in candidates
        if (path / "seed.txt").is_file() and (path / "_traj_data").is_dir()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one official clean source for {task_name}; got {candidates}"
        )
    return candidates[0]


def _read_existing(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    result: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["task"]), int(row["local_episode_index"]))
            if key in result:
                raise ValueError(f"duplicate existing boundary record {key}")
            result.add(key)
    return result


def _hdf5_length(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        return int(handle["joint_action/vector"].shape[0])


def _extract_episode(
    task_name: str,
    local_index: int,
    seed: int,
    source: Path,
    base_args: dict[str, Any],
    *,
    allow_recorded_terminal_on_replay_failure: bool = False,
) -> dict[str, Any]:
    args = copy.deepcopy(base_args)
    args["task_name"] = task_name
    args["save_path"] = str(source)
    task_env = _task_instance(task_name)
    try:
        task_env.setup_demo(
            now_ep_num=local_index,
            seed=int(seed),
            is_test=True,
            **args,
        )
        count = stage_count_for_task(task_name, task_env)
        initially_satisfied = [
            phase
            for phase in range(count)
            if task_stage_success(task_env, task_name, phase)
        ]
        if initially_satisfied:
            raise RuntimeError(
                "episode starts with stage-success already true for phases "
                f"{initially_satisfied}"
            )
        trajectory = task_env.load_tran_data(local_index)
        task_env.set_path_lst(
            {
                "need_plan": False,
                "left_joint_path": trajectory["left_joint_path"],
                "right_joint_path": trajectory["right_joint_path"],
            }
        )
        capture = ExpertSubtaskCapture(task_env)
        capture.install()
        try:
            task_env.play_once()
        finally:
            capture.restore()
        replay_plan_success = bool(task_env.plan_success)
        replay_official_success = bool(
            task_stage_success(
                task_env,
                task_name,
                stage_count_for_task(task_name, task_env) - 1,
            )
        )
        if not replay_plan_success:
            raise RuntimeError("recorded expert replay did not complete its motion plan")

        expected_length = _hdf5_length(
            source / "data" / f"episode{local_index}.hdf5"
        )
        if capture.frame_count != expected_length:
            raise RuntimeError(
                f"frame-count mismatch replay={capture.frame_count} hdf5={expected_length}"
            )
        texts = stage_texts_for_task(task_name, task_env)
        missing = [phase for phase in range(count - 1) if phase not in capture.stage_events]
        if missing:
            raise RuntimeError(f"expert never reached internal stages {missing}")
        if not replay_official_success:
            if not allow_recorded_terminal_on_replay_failure:
                raise RuntimeError(
                    "recorded expert replay did not reach official success"
                )
            print(
                f"[v6-boundary] WARNING task={task_name} episode={local_index} "
                "uses the official recorded HDF5 terminal frame because exact-length "
                "action replay drifted after all internal stages",
                flush=True,
            )
        stages: list[dict[str, Any]] = []
        previous = 0
        for phase in range(count):
            terminal = phase == count - 1
            end_frame = (
                expected_length
                if terminal
                else int(capture.stage_events[phase]["frame_count"])
            )
            if end_frame <= previous:
                raise RuntimeError(
                    f"non-increasing stage frames at phase {phase}: {previous}->{end_frame}"
                )
            stages.append(
                {
                    "phase": phase,
                    "subtask_text": texts[phase],
                    "start_frame": previous,
                    "end_frame": end_frame,
                    "terminal": terminal,
                    "boundary_source": (
                        "official_task_success_episode_final"
                        if terminal
                        else "first_recorded_frame_with_stage_success"
                    ),
                }
            )
            previous = end_frame
        return {
            "contract_version": STAGE_CONTRACT_VERSION,
            "task": task_name,
            "local_episode_index": local_index,
            "seed": int(seed),
            "frame_count": expected_length,
            "stage_count": count,
            "terminal_replay_plan_success": replay_plan_success,
            "terminal_replay_official_success": replay_official_success,
            "terminal_goal_source": "official_recorded_hdf5_final_frame",
            "source_hdf5": str(
                source / "data" / f"episode{local_index}.hdf5"
            ),
            "source_hdf5_bytes": (
                source / "data" / f"episode{local_index}.hdf5"
            ).stat().st_size,
            "stages": stages,
        }
    finally:
        try:
            task_env.close_env()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    tasks = sorted(set(args.tasks or SUBGOAL_TASKS))
    unknown = sorted(set(tasks) - SUBGOAL_TASKS)
    if unknown:
        raise ValueError(f"tasks have no v6 contract: {unknown}")
    if not 0 <= args.episode_start < args.episode_end <= 50:
        raise ValueError("episode range must satisfy 0 <= start < end <= 50")
    existing = _read_existing(args.output) if args.resume else set()
    if args.output.exists() and not args.resume:
        raise FileExistsError(
            f"output already exists; pass --resume to append missing rows: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base_args = _load_base_config(args.task_config)
    completed = 0
    for task_name in tasks:
        source = _source_dir(args.raw_root, task_name)
        seeds = [int(value) for value in (source / "seed.txt").read_text().split()]
        if len(seeds) != 50:
            raise ValueError(f"{task_name} expected 50 seeds, found {len(seeds)}")
        for local_index in range(args.episode_start, args.episode_end):
            key = (task_name, local_index)
            if key in existing:
                continue
            try:
                row = _extract_episode(
                    task_name,
                    local_index,
                    seeds[local_index],
                    source,
                    base_args,
                    allow_recorded_terminal_on_replay_failure=(
                        args.allow_recorded_terminal_on_replay_failure
                    ),
                )
            except Exception as exc:
                print(
                    f"[v6-boundary] failed task={task_name} episode={local_index} "
                    f"seed={seeds[local_index]} error={exc}",
                    flush=True,
                )
                traceback.print_exc()
                return 1
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            completed += 1
            print(
                f"[v6-boundary] complete task={task_name} episode={local_index} "
                f"seed={seeds[local_index]} stages={row['stage_count']} "
                f"frames={row['frame_count']}",
                flush=True,
            )
    print(
        json.dumps(
            {
                "contract_version": STAGE_CONTRACT_VERSION,
                "new_records": completed,
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()
    raise SystemExit(main())
