#!/usr/bin/env python3
"""Validate and merge sharded RoboTwin v6 stage-boundary manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "policy"))

from cosmos_policy.stage_success import (  # noqa: E402
    STAGE_CONTRACT_VERSION,
    SUBGOAL_TASKS,
    TASK_STAGE_TEXTS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--replacement-input",
        type=Path,
        action="append",
        default=[],
        help=(
            "complete per-task correction manifests; every task present here "
            "must cover episodes 0..49 and atomically replaces that task in "
            "the base inputs"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for path in args.input:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("contract_version") != STAGE_CONTRACT_VERSION:
                    raise ValueError(
                        f"{path}:{line_number}: wrong contract version"
                    )
                key = (str(row["task"]), int(row["local_episode_index"]))
                if key in records:
                    raise ValueError(f"duplicate boundary record {key}")
                records[key] = row

    replacement_records: dict[tuple[str, int], dict[str, Any]] = {}
    for path in args.replacement_input:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("contract_version") != STAGE_CONTRACT_VERSION:
                    raise ValueError(
                        f"{path}:{line_number}: wrong replacement contract version"
                    )
                key = (str(row["task"]), int(row["local_episode_index"]))
                if key in replacement_records:
                    raise ValueError(f"duplicate replacement boundary record {key}")
                replacement_records[key] = row
    replacement_tasks = sorted({task for task, _ in replacement_records})
    for task_name in replacement_tasks:
        if task_name not in SUBGOAL_TASKS:
            raise ValueError(f"replacement task has no v6 contract: {task_name}")
        actual = {
            episode
            for task, episode in replacement_records
            if task == task_name
        }
        expected_episodes = set(range(50))
        if actual != expected_episodes:
            raise ValueError(
                f"replacement task {task_name} must cover episodes 0..49; "
                f"missing={sorted(expected_episodes - actual)} "
                f"extra={sorted(actual - expected_episodes)}"
            )
        for episode in expected_episodes:
            records[(task_name, episode)] = replacement_records[
                (task_name, episode)
            ]

    expected = {
        (task_name, episode)
        for task_name in SUBGOAL_TASKS
        for episode in range(50)
    }
    missing = sorted(expected - set(records))
    extra = sorted(set(records) - expected)
    if missing or extra:
        raise ValueError(
            f"coverage must be exactly 22x50; records={len(records)} "
            f"missing={missing[:20]} extra={extra[:20]}"
        )

    summaries: dict[str, Any] = {}
    replay_terminal_exceptions: list[dict[str, Any]] = []
    for task_name in sorted(SUBGOAL_TASKS):
        rows = [records[(task_name, episode)] for episode in range(50)]
        counts = Counter(int(row["stage_count"]) for row in rows)
        phase_fractions: dict[int, list[float]] = defaultdict(list)
        phase_windows: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            replay_official_success = bool(
                row.get("terminal_replay_official_success", True)
            )
            if not replay_official_success:
                if not bool(row.get("terminal_replay_plan_success", False)):
                    raise ValueError(
                        f"terminal replay exception has no successful motion plan: "
                        f"{task_name}/{row['local_episode_index']}"
                    )
                if row.get("terminal_goal_source") != (
                    "official_recorded_hdf5_final_frame"
                ):
                    raise ValueError(
                        f"terminal replay exception has an invalid goal source: "
                        f"{task_name}/{row['local_episode_index']}"
                    )
                replay_terminal_exceptions.append(
                    {
                        "task": task_name,
                        "local_episode_index": int(row["local_episode_index"]),
                        "seed": int(row["seed"]),
                    }
                )
            frame_count = int(row["frame_count"])
            stages = row["stages"]
            if len(stages) != int(row["stage_count"]):
                raise ValueError(f"stage-count mismatch for {task_name}")
            expected_texts = (
                (TASK_STAGE_TEXTS[task_name][-1],)
                if task_name == "place_bread_basket" and len(stages) == 1
                else TASK_STAGE_TEXTS[task_name]
            )
            if tuple(stage["subtask_text"] for stage in stages) != tuple(
                expected_texts
            ):
                raise ValueError(f"stage-text mismatch for {task_name}")
            previous = 0
            for phase, stage in enumerate(stages):
                start = int(stage["start_frame"])
                end = int(stage["end_frame"])
                if start != previous or not start < end <= frame_count:
                    raise ValueError(
                        f"invalid boundary {task_name}/{row['local_episode_index']} "
                        f"phase={phase} start={start} end={end} frames={frame_count}"
                    )
                phase_fractions[phase].append(end / frame_count)
                phase_windows[phase].append(end - start)
                previous = end
            if previous != frame_count:
                raise ValueError(f"terminal boundary mismatch for {task_name}")
        summaries[task_name] = {
            "episodes": 50,
            "terminal_replay_official_success": sum(
                bool(row.get("terminal_replay_official_success", True))
                for row in rows
            ),
            "stage_count_histogram": {
                str(key): value for key, value in sorted(counts.items())
            },
            "phases": {
                str(phase): {
                    "endpoint_fraction_min": min(values),
                    "endpoint_fraction_median": statistics.median(values),
                    "endpoint_fraction_max": max(values),
                    "window_frames_min": min(phase_windows[phase]),
                    "window_frames_median": statistics.median(
                        phase_windows[phase]
                    ),
                    "window_frames_max": max(phase_windows[phase]),
                }
                for phase, values in sorted(phase_fractions.items())
            },
        }

    if args.output.exists() or args.summary.exists():
        raise FileExistsError("refusing to overwrite merged output or summary")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for key in sorted(records):
            stream.write(
                json.dumps(records[key], ensure_ascii=False, sort_keys=True) + "\n"
            )
    summary = {
        "contract_version": STAGE_CONTRACT_VERSION,
        "records": len(records),
        "tasks": len(SUBGOAL_TASKS),
        "episodes_per_task": 50,
        "replacement_tasks": replacement_tasks,
        "terminal_replay_exceptions": replay_terminal_exceptions,
        "terminal_replay_exception_count": len(replay_terminal_exceptions),
        "merged_jsonl": str(args.output),
        "merged_jsonl_sha256": _sha256(args.output),
        "task_summaries": summaries,
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in summary if key != "task_summaries"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
