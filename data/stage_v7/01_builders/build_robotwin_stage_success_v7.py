#!/usr/bin/env python3
"""Build GoalWAM stage-success-v7 as a lightweight transform of v6.

V7 keeps the authoritative v6 predicates and fixed single-frame goals for the
remaining multi-stage tasks, but converts three empirically weak families to a
single terminal episode goal:

* open_microwave
* handover_block
* place_object_basket

No parquet or video payload is copied.  The output runtime links the immutable
official LeRobot v3 payload already referenced by v6 and materializes only the
small metadata files whose goal policy changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE = Path(os.environ.get("VIGAR_DATASET_V6", "/path/to/stage_v6"))
DEFAULT_OUTPUT = Path(os.environ.get("VIGAR_DATASET_V7", "/path/to/stage_v7"))

EPISODE_GOAL_OVERRIDES = frozenset(
    {"open_microwave", "handover_block", "place_object_basket"}
)
GOAL_GRAPH_VERSION = "official50-clean50-stage-success-v7-19s31e/v1"
RECIPE = "official50-clean50-stage-success-v7-19s31e-train2500-v1"

GOAL_GRAPH_NAME = "manual_goal_graph_19s31e_stagev7_v1.json"
AUDIT_NAME = "boundary_audit_stagev7_v1.json"
MANIFEST_NAME = "train2500_manifest_stagev7_v1.json"
ACTION_STATS_NAME = "train2500_action_stats_stagev7_v1.json"
EVAL_SCHEMA_NAME = "robotwin_subtask_oracle_eval_stagev7_v1.json"
BOUNDARIES_NAME = "stage_boundaries_v7.jsonl"
BOUNDARY_SUMMARY_NAME = "stage_boundaries_v7.summary.json"
CHANGE_RECORD_NAME = "goal_policy_v7_change_record.json"

SOURCE_GRAPH_NAME = "manual_goal_graph_22s28e_stagev6_v2.json"
SOURCE_AUDIT_NAME = "boundary_audit_stagev6_v2.json"
SOURCE_MANIFEST_NAME = "train2500_manifest_stagev6_v2.json"
SOURCE_ACTION_STATS_NAME = "train2500_action_stats_stagev6_v2.json"
SOURCE_EVAL_SCHEMA_NAME = "robotwin_subtask_oracle_eval_stagev6_v2.json"
SOURCE_BOUNDARIES_NAME = "stage_boundaries_v6.jsonl"

SUCCESS_EVIDENCE = {
    "checkpoint": "iter_000032000",
    "eval_split": "clean50",
    "episodes_per_task": 50,
    "open_microwave": {"successes": 1, "episodes": 50},
    "handover_block": {"successes": 14, "episodes": 50},
    "place_object_basket": {"successes": 9, "episodes": 50},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="validate and print the prospective v7 contract without writing",
    )
    return parser.parse_args()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json_bytes(row).decode("utf-8"))
            stream.write("\n")


def require_source(source: Path) -> None:
    required = (
        "BUILD_COMPLETE",
        "runtime.meta.json",
        "meta/annotations.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "family_mapping_v1.json",
        SOURCE_GRAPH_NAME,
        SOURCE_AUDIT_NAME,
        SOURCE_MANIFEST_NAME,
        SOURCE_ACTION_STATS_NAME,
        SOURCE_EVAL_SCHEMA_NAME,
        SOURCE_BOUNDARIES_NAME,
        "stage_contract_v6.json",
        "data",
        "videos",
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/episodes",
    )
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"incomplete v6 source {source}: missing={missing}")


def transform_annotations(
    source_annotations: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    annotations = json.loads(json.dumps(source_annotations))
    converted = 0
    for key, row in annotations.items():
        family = str(row["family"])
        row["goal_graph_version"] = GOAL_GRAPH_VERSION
        if family not in EPISODE_GOAL_OVERRIDES:
            continue
        frame_count = int(row["frame_count"])
        if frame_count < 1:
            raise ValueError(f"invalid frame count for episode {key}: {frame_count}")
        # Keep the episode's own official prompt.  RoboTwin clean50 contains
        # deliberate language variation within a family, and v7 must not
        # collapse that variation while changing only the visual goal policy.
        prompt = str(row["task_name"]).strip()
        row["goal_mode"] = "episode_goal"
        row["action_steps"] = [
            {
                "track": "default",
                "start_frame": 0,
                "end_frame": frame_count,
                "action_text": prompt,
                "skill": prompt,
                "is_mistake": False,
                "info": {
                    "family": family,
                    "stage_index": 0,
                    "semantic_stage_index": 0,
                    "stage_count": 1,
                    "boundary_source": "v7_episode_goal_override",
                    "stage_contract_version": None,
                    "stage_criterion": None,
                },
            }
        ]
        row["key_frame"] = {"single": [frame_count - 1], "dual": []}
        converted += 1
    if converted != 150:
        raise ValueError(f"expected 150 converted episodes, got {converted}")
    return annotations


def transform_goal_graph(
    source_graph: dict[str, Any],
) -> dict[str, Any]:
    graph = json.loads(json.dumps(source_graph))
    graph["version"] = GOAL_GRAPH_VERSION
    design = graph["design"]
    design["subgoal_families"] = 19
    design["episode_goal_families"] = 31
    design["principle"] = (
        "Use the first expert frame satisfying the authoritative stage-success "
        "predicate for each retained intermediate stage, and use the terminal "
        "expert frame for every episode-goal family."
    )
    design["v7_episode_goal_overrides"] = sorted(EPISODE_GOAL_OVERRIDES)
    design["v7_override_principle"] = (
        "Use one terminal episode goal for families whose v6 intermediate-goal "
        "composition underperformed clean50; retain all other v6 predicates and "
        "single-match monotonic switching unchanged."
    )
    for family in EPISODE_GOAL_OVERRIDES:
        row = graph["families"][family]
        terminal_text = str(row["stage_texts"][-1])
        row.update(
            {
                "mode": "episode_goal",
                "stage_texts": [terminal_text],
                "stage_criteria": [],
                "boundary_resolver": "terminal frame",
                "observed_subtask_counts": [1],
                "count_histogram": {"1": 50},
                "boundary_strategy_histogram": {
                    "v7_episode_goal_override": 50
                },
                "v7_replaced_stage_count": 2,
            }
        )
    return graph


def transform_audit(
    source_audit: dict[str, Any],
    annotations: dict[str, dict[str, Any]],
    active_boundary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    audit = json.loads(json.dumps(source_audit))
    audit["goal_graph_version"] = GOAL_GRAPH_VERSION
    for row in audit["episodes"]:
        if row["family"] not in EPISODE_GOAL_OVERRIDES:
            continue
        length = int(row["length"])
        row["goal_mode"] = "episode_goal"
        row["endpoints"] = [length]
        row["normalized_endpoints"] = [1.0]
        row["strategy"] = "v7_episode_goal_override"
    summary = audit["summary"]
    summary["subgoal_families"] = 19
    summary["episode_goal_families"] = 31
    summary["subtasks"] = sum(
        len(row["action_steps"]) for row in annotations.values()
    )
    summary["terminal_replay_exceptions"] = sum(
        not bool(row.get("terminal_replay_official_success", True))
        for row in active_boundary_rows
    )
    summary["v7_converted_episodes"] = 150
    return audit


def metadata_hashes(meta: Path) -> dict[str, str]:
    names = ("info.json", "episodes.jsonl", "tasks.jsonl", "annotations.json")
    return {name: sha256_file(meta / name) for name in names}


def metadata_digest(hashes: dict[str, str]) -> str:
    names = ("info.json", "episodes.jsonl", "tasks.jsonl", "annotations.json")
    payload = [{"name": name, "sha256": hashes[name]} for name in names]
    return sha256_bytes(canonical_json_bytes(payload))


def transform_manifest(
    source_manifest: dict[str, Any],
    annotations: dict[str, dict[str, Any]],
    hashes: dict[str, str],
) -> dict[str, Any]:
    manifest = json.loads(json.dumps(source_manifest))
    manifest["recipe"] = RECIPE
    manifest["source"].update(hashes)
    manifest["source"]["dataset_metadata_sha256"] = metadata_digest(hashes)
    contracts = manifest["contracts"]
    contracts["materialized_annotation_goal_graph_version"] = GOAL_GRAPH_VERSION
    contracts["semantic_family_stage_units"] = 74
    contracts["stage_boundary_source"] = (
        "v6 same-seed first-true stage_success for 19 subgoal families; "
        "terminal frame for three v7 episode-goal overrides"
    )
    contracts["v7_episode_goal_overrides"] = sorted(EPISODE_GOAL_OVERRIDES)
    for family in EPISODE_GOAL_OVERRIDES:
        manifest["families"][family] = {
            "episodes": 50,
            "split_episodes": {"train": 50, "val": 0, "test": 0},
            "expected_subtask_count": 1,
        }
    total_subtasks = sum(len(row["action_steps"]) for row in annotations.values())
    summary = manifest["summary"]
    summary["single_subtask_families"] = 31
    summary["multi_subtask_families"] = 19
    summary["semantic_family_stage_units"] = 74
    summary["by_split"]["train"]["subtasks"] = total_subtasks
    return manifest


def transform_eval_schema(
    source_schema: dict[str, Any],
    annotations_sha256: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    schema = json.loads(json.dumps(source_schema))
    contract = schema["contract"]
    contract["goal_policy_version"] = GOAL_GRAPH_VERSION
    contract["goal_image_source"] = (
        "same-seed fixed expert frame: first authoritative stage_success for "
        "19 subgoal families, terminal frame for 31 episode-goal families"
    )
    contract["v7_episode_goal_overrides"] = sorted(EPISODE_GOAL_OVERRIDES)
    schema["source"] = {
        "annotations_sha256": annotations_sha256,
        "manifest_assignment_sha256": manifest["assignment"][
            "assignment_sha256"
        ],
        "manifest_sha256": manifest_sha256,
    }
    for family in EPISODE_GOAL_OVERRIDES:
        row = schema["tasks"][family]
        terminal_text = str(row["steps"][-1]["subtask_text"])
        row.clear()
        row.update(
            {
                "expected_subtask_count": 1,
                "goal_graph_mode": "episode_goal_override_v7",
                "goal_graph_rationale": (
                    "V7 replaces the underperforming two-stage composition with "
                    "one fixed terminal episode goal."
                ),
                "simulator_task": family,
                "steps": [
                    {
                        "endpoint_fraction": 1.0,
                        "index": 0,
                        "source_train_episodes": 50,
                        "subtask_text": terminal_text,
                    }
                ],
            }
        )
    return schema


def link_payload(staging: Path, source: Path) -> None:
    (staging / "meta").mkdir()
    for relative in (
        "data",
        "videos",
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/episodes",
    ):
        src = source / relative
        dst = staging / relative
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    for relative in (
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "family_mapping_v1.json",
        "stage_contract_v6.json",
    ):
        shutil.copy2(source / relative, staging / relative)


def validate_contract(
    annotations: dict[str, dict[str, Any]],
    graph: dict[str, Any],
    manifest: dict[str, Any],
    eval_schema: dict[str, Any],
) -> dict[str, Any]:
    if set(map(int, annotations)) != set(range(2500)):
        raise ValueError("annotation coverage must be exactly episode 0..2499")
    families = set(graph["families"])
    if len(families) != 50 or set(eval_schema["tasks"]) != families:
        raise ValueError("graph/eval schema must cover the same 50 families")
    subgoal = sorted(
        family
        for family, row in graph["families"].items()
        if row["mode"] == "subgoal"
    )
    episode_goal = sorted(families - set(subgoal))
    if len(subgoal) != 19 or len(episode_goal) != 31:
        raise ValueError(
            f"expected 19 subgoal + 31 episode-goal families, got "
            f"{len(subgoal)} + {len(episode_goal)}"
        )
    if not EPISODE_GOAL_OVERRIDES <= set(episode_goal):
        raise ValueError("v7 overrides were not converted to episode goal")
    total_subtasks = sum(len(row["action_steps"]) for row in annotations.values())
    semantic_units = int(manifest["contracts"]["semantic_family_stage_units"])
    if total_subtasks != 3691 or semantic_units != 74:
        raise ValueError(
            f"expected 3691 materialized episode-stages and 74 semantic units, "
            f"got {total_subtasks} and {semantic_units}"
        )
    for row in annotations.values():
        family = row["family"]
        if family not in EPISODE_GOAL_OVERRIDES:
            continue
        steps = row["action_steps"]
        if len(steps) != 1:
            raise ValueError(f"{family} remains multi-stage")
        if steps[0]["start_frame"] != 0 or steps[0]["end_frame"] != row["frame_count"]:
            raise ValueError(f"{family} does not span the complete episode")
        if row["key_frame"]["single"] != [row["frame_count"] - 1]:
            raise ValueError(f"{family} does not use the terminal fixed goal")
    return {
        "tasks": 50,
        "episodes": 2500,
        "episodes_per_task": 50,
        "subgoal_tasks": subgoal,
        "episode_goal_tasks": episode_goal,
        "semantic_family_stage_units": 74,
        "materialized_episode_stages": total_subtasks,
        "training_windows": 74 * 6400,
        "windows_per_stage_unit": 6400,
    }


def build(source: Path, output: Path, audit_only: bool) -> dict[str, Any]:
    require_source(source)
    source_meta = read_json(source / "runtime.meta.json")
    source_annotations = read_json(source / "meta/annotations.json")
    source_subgoal = set(source_meta["subgoal_tasks"])
    if len(source_subgoal) != 22 or not EPISODE_GOAL_OVERRIDES <= source_subgoal:
        raise ValueError(
            "v7 source must be the 22-subgoal v6 runtime and contain all overrides"
        )

    annotations = transform_annotations(source_annotations)
    graph = transform_goal_graph(read_json(source / SOURCE_GRAPH_NAME))
    source_boundary_rows = read_jsonl(source / SOURCE_BOUNDARIES_NAME)
    active_subgoal = source_subgoal - EPISODE_GOAL_OVERRIDES
    boundary_rows = [
        row for row in source_boundary_rows if row["task"] in active_subgoal
    ]
    if len(boundary_rows) != 950:
        raise ValueError(f"expected 19x50 active boundary rows, got {len(boundary_rows)}")
    audit = transform_audit(
        read_json(source / SOURCE_AUDIT_NAME), annotations, boundary_rows
    )

    if audit_only:
        # Hashes that depend on serialized output are intentionally omitted.
        source_manifest = read_json(source / SOURCE_MANIFEST_NAME)
        provisional_manifest = transform_manifest(
            source_manifest,
            annotations,
            {
                "info.json": sha256_file(source / "meta/info.json"),
                "episodes.jsonl": sha256_file(source / "meta/episodes.jsonl"),
                "tasks.jsonl": sha256_file(source / "meta/tasks.jsonl"),
                "annotations.json": sha256_bytes(
                    (json.dumps(
                        annotations,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    ) + "\n").encode("utf-8")
                ),
            },
        )
        provisional_schema = transform_eval_schema(
            read_json(source / SOURCE_EVAL_SCHEMA_NAME),
            provisional_manifest["source"]["annotations.json"],
            provisional_manifest,
            "audit-only",
        )
        return validate_contract(
            annotations, graph, provisional_manifest, provisional_schema
        )

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        link_payload(staging, source)
        write_json(staging / "meta/annotations.json", annotations)
        write_json(staging / GOAL_GRAPH_NAME, graph)
        write_json(staging / AUDIT_NAME, audit)
        write_jsonl(staging / BOUNDARIES_NAME, boundary_rows)
        write_json(
            staging / BOUNDARY_SUMMARY_NAME,
            {
                "schema_version": "robotwin-stage-boundaries-summary/v1",
                "goal_policy_version": GOAL_GRAPH_VERSION,
                "stage_contract_version": source_meta["stage_contract_version"],
                "source_boundary_file": str(source / SOURCE_BOUNDARIES_NAME),
                "source_boundary_sha256": sha256_file(
                    source / SOURCE_BOUNDARIES_NAME
                ),
                "active_subgoal_tasks": sorted(active_subgoal),
                "records": len(boundary_rows),
                "expected_records": 950,
            },
        )

        hashes = metadata_hashes(staging / "meta")
        manifest = transform_manifest(
            read_json(source / SOURCE_MANIFEST_NAME), annotations, hashes
        )
        write_json(staging / MANIFEST_NAME, manifest)
        manifest_sha256 = sha256_file(staging / MANIFEST_NAME)

        action_stats = read_json(source / SOURCE_ACTION_STATS_NAME)
        action_stats["provenance"].update(
            {
                "dataset": str(output),
                "annotations_sha256": hashes["annotations.json"],
                "episode_manifest_sha256": manifest_sha256,
                "goal_policy_version": GOAL_GRAPH_VERSION,
            }
        )
        write_json(staging / ACTION_STATS_NAME, action_stats)

        eval_schema = transform_eval_schema(
            read_json(source / SOURCE_EVAL_SCHEMA_NAME),
            hashes["annotations.json"],
            manifest,
            manifest_sha256,
        )
        write_json(staging / EVAL_SCHEMA_NAME, eval_schema)
        counts = validate_contract(annotations, graph, manifest, eval_schema)

        change_record = {
            "schema_version": "goalwam-goal-policy-change/v1",
            "version": GOAL_GRAPH_VERSION,
            "source_runtime": str(source),
            "source_runtime_meta_sha256": sha256_file(
                source / "runtime.meta.json"
            ),
            "converted_from_subgoal_to_episode_goal": sorted(
                EPISODE_GOAL_OVERRIDES
            ),
            "unchanged_subgoal_tasks": sorted(active_subgoal),
            "decision_evidence": SUCCESS_EVIDENCE,
            "decision": (
                "Replace each selected two-stage annotation by one full-episode "
                "segment and use the terminal expert frame as its fixed goal."
            ),
            "goal_bank": False,
            "switch_confirmation_steps": 1,
            "switch_latch": "monotonic",
        }
        write_json(staging / CHANGE_RECORD_NAME, change_record)

        artifact_names = (
            "meta/annotations.json",
            "meta/episodes.jsonl",
            "meta/tasks.jsonl",
            "family_mapping_v1.json",
            "stage_contract_v6.json",
            GOAL_GRAPH_NAME,
            AUDIT_NAME,
            MANIFEST_NAME,
            ACTION_STATS_NAME,
            EVAL_SCHEMA_NAME,
            BOUNDARIES_NAME,
            BOUNDARY_SUMMARY_NAME,
            CHANGE_RECORD_NAME,
        )
        runtime_meta = {
            "schema_version": source_meta["schema_version"],
            "goal_policy_version": GOAL_GRAPH_VERSION,
            "source_runtime_root": str(source),
            "source_dataset_root": source_meta["source_dataset_root"],
            "raw_annotation_source_root": source_meta[
                "raw_annotation_source_root"
            ],
            "runtime_root": str(output),
            "layout": (
                "lightweight symlink runtime; official parquet and videos are "
                "not copied"
            ),
            "total_tasks": 50,
            "episodes_per_task": 50,
            "total_episodes": 2500,
            "subgoal_tasks": counts["subgoal_tasks"],
            "episode_goal_tasks": counts["episode_goal_tasks"],
            "v7_episode_goal_overrides": sorted(EPISODE_GOAL_OVERRIDES),
            "goal_graph_version": GOAL_GRAPH_VERSION,
            "stage_contract_version": source_meta["stage_contract_version"],
            "training_goal_graph_version": None,
            "training_split": "train",
            "semantic_family_stage_units": 74,
            "artifacts": {
                name: {
                    "sha256": sha256_file(staging / name),
                    "bytes": (staging / name).stat().st_size,
                }
                for name in artifact_names
            },
            "symlinks": {
                relative: os.readlink(staging / relative)
                for relative in (
                    "data",
                    "videos",
                    "meta/info.json",
                    "meta/stats.json",
                    "meta/tasks.parquet",
                    "meta/episodes",
                )
            },
        }
        write_json(staging / "runtime.meta.json", runtime_meta)
        (staging / "BUILD_COMPLETE").write_text(
            sha256_file(staging / "runtime.meta.json") + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
        return {
            **counts,
            "output_root": str(output),
            "annotations_sha256": hashes["annotations.json"],
            "manifest_sha256": manifest_sha256,
            "action_stats_sha256": sha256_file(output / ACTION_STATS_NAME),
            "eval_schema_sha256": sha256_file(output / EVAL_SCHEMA_NAME),
            "runtime_meta_sha256": sha256_file(output / "runtime.meta.json"),
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    result = build(args.source_root, args.output_root, args.audit_only)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    if not args.audit_only:
        print(f"BUILD_OK={args.output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
