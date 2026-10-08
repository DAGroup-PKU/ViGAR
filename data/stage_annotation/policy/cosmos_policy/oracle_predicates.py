"""Reusable semantic predicates for RoboTwin oracle subtask switching.

The registry in this module is intentionally simulator-agnostic.  It consumes
two JSON-serialisable state snapshots (captured by :mod:`subtask_oracle`) and
builds a small predicate contract.  The same contract can then be evaluated in
the policy rollout without exposing simulator state to the policy model.

Contracts use disjunctive normal form: at least one ``alternatives`` entry must
match, while every clause inside the selected alternative must match.  Builders
prefer persistent task state (object pose, articulation state, containment) to
transient contact, and use end-effector pose only as a guarded last resort.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable

import numpy as np

from cosmos_policy.stage_success import (
    STAGE_CONTRACT_VERSION,
    SUBGOAL_TASKS,
    stage_scalar_key,
)
from cosmos_policy.task_metrics import PUT_BOTTLES_DUSTBIN_COUNT_KEY


PREDICATE_CONTRACT_VERSION = "robotwin-oracle-predicate/v6"

# Captured for every expert/policy simulator state.  It is primarily useful for
# terminal-phase audit: the last phase never drives another goal switch, so the
# simulator's own task-success predicate is the authoritative completion check.
OFFICIAL_TASK_SUCCESS_KEY = "__official_task_success__"


@dataclass(frozen=True)
class PredicateRule:
    name: str
    patterns: tuple[str, ...]

    def matches(self, text: str) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in self.patterns)


# Order matters: specific interactions must be classified before generic
# movement/grasp verbs.  These rules cover every semantic step in the fixed-50
# schema and remain useful for future schemas with paraphrased instructions.
PREDICATE_RULES = (
    PredicateRule("handover", (r"\bhandoff\b", r"\bhandover\b", r"\btransfer\b", r"\bother arm\b", r"^hand\b")),
    PredicateRule("articulation", (r"\bopen\b", r"\bpull\b.*\b(cabinet|drawer)\b", r"\bturn\b.*\b(switch|qr|panel|sign)\b", r"\brotate\b")),
    PredicateRule("press", (r"\bclick\b", r"\bpress\b", r"\bstamp\b")),
    PredicateRule("shake", (r"\bshake\b",)),
    PredicateRule("scan", (r"\bscan\b",)),
    PredicateRule("dump", (r"\bdump\b", r"\bpour\b")),
    PredicateRule("orient", (r"\bturn\b.*\bupright\b", r"\bstands? upright\b")),
    PredicateRule("place", (r"\bplace\b", r"\bput\b", r"\bset\b", r"\bposition\b", r"\bhang\b", r"\bmove\b.*\b(into|onto|at|rank|spot|area|front|away|target|blue|black|gray|olive)\b")),
    PredicateRule("pick", (r"\bpick\b", r"\bgrasp\b", r"\bgrip\b", r"\blift\b", r"\braise\b", r"\btake hold\b")),
    PredicateRule("transport", (r"\bmove\b",)),
)


# These are semantic exceptions, not task-specific implementations.  They make
# ambiguous short instructions explicit while still routing them to reusable
# builders below.
TASK_PHASE_OVERRIDES: dict[tuple[str, int], str] = {
    ("handover_block", 1): "handover",
    ("handover_block", 2): "handover",
    ("handover_mic", 1): "handover",
    ("hanging_mug", 0): "handover",
    ("dump_bin_bigbin", 0): "handover",
    ("put_object_cabinet", 1): "articulation",
    ("place_phone_stand", 0): "phone_grasp",
    ("put_bottles_dustbin", 0): "monotonic_count",
    ("put_bottles_dustbin", 1): "monotonic_count",
}

# Internal phases use the same task-state thresholds as atomic evaluation:
# 0->1 after one bottle is inside; 1->2 after two bottles are inside.  The
# terminal third-bottle completion remains RoboTwin's official success check.
MONOTONIC_COUNT_THRESHOLDS: dict[tuple[str, int], tuple[str, int]] = {
    ("put_bottles_dustbin", 0): (PUT_BOTTLES_DUSTBIN_COUNT_KEY, 1),
    ("put_bottles_dustbin", 1): (PUT_BOTTLES_DUSTBIN_COUNT_KEY, 2),
}


def resolve_predicate_type(task_name: str, phase: int, text: str) -> str:
    if str(task_name) in SUBGOAL_TASKS:
        return "stage_success"
    override = TASK_PHASE_OVERRIDES.get((str(task_name), int(phase)))
    if override:
        return override
    normalized = " ".join(str(text).strip().split())
    for rule in PREDICATE_RULES:
        if rule.matches(normalized):
            return rule.name
    return "generic"


def _quat_distance(a: list[float], b: list[float]) -> float:
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    qa /= max(float(np.linalg.norm(qa)), 1e-12)
    qb /= max(float(np.linalg.norm(qb)), 1e-12)
    return float(2.0 * math.acos(float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))))


def _pose_delta(before: dict[str, Any], after: dict[str, Any]) -> tuple[float, float]:
    position = float(np.linalg.norm(np.asarray(after["p"]) - np.asarray(before["p"])))
    return position, _quat_distance(before["q"], after["q"])


def _adaptive_tolerance(delta: float, floor: float, ceiling: float) -> float:
    # Strictly below the observed delta so the phase-start state cannot satisfy
    # a target predicate.  The floor is reduced for very small transitions.
    return max(min(float(delta) * 0.45, ceiling), min(floor, float(delta) * 0.45))


def _pose_clause(namespace: str, key: str, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any] | None:
    position_delta, rotation_delta = _pose_delta(before, after)
    if position_delta < 0.003 and rotation_delta < 0.05:
        return None
    return {
        "kind": "pose_target",
        "namespace": namespace,
        "key": key,
        "target": after,
        "check_position": position_delta >= 0.003,
        "check_rotation": rotation_delta >= 0.05,
        "position_tolerance": _adaptive_tolerance(position_delta, 0.008, 0.055),
        "rotation_tolerance": _adaptive_tolerance(rotation_delta, 0.08, 0.55),
        "observed_position_delta": position_delta,
        "observed_rotation_delta": rotation_delta,
    }


def _vector_clause(namespace: str, key: str, before: Any, after: Any) -> dict[str, Any] | None:
    lhs = np.asarray(before, dtype=np.float64).reshape(-1)
    rhs = np.asarray(after, dtype=np.float64).reshape(-1)
    if lhs.shape != rhs.shape or not lhs.size:
        return None
    delta = float(np.max(np.abs(rhs - lhs)))
    if delta < 0.004:
        return None
    return {
        "kind": "vector_target",
        "namespace": namespace,
        "key": key,
        "target": rhs.tolist(),
        "tolerance": _adaptive_tolerance(delta, 0.004, 0.12),
        "observed_delta": delta,
    }


def _scalar_clause(namespace: str, key: str, before: Any, after: Any) -> dict[str, Any] | None:
    if isinstance(after, bool):
        if bool(before) == bool(after):
            return None
        return {
            "kind": "scalar_target",
            "namespace": namespace,
            "key": key,
            "target": bool(after),
            "tolerance": 0.0,
        }
    try:
        delta = abs(float(after) - float(before))
    except (TypeError, ValueError):
        if before == after:
            return None
        return {
            "kind": "scalar_target",
            "namespace": namespace,
            "key": key,
            "target": after,
            "tolerance": 0.0,
        }
    if delta < 0.02:
        return None
    return {
        "kind": "scalar_target",
        "namespace": namespace,
        "key": key,
        "target": float(after),
        "tolerance": _adaptive_tolerance(delta, 0.01, 0.2),
        "observed_delta": delta,
    }


def _common_keys(start: dict[str, Any], target: dict[str, Any], namespace: str) -> list[str]:
    return sorted(set(start.get(namespace, {})) & set(target.get(namespace, {})))


def _candidate_groups(start: dict[str, Any], target: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {
        "task_entity": [],
        "actor": [],
        "relation": [],
        "articulation": [],
        "contact": [],
        "gripper": [],
        "task_scalar": [],
        "end_effector": [],
    }
    for namespace, bucket in (("task_entities", "task_entity"), ("actors", "actor"), ("end_effectors", "end_effector")):
        for key in _common_keys(start, target, namespace):
            clause = _pose_clause(namespace, key, start[namespace][key], target[namespace][key])
            if clause:
                groups[bucket].append(clause)
    task_keys = _common_keys(start, target, "task_entities")
    for left_index, left_key in enumerate(task_keys):
        for right_key in task_keys[left_index + 1 :]:
            start_distance = float(
                np.linalg.norm(
                    np.asarray(start["task_entities"][left_key]["p"], dtype=np.float64)
                    - np.asarray(start["task_entities"][right_key]["p"], dtype=np.float64)
                )
            )
            target_distance = float(
                np.linalg.norm(
                    np.asarray(target["task_entities"][left_key]["p"], dtype=np.float64)
                    - np.asarray(target["task_entities"][right_key]["p"], dtype=np.float64)
                )
            )
            delta = abs(target_distance - start_distance)
            if delta >= 0.01:
                groups["relation"].append(
                    {
                        "kind": "distance_target",
                        "namespace": "task_entities",
                        "keys": [left_key, right_key],
                        "target": target_distance,
                        "tolerance": _adaptive_tolerance(delta, 0.008, 0.05),
                        "observed_delta": delta,
                        "direction": "closer" if target_distance < start_distance else "farther",
                    }
                )
    for key in _common_keys(start, target, "articulations"):
        clause = _vector_clause("articulations", key, start["articulations"][key], target["articulations"][key])
        if clause:
            groups["articulation"].append(clause)
    for arm in _common_keys(start, target, "grippers"):
        clause = _scalar_clause("grippers", arm, start["grippers"][arm], target["grippers"][arm])
        if clause:
            groups["gripper"].append(clause)
    for key in _common_keys(start, target, "task_scalars"):
        clause = _scalar_clause("task_scalars", key, start["task_scalars"][key], target["task_scalars"][key])
        if clause:
            groups["task_scalar"].append(clause)

    start_contacts = set(start.get("contacts", []))
    target_contacts = set(target.get("contacts", []))
    for key in sorted(start_contacts ^ target_contacts):
        groups["contact"].append(
            {
                "kind": "membership",
                "namespace": "contacts",
                "key": key,
                "target": key in target_contacts,
            }
        )

    for values in groups.values():
        values.sort(key=_clause_strength, reverse=True)
    return groups


def _clause_strength(clause: dict[str, Any]) -> float:
    if clause["kind"] == "pose_target":
        return float(clause.get("observed_position_delta", 0.0)) + 0.05 * float(
            clause.get("observed_rotation_delta", 0.0)
        )
    if clause["kind"] in {"vector_target", "scalar_target", "distance_target"}:
        return float(clause.get("observed_delta", 1.0))
    if clause["kind"] == "membership":
        return 1.0
    return 0.0


def _alternative(name: str, clauses: list[dict[str, Any]], strength: str = "primary") -> dict[str, Any]:
    return {"name": name, "strength": strength, "all": clauses}


def _top(groups: dict[str, list[dict[str, Any]]], name: str) -> dict[str, Any] | None:
    values = groups.get(name, [])
    return values[0] if values else None


def _expected_object_count(text: str) -> int:
    return 2 if re.search(r"\bboth\b|\btogether\b|at the same time", text, re.IGNORECASE) else 1


def _top_entities(groups: dict[str, list[dict[str, Any]]], count: int) -> list[dict[str, Any]]:
    primary = groups["task_entity"] or groups["actor"]
    if not primary:
        return []
    strongest = _clause_strength(primary[0])
    significant = [
        clause
        for clause in primary
        if _clause_strength(clause) >= max(0.01, strongest * 0.3)
    ]
    return significant[: max(1, count)]


def _pick_builder(
    groups: dict[str, list[dict[str, Any]]], expected_objects: int = 1
) -> list[dict[str, Any]]:
    entities = _top_entities(groups, expected_objects)
    if expected_objects > 1 and len(entities) < expected_objects:
        return []
    contact = next((c for c in groups["contact"] if c["target"] is True), None)
    if entities:
        clauses = list(entities)
        clauses.extend(groups["gripper"][:expected_objects])
        return [_alternative("object_lift_or_pick", clauses)]
    if contact:
        clauses = [contact]
        if groups["gripper"]:
            clauses.append(groups["gripper"][0])
        return [_alternative("grasp_contact", clauses)]
    if groups["gripper"]:
        return [_alternative("gripper_grasp", [groups["gripper"][0]], "secondary")]
    return []


def _place_builder(
    groups: dict[str, list[dict[str, Any]]], expected_objects: int = 1
) -> list[dict[str, Any]]:
    entities = _top_entities(groups, expected_objects)
    if expected_objects > 1 and len(entities) < expected_objects:
        return []
    if entities:
        clauses = list(entities)
        moved_keys = {clause["key"] for clause in entities}
        relation = next(
            (
                clause
                for clause in groups["relation"]
                if moved_keys.intersection(clause["keys"])
            ),
            None,
        )
        if relation:
            clauses.append(relation)
        clauses.extend(groups["gripper"][:expected_objects])
        return [_alternative("object_at_target", clauses)]
    return []


def _handover_builder(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    gained = [c for c in groups["contact"] if c["target"] is True]
    if gained:
        clauses = [gained[0]]
        if groups["gripper"]:
            clauses.append(groups["gripper"][0])
        return [_alternative("grasp_ownership_change", clauses)]
    return _place_builder(groups)


def _phone_grasp_builder(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Recognise the stable grasp boundary before phone insertion.

    RoboTwin exposes the first ``place_phone_stand`` expert boundary after the
    phone has been grasped.  Depending on the sampled phone pose, that boundary
    either contains a measurable phone-pose delta or only the persistent
    phone/gripper contact and closed-gripper transition.  Treat both as the
    same semantic phase instead of falling back to an end-effector waypoint.
    """
    phone_entity = next(
        (
            clause
            for bucket in ("task_entity", "actor")
            for clause in groups[bucket]
            if "phone" in str(clause.get("key", "")).lower()
        ),
        None,
    )
    gained_contacts = [
        clause
        for clause in groups["contact"]
        if clause.get("target") is True and "gripper:" in str(clause.get("key", ""))
    ]
    phone_contact = next(
        (
            clause
            for clause in gained_contacts
            if "phone" in str(clause.get("key", "")).lower()
        ),
        None,
    )
    contact = phone_contact or (gained_contacts[0] if gained_contacts else None)

    gripper = None
    if contact:
        contact_key = str(contact.get("key", ""))
        gripper = next(
            (
                clause
                for clause in groups["gripper"]
                if f"gripper:{clause.get('key')}" in contact_key
            ),
            None,
        )
    gripper = gripper or _top(groups, "gripper")

    if phone_entity:
        clauses = [phone_entity]
        if gripper:
            clauses.append(gripper)
        return [_alternative("phone_grasped_for_insertion", clauses)]
    if contact:
        clauses = [contact]
        if gripper:
            clauses.append(gripper)
        return [_alternative("phone_grasped_for_insertion", clauses)]
    return []


def _articulation_builder(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    clause = _top(groups, "articulation") or _top(groups, "task_scalar")
    if clause:
        return [_alternative("articulation_or_trigger_state", [clause])]
    entity = _top(groups, "task_entity") or _top(groups, "actor")
    return [_alternative("articulated_link_pose", [entity], "secondary")] if entity else []


def _press_builder(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    clause = _top(groups, "task_scalar") or _top(groups, "articulation")
    if clause:
        return [_alternative("trigger_state", [clause])]
    contact = next((c for c in groups["contact"] if c["target"] is True), None)
    return [_alternative("press_contact", [contact], "secondary")] if contact else []


def _interaction_builder(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    entity = _top(groups, "task_entity") or _top(groups, "actor")
    if entity:
        return [_alternative("object_interaction_state", [entity])]
    scalar = _top(groups, "task_scalar") or _top(groups, "articulation")
    return [_alternative("interaction_trigger", [scalar])] if scalar else []


PredicateBuilder = Callable[[dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]


PREDICATE_BUILDERS: dict[str, PredicateBuilder] = {
    # Registered stage-success tasks are handled before fitted-state candidate generation.
    "stage_success": _interaction_builder,
    "pick": _pick_builder,
    "place": _place_builder,
    "orient": _place_builder,
    "transport": _place_builder,
    "handover": _handover_builder,
    "phone_grasp": _phone_grasp_builder,
    "articulation": _articulation_builder,
    "press": _press_builder,
    "shake": _interaction_builder,
    "scan": _interaction_builder,
    "dump": _interaction_builder,
    "generic": _interaction_builder,
    # Built explicitly in ``build_predicate_contract`` from a monotonic task
    # scalar rather than inferred expert poses.
    "monotonic_count": _interaction_builder,
}


def _fallback(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    # A guarded end-effector target is the last resort for semantic stages with
    # no persistent object change (e.g. a handoff waypoint).  Pair it with a
    # gripper/contact condition whenever the replay exposes one.
    ee = _top(groups, "end_effector")
    if ee:
        clauses = [ee]
        guard = _top(groups, "gripper")
        if guard:
            clauses.append(guard)
        return [_alternative("guarded_end_effector_target", clauses, "fallback")]
    for bucket in ("task_entity", "actor", "articulation", "task_scalar", "contact", "gripper"):
        clause = _top(groups, bucket)
        if clause:
            return [_alternative(f"fallback_{bucket}", [clause], "fallback")]
    return []


def build_predicate_contract(
    start: dict[str, Any],
    target: dict[str, Any],
    *,
    task_name: str,
    phase: int,
    subtask_text: str,
    predicate_type: str | None = None,
    terminal_phase: bool = False,
) -> dict[str, Any]:
    if str(task_name) in SUBGOAL_TASKS:
        scalar_key = stage_scalar_key(str(task_name), int(phase))
        expected_objects = (
            int(phase) + 1
            if str(task_name)
            in {
                "place_bread_basket",
                "place_cans_plasticbox",
                "place_dual_shoes",
                "put_bottles_dustbin",
                "blocks_ranking_rgb",
                "blocks_ranking_size",
                "stack_blocks_two",
                "stack_blocks_three",
                "stack_bowls_three",
            }
            else 1
        )
        alternatives = [
            _alternative(
                "authoritative_stage_success",
                [
                    {
                        "kind": "scalar_target",
                        "namespace": "task_scalars",
                        "key": scalar_key,
                        "target": True,
                        "tolerance": 0.0,
                    }
                ],
            )
        ]
        contract = {
            "contract_version": PREDICATE_CONTRACT_VERSION,
            "stage_contract_version": STAGE_CONTRACT_VERSION,
            "predicate_type": "stage_success",
            "task_name": str(task_name),
            "phase": int(phase),
            "subtask_text": str(subtask_text),
            "terminal_phase": bool(terminal_phase),
            "expected_object_count": expected_objects,
            "alternatives": alternatives,
            "required_components": 1,
            "quality": "semantic",
            "candidate_counts": {"stage_success": 1},
        }
        if not terminal_phase and state_matches_predicate(start, contract):
            contract["alternatives"] = []
            contract["required_components"] = 0
            contract["quality"] = "non_discriminative"
        return contract

    resolved = predicate_type or resolve_predicate_type(task_name, phase, subtask_text)
    if resolved not in PREDICATE_BUILDERS:
        raise ValueError(f"unknown oracle predicate type {resolved!r}")
    groups = _candidate_groups(start, target)
    expected_objects = _expected_object_count(subtask_text)
    target_success = target.get("task_scalars", {}).get(OFFICIAL_TASK_SUCCESS_KEY)
    count_contract = MONOTONIC_COUNT_THRESHOLDS.get((str(task_name), int(phase)))
    if resolved == "monotonic_count" and count_contract is not None:
        # Keep the diagnostic/audit field aligned with the authoritative
        # threshold instead of relying on a language-number parser.
        expected_objects = int(count_contract[1])
    if terminal_phase and target_success is True:
        # A terminal phase is not used to switch to another goal.  Prefer the
        # task's official success predicate even when it was already true at
        # phase start (for example, RoboTwin's shake_bottle declares success
        # once the bottle is held above a height threshold).
        alternatives = [
            _alternative(
                "official_task_success",
                [
                    {
                        "kind": "scalar_target",
                        "namespace": "task_scalars",
                        "key": OFFICIAL_TASK_SUCCESS_KEY,
                        "target": True,
                        "tolerance": 0.0,
                    }
                ],
            )
        ]
    elif resolved == "monotonic_count":
        if count_contract is None:
            raise ValueError(
                f"no monotonic count contract for {task_name!r} phase {phase}"
            )
        scalar_key, threshold = count_contract
        alternatives = [
            _alternative(
                "monotonic_task_count",
                [
                    {
                        "kind": "scalar_at_least",
                        "namespace": "task_scalars",
                        "key": scalar_key,
                        "threshold": int(threshold),
                    }
                ],
            )
        ]
    elif resolved == "pick":
        alternatives = _pick_builder(groups, expected_objects)
    elif resolved in {"place", "orient", "transport"}:
        alternatives = _place_builder(groups, expected_objects)
    else:
        alternatives = PREDICATE_BUILDERS[resolved](groups)
    if not alternatives:
        alternatives = _fallback(groups)
    components = sum(len(group["all"]) for group in alternatives)
    contract = {
        "contract_version": PREDICATE_CONTRACT_VERSION,
        "predicate_type": resolved,
        "task_name": str(task_name),
        "phase": int(phase),
        "subtask_text": str(subtask_text),
        "terminal_phase": bool(terminal_phase),
        "expected_object_count": expected_objects,
        "alternatives": alternatives,
        # Kept for compatibility with the v1 caller and diagnostics.
        "required_components": components,
        "quality": "fallback" if any(group["strength"] == "fallback" for group in alternatives) else "semantic",
        "candidate_counts": {name: len(values) for name, values in groups.items()},
    }
    if components and not terminal_phase and state_matches_predicate(start, contract):
        # A phase predicate that already holds at phase start would switch early.
        contract["alternatives"] = []
        contract["required_components"] = 0
        contract["quality"] = "non_discriminative"
    return contract


def _lookup(state: dict[str, Any], namespace: str, key: str) -> tuple[bool, Any]:
    values = state.get(namespace, {})
    if namespace == "contacts":
        return True, key in set(state.get("contacts", []))
    if key not in values:
        return False, None
    return True, values[key]


def clause_matches_state(state: dict[str, Any], clause: dict[str, Any]) -> bool:
    if clause["kind"] == "distance_target":
        values = state.get(str(clause["namespace"]), {})
        left_key, right_key = clause["keys"]
        if left_key not in values or right_key not in values:
            return False
        actual = float(
            np.linalg.norm(
                np.asarray(values[left_key]["p"], dtype=np.float64)
                - np.asarray(values[right_key]["p"], dtype=np.float64)
            )
        )
        return abs(actual - float(clause["target"])) <= float(clause["tolerance"])
    exists, actual = _lookup(state, str(clause["namespace"]), str(clause["key"]))
    if not exists:
        return False
    kind = clause["kind"]
    if kind == "scalar_at_least":
        try:
            return float(actual) >= float(clause["threshold"])
        except (TypeError, ValueError):
            return False
    if kind == "membership":
        return bool(actual) is bool(clause["target"])
    if kind == "pose_target":
        target = clause["target"]
        if clause.get("check_position") and np.linalg.norm(
            np.asarray(actual["p"], dtype=np.float64) - np.asarray(target["p"], dtype=np.float64)
        ) > float(clause["position_tolerance"]):
            return False
        if clause.get("check_rotation") and _quat_distance(actual["q"], target["q"]) > float(
            clause["rotation_tolerance"]
        ):
            return False
        return True
    if kind == "vector_target":
        lhs = np.asarray(actual, dtype=np.float64).reshape(-1)
        rhs = np.asarray(clause["target"], dtype=np.float64).reshape(-1)
        return bool(lhs.shape == rhs.shape and lhs.size and np.max(np.abs(lhs - rhs)) <= float(clause["tolerance"]))
    if kind == "scalar_target":
        target = clause["target"]
        if isinstance(target, bool) or isinstance(target, str):
            return actual == target
        try:
            return abs(float(actual) - float(target)) <= float(clause["tolerance"])
        except (TypeError, ValueError):
            return actual == target
    raise ValueError(f"unknown oracle predicate clause kind {kind!r}")


def state_matches_predicate(state: dict[str, Any], contract: dict[str, Any]) -> bool:
    if int(contract.get("required_components", 0)) <= 0:
        return False
    alternatives = contract.get("alternatives", [])
    return any(group.get("all") and all(clause_matches_state(state, clause) for clause in group["all"]) for group in alternatives)


def registry_description() -> dict[str, Any]:
    return {
        "contract_version": PREDICATE_CONTRACT_VERSION,
        "types": sorted(PREDICATE_BUILDERS),
        "rules": [{"name": rule.name, "patterns": list(rule.patterns)} for rule in PREDICATE_RULES],
        "task_phase_overrides": {
            f"{task}:{phase}": predicate_type
            for (task, phase), predicate_type in sorted(TASK_PHASE_OVERRIDES.items())
        },
    }
