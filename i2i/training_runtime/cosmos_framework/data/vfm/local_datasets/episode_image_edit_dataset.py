# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Episode image-editing dataset for Cosmos3 VFM SFT.

Each sample is:

    text instruction + an observation frame -> a later target frame from the same episode

The target is the episode's final frame by default (``target_mode="episode_final"``), or a
future subgoal frame selected from the annotated subtask segments (``target_mode="segment_final"``,
LeRobot only). Most frames target their current segment's final frame; by default, frames in the
last 15% of a non-final segment target the next segment's final frame. Tail frames in the final
segment keep its own final frame as their goal. The dataset enumerates every eligible observation
frame in every episode, so an epoch covers the original frame set instead of a sampled subset.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import random
import re
import subprocess
import tempfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from cosmos_framework.data.vfm.sequence_packing import SequencePlan
from cosmos_framework.data.vfm.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.model.vfm.vlm.qwen3_vl.utils import tokenize_caption
from cosmos_framework.utils import log
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate

_EPISODE_RE = re.compile(r"episode(\d+)$")
_SYSTEM_PROMPT_IMAGE_EDITING = "You are a helpful assistant who will edit images based on the user's instructions."
_DEFAULT_NEXT_SUBGOAL_TAIL_FRACTION = 0.15
# Legacy aliases for the AgiBot layout. RoboTwin uses the profile selected from
# ``lerobot_video_key`` below.
_THREE_CAMERA_TARGET_HW = (320, 384)
_THREE_CAMERA_CONCAT_VIDEO_KEY = "observation.images.concat_view_320x384"
_THREE_CAMERA_CONCAT_SPECS = {
    "observation.images.head_front_color": (_THREE_CAMERA_CONCAT_VIDEO_KEY, _THREE_CAMERA_TARGET_HW, "4,3"),
    "observation.images.cam_high": (
        "observation.images.concat_view_384x320",
        (384, 320),
        "3,4",
    ),
}

# Bounded per-frame raw-frame decode cache (see _decode_frame). Module-level so it is shared
# across all dataset instances within a worker process. Keyed by (video_path, frame_idx,
# target_height, target_width). This mainly pays off for ``segment_final`` targets: the leading
# portion of a subtask segment shares its current final target and the configurable tail shares the
# next segment final, so each target is decoded once and reused instead of re-decoded per sample. Bounded by an
# LRU cap so cold input frames (each seen once per epoch) get evicted while hot targets stay warm.
# Cap comes from EPISODE_DECODE_CACHE_MAX (frames); EPISODE_DECODE_CACHE=1 selects a default cap.
# 0 => disabled (production default unchanged).
_DECODE_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()


def _decode_cache_max() -> int:
    raw = os.environ.get("EPISODE_DECODE_CACHE_MAX")
    if raw is not None:
        try:
            return max(0, int(raw))
        except ValueError:
            return 0
    return 1024 if os.environ.get("EPISODE_DECODE_CACHE") == "1" else 0


def _as_bool(value: Any) -> bool:
    """Coerce a config/env value to bool. Accepts real bools/ints and env strings.

    ``oc.env`` interpolations arrive as strings (e.g. ``"false"``), for which ``bool("false")``
    would wrongly be ``True``; parse common truthy tokens explicitly instead.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class _EpisodeCandidate:
    """One discovered episode before video metadata (dims/frames) is read.

    Layout-specific discovery produces these; the shared metadata pass turns
    them into :class:`_Episode` instances.
    """

    episode_id: int
    video_path: Path
    instructions: tuple[str, ...]
    instruction_path: Path
    task_name: str = ""
    meta_length: int | None = None  # authoritative frame count when the layout provides one (LeRobot)
    video_paths: tuple[Path, ...] = ()
    metadata_hints: tuple[dict[str, int | float], ...] = ()
    # (start_frame, end_frame_exclusive, is_mistake, action_text, cot) subtask segments, sorted by start.
    # Populated only for LeRobot layout when target_mode == "segment_final"; empty otherwise.
    segments: tuple[tuple[int, int, bool, str, str], ...] = ()


@dataclass(frozen=True)
class _Episode:
    episode_id: int
    video_path: Path
    instruction_path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    instructions: tuple[str, ...]
    aspect_ratio: str
    target_width: int
    target_height: int
    video_paths: tuple[Path, ...] = ()
    video_metadata: tuple[dict[str, int | float], ...] = ()
    task_name: str = ""
    # Subtask segments clamped to total_frames (segment_final mode; empty otherwise) as
    # (start, end_exclusive, is_mistake, action_text, cot), and the sorted exclusive per-segment end
    # frames used for the bisect target lookup.
    segments: tuple[tuple[int, int, bool, str, str], ...] = ()
    segment_ends: tuple[int, ...] = ()


def _episode_sort_key(path: Path) -> tuple[int, str]:
    match = _EPISODE_RE.fullmatch(path.stem)
    if match is None:
        return (10**12, path.stem)
    return (int(match.group(1)), path.stem)


def _episode_id(path: Path) -> int:
    match = _EPISODE_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Expected episode<N> filename, got {path.name!r}")
    return int(match.group(1))


def _load_instruction_set(path: Path, instruction_keys: tuple[str, ...] | None) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(payload).__name__}")

    candidates: list[str] = []
    keys = instruction_keys or tuple(payload.keys())
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                candidates.append(text)
        elif isinstance(value, list):
            candidates.extend(item.strip() for item in value if isinstance(item, str) and item.strip())

    if not candidates:
        selected = ", ".join(instruction_keys) if instruction_keys else "any string/list-valued key"
        raise ValueError(f"{path} does not contain instructions under {selected}")

    return tuple(candidates)


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines file into a list of objects, skipping blank lines."""

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def _load_lerobot_instructions(meta_dir: Path) -> dict[int, tuple[tuple[str, ...], int | None]]:
    """Map episode_index -> (instruction strings, frame length) from a LeRobot ``meta/`` dir.

    Prefers the per-episode ``tasks`` list in ``episodes.jsonl``. Falls back to
    resolving ``task_index`` against ``tasks.jsonl`` for datasets that store the
    index instead of the text. The ``length`` field (authoritative frame count)
    is returned alongside when present, else ``None``.
    """

    episodes_path = meta_dir / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"LeRobot dataset missing {episodes_path}")
    episode_rows = _read_jsonl(episodes_path)

    task_by_index: dict[int, str] = {}
    tasks_path = meta_dir / "tasks.jsonl"
    if tasks_path.is_file():
        for row in _read_jsonl(tasks_path):
            if "task_index" in row and isinstance(row.get("task"), str):
                text = row["task"].strip()
                if text:
                    task_by_index[int(row["task_index"])] = text

    instructions: dict[int, tuple[tuple[str, ...], int | None]] = {}
    for row in episode_rows:
        if "episode_index" not in row:
            raise ValueError(f"{episodes_path} row missing 'episode_index': {row}")
        eid = int(row["episode_index"])
        texts: list[str] = []
        raw_tasks = row.get("tasks")
        if isinstance(raw_tasks, str):
            raw_tasks = [raw_tasks]
        if isinstance(raw_tasks, list):
            texts.extend(item.strip() for item in raw_tasks if isinstance(item, str) and item.strip())
        if not texts:
            raw_index = row.get("task_index")
            indices = raw_index if isinstance(raw_index, list) else [raw_index]
            for idx in indices:
                if idx is None:
                    continue
                resolved = task_by_index.get(int(idx))
                if resolved:
                    texts.append(resolved)
        if not texts:
            raise ValueError(f"{episodes_path}: episode {eid} has no usable task/instruction text")
        raw_length = row.get("length")
        length = int(raw_length) if isinstance(raw_length, (int, float)) and int(raw_length) > 0 else None
        instructions[eid] = (tuple(dict.fromkeys(texts)), length)  # de-dupe, preserve order
    if not instructions:
        raise ValueError(f"{episodes_path} contained no episodes")
    return instructions


def _task_names_from_annotations(meta_dir: Path) -> dict[int, str]:
    """Return fallback per-episode task labels from ``annotations.json`` when available.

    The episode-level task in ``episodes.jsonl`` is authoritative for model conditioning and
    evaluation grouping. Annotation labels are retained only for legacy datasets whose episode
    metadata cannot provide a canonical instruction.
    """

    path = meta_dir / "annotations.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by episode index")

    task_names: dict[int, str] = {}
    for key, value in data.items():
        try:
            episode_id = int(key)
        except (TypeError, ValueError):
            continue
        raw_name = value.get("task_name") if isinstance(value, dict) else None
        if isinstance(raw_name, str) and raw_name.strip():
            task_names[episode_id] = raw_name.strip()
    return task_names


def _segments_from_action_annotations_file(
    path: Path,
) -> dict[int, tuple[tuple[int, int, bool, str, str], ...]]:
    """Load legacy ``action_steps`` annotations keyed by ``str(episode_index)``.

    Each value carries ``action_steps = [{start_frame, end_frame, is_mistake, action_text, ...}]``
    in the LeRobot video frame space (``end_frame`` exclusive). The ``episode_index`` field
    *inside* each value is unreliable, so the dict key is authoritative. ``action_text`` (the
    natural-language subtask instruction, e.g. "Pick up bun") is carried through so
    ``segment_final`` can condition on the current subgoal rather than the constant episode-level
    task text. A checked ``info.cot`` is carried separately for reasoner supervision. Returns
    ``{}`` if the file is absent so ``auto`` can fall back to ``instruction_segments``.
    """

    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by episode index")

    out: dict[int, tuple[tuple[int, int, bool, str, str], ...]] = {}
    for key, value in data.items():
        try:
            eid = int(key)
        except (TypeError, ValueError):
            continue
        steps = value.get("action_steps") if isinstance(value, dict) else None
        if not isinstance(steps, list):
            continue
        segs: list[tuple[int, int, bool, str, str]] = []
        for step in steps:
            if not isinstance(step, dict) or "start_frame" not in step or "end_frame" not in step:
                continue
            start, end = int(step["start_frame"]), int(step["end_frame"])
            if end > start:
                raw_text = step.get("action_text")
                text = raw_text.strip() if isinstance(raw_text, str) else ""
                info = step.get("info") if isinstance(step.get("info"), dict) else {}
                raw_cot = info.get("cot") if isinstance(info, dict) else ""
                cot_check = info.get("cot_check") if isinstance(info, dict) else None
                checked = (
                    isinstance(cot_check, dict)
                    and cot_check.get("verdict") == "PASS"
                    and cot_check.get("alignment_verdict") == "PASS"
                )
                cot = raw_cot.strip() if checked and isinstance(raw_cot, str) else ""
                segs.append((start, end, bool(step.get("is_mistake", False)), text, cot))
        if segs:
            out[eid] = tuple(sorted(segs))
    return out


def _segments_from_annotations(meta_dir: Path) -> dict[int, tuple[tuple[int, int, bool, str, str], ...]]:
    """Subtask segments from ``meta/annotations.json``.

    Accept both the legacy episode-keyed ``action_steps`` schema and the fixed-goal contract's
    ``{"episodes": {episode_index: {"segments": [...]}}}`` schema.
    """

    path = meta_dir / "annotations.json"
    legacy = _segments_from_action_annotations_file(path)
    if legacy or not path.is_file():
        return legacy

    data = json.loads(path.read_text(encoding="utf-8"))
    episodes = data.get("episodes") if isinstance(data, dict) else None
    if not isinstance(episodes, dict):
        return {}

    out: dict[int, tuple[tuple[int, int, bool, str, str], ...]] = {}
    for key, value in episodes.items():
        try:
            eid = int(key)
        except (TypeError, ValueError):
            continue
        raw_segments = value.get("segments") if isinstance(value, dict) else None
        if not isinstance(raw_segments, list):
            continue
        segments: list[tuple[int, int, bool, str, str]] = []
        for segment in raw_segments:
            if (
                not isinstance(segment, dict)
                or "start_frame" not in segment
                or "end_frame_exclusive" not in segment
            ):
                continue
            start = int(segment["start_frame"])
            end = int(segment["end_frame_exclusive"])
            if end <= start:
                continue
            raw_text = segment.get("stage_text") or segment.get("task_instruction")
            text = raw_text.strip() if isinstance(raw_text, str) else ""
            segments.append((start, end, bool(segment.get("is_mistake", False)), text, ""))
        if segments:
            out[eid] = tuple(sorted(segments))
    return out


def _segments_from_legacy_action_annotations(
    meta_dir: Path,
) -> dict[int, tuple[tuple[int, int, bool, str, str], ...]]:
    """Subtask segments archived as ``meta/legacy_action_annotations.json``."""

    return _segments_from_action_annotations_file(meta_dir / "legacy_action_annotations.json")


def _segments_from_instruction_segments(meta_dir: Path) -> dict[int, tuple[tuple[int, int, bool, str, str], ...]]:
    """Subtask segments from ``meta/info.json['instruction_segments']`` keyed by ``str(episode_index)``.

    Uses ``start_frame_index/end_frame_index`` (same frame space as ``action_steps``); this source
    has no mistake flag, so every segment is treated as non-mistake. It carries the subtask text
    when present under ``instruction``/``text``/``action_text`` (else an empty string, which makes
    ``segment_final`` fall back to the episode-level task). Returns ``{}`` if absent.
    """

    path = meta_dir / "info.json"
    if not path.is_file():
        return {}
    info = json.loads(path.read_text(encoding="utf-8"))
    seg_map = info.get("instruction_segments") if isinstance(info, dict) else None
    if not isinstance(seg_map, dict):
        return {}

    out: dict[int, tuple[tuple[int, int, bool, str, str], ...]] = {}
    for key, value in seg_map.items():
        try:
            eid = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, list):
            continue
        segs: list[tuple[int, int, bool, str, str]] = []
        for step in value:
            if not isinstance(step, dict) or "start_frame_index" not in step or "end_frame_index" not in step:
                continue
            start, end = int(step["start_frame_index"]), int(step["end_frame_index"])
            if end > start:
                raw_text = step.get("instruction") or step.get("text") or step.get("action_text")
                text = raw_text.strip() if isinstance(raw_text, str) else ""
                segs.append((start, end, False, text, ""))
        if segs:
            out[eid] = tuple(sorted(segs))
    return out


def _load_lerobot_segments(meta_dir: Path, source: str) -> dict[int, tuple[tuple[int, int, bool, str, str], ...]]:
    """Load per-episode subtask segments for ``segment_final`` targets.

    ``source`` selects the metadata file: ``"annotations"`` (``meta/annotations.json``),
    ``"legacy_action_annotations"`` (``meta/legacy_action_annotations.json``),
    ``"instruction_segments"`` (``meta/info.json``), or ``"auto"`` (those sources in that
    order). ``"episode_final"`` is constructed by
    :meth:`EpisodeImageEditDataset._discover_candidates_lerobot`, where per-episode lengths are
    available. Raises if the selected metadata source yields nothing usable.
    """

    src = source.lower()
    if src == "annotations":
        segments = _segments_from_annotations(meta_dir)
    elif src == "legacy_action_annotations":
        segments = _segments_from_legacy_action_annotations(meta_dir)
    elif src == "instruction_segments":
        segments = _segments_from_instruction_segments(meta_dir)
    elif src == "auto":
        segments = (
            _segments_from_annotations(meta_dir)
            or _segments_from_legacy_action_annotations(meta_dir)
            or _segments_from_instruction_segments(meta_dir)
        )
    else:
        raise ValueError(
            "lerobot_segment_source must be 'auto', 'annotations', 'legacy_action_annotations', "
            f"'instruction_segments', or 'episode_final', got {source!r}"
        )
    if not segments:
        raise FileNotFoundError(
            f"No subtask segmentation found under {meta_dir} for lerobot_segment_source={source!r}. "
            "Expected meta/annotations.json (fixed-goal segments or action_steps), "
            "meta/legacy_action_annotations.json (action_steps), "
            "or meta/info.json (instruction_segments)."
        )
    return segments


def _prepare_episode_segments(
    raw_segments: tuple[tuple[int, int, bool, str, str], ...],
    total_frames: int,
    episode_id: int,
) -> tuple[tuple[tuple[int, int, bool, str, str], ...], tuple[int, ...]]:
    """Sort, clamp to ``total_frames``, and validate one episode's subtask segments.

    Returns ``(segments, segment_ends)`` where ``segment_ends`` are the exclusive per-segment
    end frames used for the bisect target lookup. Each segment is ``(start, end, is_mistake,
    action_text, cot)``. Emits a warning (not an error) when the segments are not a contiguous
    ``[0, total_frames)`` cover. Frames outside the returned segment ranges are excluded when the
    dataset builds its segment-final sample index.
    """

    if not raw_segments:
        return (), ()
    clamped: list[tuple[int, int, bool, str, str]] = []
    for start, end, mistake, text, cot in sorted(raw_segments):
        if start >= total_frames:
            continue  # segment lies entirely past the decodable range
        clamped.append((start, min(end, total_frames), mistake, text, cot))
    if not clamped:
        log.warning(
            f"Episode {episode_id}: all subtask segments start past total_frames={total_frames}; "
            "falling back to episode-final target",
            rank0_only=True,
        )
        return (), ()

    prev_end = 0
    contiguous = clamped[0][0] == 0
    for start, end, _, _, _ in clamped:
        if start != prev_end:
            contiguous = False
        prev_end = end
    if not contiguous or prev_end != total_frames:
        log.warning(
            f"Episode {episode_id}: subtask segments are not a contiguous [0,{total_frames}) cover "
            f"(starts/ends {[(s, e) for s, e, _, _, _ in clamped]}); uncovered frames will be excluded",
            rank0_only=True,
        )
    return tuple(clamped), tuple(end for _, end, _, _, _ in clamped)


def _segment_index_for(segment_ends: tuple[int, ...], input_idx: int) -> int:
    """Index of the subtask segment containing ``input_idx``, or ``-1`` if there are no segments.

    ``segment_ends`` are sorted exclusive segment ends. The containing segment is the first whose
    end exceeds ``input_idx``; an ``input_idx`` at/after the last end (possible when an annotation
    end was clamped below the decodable count) maps to the last segment.
    """

    if not segment_ends:
        return -1
    k = bisect.bisect_right(segment_ends, input_idx)
    if k >= len(segment_ends):
        k = len(segment_ends) - 1
    return k


def _subgoal_target_segment_index(
    segments: tuple[tuple[int, int, bool, str, str], ...],
    input_idx: int,
    next_subgoal_tail_fraction: float = _DEFAULT_NEXT_SUBGOAL_TAIL_FRACTION,
) -> int | None:
    """Return the segment whose final frame is the future subgoal for ``input_idx``.

    Inputs in the configurable tail fraction (15% by default, rounded up to whole frames) of a
    non-final segment target the next segment. Tail frames in the final segment continue to target
    that segment's final frame. A zero fraction disables next-segment redirection.
    """

    if not 0.0 <= next_subgoal_tail_fraction <= 1.0:
        raise ValueError(f"next_subgoal_tail_fraction must be in [0, 1], got {next_subgoal_tail_fraction!r}")

    for segment_index, (start, end, _, _, _) in enumerate(segments):
        if not start <= input_idx < end:
            continue
        segment_length = end - start
        tail_frames = math.ceil(segment_length * next_subgoal_tail_fraction)
        if tail_frames > 0 and input_idx >= end - tail_frames:
            next_segment = segment_index + 1
            return next_segment if next_segment < len(segments) else segment_index
        return segment_index
    return None


def _segment_target_index(
    segments: tuple[tuple[int, int, bool, str, str], ...],
    input_idx: int,
    total_frames: int,
    next_subgoal_tail_fraction: float = _DEFAULT_NEXT_SUBGOAL_TAIL_FRACTION,
) -> tuple[int, int] | None:
    """Return ``(target_segment_index, final_frame_index)`` for a subgoal sample."""

    target_segment_index = _subgoal_target_segment_index(segments, input_idx, next_subgoal_tail_fraction)
    if target_segment_index is None:
        return None
    target_end = segments[target_segment_index][1]
    return target_segment_index, min(target_end - 1, total_frames - 1)


def _frame_in_ranges(frame_idx: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    """True if ``frame_idx`` falls in any ``[start, end)`` range (used to skip mistake frames)."""

    return any(start <= frame_idx < end for start, end in ranges)


def _get_video_metadata(video_path: Path) -> dict[str, int | float]:
    """Read width, height, fps, and frame count without importing S3 helpers."""

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        "v:0",
        str(video_path),
    ]
    result = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, check=True, text=True)
    probe_data = json.loads(result.stdout)
    if not probe_data.get("streams"):
        raise ValueError(f"No video stream found in {video_path}")

    stream = probe_data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps_parts = stream["r_frame_rate"].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1])
    if "nb_frames" in stream:
        total_frames = int(stream["nb_frames"])
    else:
        duration = float(stream.get("duration") or 0)
        total_frames = int(duration * fps)

    return {"width": width, "height": height, "fps": fps, "total_frames": total_frames}


def _metadata_cache_path(dataset_dir: Path) -> Path:
    return dataset_dir / ".cache" / "episode_image_edit_metadata.json"


def _load_metadata_cache(path: Path) -> dict[str, dict[str, int | float]]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        log.warning(f"Ignoring unreadable episode metadata cache {path}: {e}", rank0_only=True)
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): v for k, v in payload.items() if isinstance(v, dict)}


def _write_metadata_cache(path: Path, payload: dict[str, dict[str, int | float]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
            json.dump(payload, f, sort_keys=True)
            tmp_path = Path(f.name)
        tmp_path.replace(path)
    except Exception as e:
        log.warning(f"Failed to write episode metadata cache {path}: {e}", rank0_only=True)


def _get_aspect_ratio(width: int, height: int) -> str:
    ratio = width / height
    if ratio < 0.65:
        return "9,16"
    if ratio < 0.88:
        return "3,4"
    if ratio < 1.16:
        return "1,1"
    if ratio < 1.55:
        return "4,3"
    return "16,9"


def _resolve_target_size(width: int, height: int, resize_resolution: str | None) -> tuple[str, int, int]:
    aspect_ratio = _get_aspect_ratio(width, height)
    if resize_resolution is None:
        return aspect_ratio, width, height
    if resize_resolution not in VIDEO_RES_SIZE_INFO:
        raise ValueError(f"Unknown resize_resolution {resize_resolution!r}; available: {sorted(VIDEO_RES_SIZE_INFO)}")
    target_width, target_height = VIDEO_RES_SIZE_INFO[resize_resolution][aspect_ratio]
    return aspect_ratio, int(target_width), int(target_height)


def _default_three_camera_video_keys(video_key: str) -> tuple[str, str, str]:
    if video_key == "observation.images.cam_high":
        return (
            video_key,
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        )
    if video_key == "observation.images.head_front_color":
        return (
            video_key,
            "observation.images.hand_left",
            "observation.images.hand_right",
        )
    raise ValueError(
        "Three-camera mode needs a known head camera. Set three_camera_video_keys to three comma-separated "
        f"features; got primary camera {video_key!r}."
    )


def _three_camera_concat_spec(video_key: str) -> tuple[str, tuple[int, int], str]:
    try:
        return _THREE_CAMERA_CONCAT_SPECS[video_key]
    except KeyError as error:
        supported = sorted(_THREE_CAMERA_CONCAT_SPECS)
        raise ValueError(
            f"Three-camera offline preprocessing has no profile for primary camera {video_key!r}; supported={supported}"
        ) from error


def _normalize_three_camera_video_keys(
    video_key: str,
    value: str | tuple[str, ...] | list[str] | None,
) -> tuple[str, str, str]:
    if value is None:
        return _default_three_camera_video_keys(video_key)
    if isinstance(value, str):
        keys = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        keys = tuple(str(item).strip() for item in value if str(item).strip())
    if len(keys) != 3:
        raise ValueError(f"three_camera_video_keys must contain exactly three features, got {keys!r}")
    if keys[0] != video_key:
        raise ValueError(f"three_camera_video_keys[0] must equal lerobot_video_key={video_key!r}, got {keys[0]!r}")
    return keys[0], keys[1], keys[2]


def _lerobot_feature_metadata(
    info: dict[str, Any],
    episode: dict[str, Any],
    video_key: str,
) -> dict[str, int | float] | None:
    feature = info.get("features", {}).get(video_key, {})
    # Offline concat streams may be shorter than the tabular episode length when a source
    # camera ends early. Probe unless preprocessing explicitly verified that every output kept
    # the authoritative episodes.jsonl length.
    if video_key.startswith("observation.images.concat_view_") and not feature.get(
        "episode_lengths_match_metadata", False
    ):
        return None
    shape = feature.get("shape") if isinstance(feature, dict) else None
    video_info = feature.get("video_info", {}) if isinstance(feature, dict) else {}
    fps = video_info.get("video.fps") if isinstance(video_info, dict) else None
    fps = fps or info.get("fps")
    length = episode.get("length")
    if not isinstance(shape, list) or len(shape) < 2 or fps is None or length is None:
        return None
    height, width = int(shape[0]), int(shape[1])
    fps_value, total_frames = float(fps), int(length)
    if width < 1 or height < 1 or fps_value <= 0 or total_frames < 1:
        return None
    return {"width": width, "height": height, "fps": fps_value, "total_frames": total_frames}


def _decode_frame(
    video_path: Path,
    frame_idx: int,
    fps: float,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Decode exactly one RGB frame via a fast, frame-accurate seek. Returns HWC uint8.

    ffmpeg's ``select=eq(n,X)`` filter decodes the *entire* video from frame 0 (and, without a
    frame limit, keeps reading to EOF) just to emit frame X. With every frame of every episode
    enumerated as a training sample, that is an O(total_frames**2) linear decode per episode per
    epoch and is the dominant cost of this dataset. Instead, put ``-ss`` *before* ``-i`` so ffmpeg
    seeks to the keyframe at/just before the target timestamp and decodes forward only ~one GOP,
    then emit the single frame at that timestamp. Verified bit-exact against the full-decode path
    for CFR 30fps HEVC (the dataset's format); the half-frame seek bias avoids ever rounding up
    onto ``frame_idx + 1``.
    """

    cache_max = _decode_cache_max()
    ck = (str(video_path), int(frame_idx), int(target_height), int(target_width))
    if cache_max > 0:
        hit = _DECODE_CACHE.get(ck)
        if hit is not None:
            _DECODE_CACHE.move_to_end(ck)
            return hit.copy()

    frame_size = target_height * target_width * 3
    # Seek half a frame early so the first frame with pts >= seek lands exactly on frame_idx
    # rather than being rounded onto the next frame.
    seek_ts = max(0.0, (frame_idx - 0.5) / fps) if fps > 0 else 0.0
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-ss",
        f"{seek_ts:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={target_width}:{target_height}:flags=bicubic+accurate_rnd",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    result = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, check=True)
    if len(result.stdout) != frame_size:
        raise RuntimeError(
            f"ffmpeg decoded {len(result.stdout)} bytes from {video_path}, expected {frame_size} "
            f"for frame {frame_idx} (fps={fps}, seek={seek_ts:.6f})"
        )
    frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(target_height, target_width, 3).copy()

    if cache_max > 0:
        _DECODE_CACHE[ck] = frame
        _DECODE_CACHE.move_to_end(ck)
        while len(_DECODE_CACHE) > cache_max:
            _DECODE_CACHE.popitem(last=False)
    return frame


def _decode_two_frames(
    video_path: Path,
    frame_idx: int,
    final_idx: int,
    fps: float,
    target_height: int,
    target_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the (input, target) RGB frame pair, resizing by scale only and never padding."""

    source = _decode_frame(video_path, frame_idx, fps, target_height, target_width)
    if final_idx == frame_idx:
        return source.copy(), source.copy()
    target = _decode_frame(video_path, final_idx, fps, target_height, target_width)
    return source, target


def _compose_three_camera_frame(
    head: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    target_hw: tuple[int, int] = _THREE_CAMERA_TARGET_HW,
) -> np.ndarray:
    """Compose head/left/right views using the canonical inverted-T layout."""

    head_h, head_w = head.shape[:2]
    half_h, half_w = head_h // 2, head_w // 2
    left_small = cv2.resize(left, (half_w, half_h), interpolation=cv2.INTER_LINEAR)
    right_small = cv2.resize(right, (half_w, half_h), interpolation=cv2.INTER_LINEAR)
    bottom = cv2.hconcat([left_small, right_small])
    canvas = cv2.vconcat([head, bottom])
    target_h, target_w = target_hw
    if canvas.shape[:2] != target_hw:
        canvas = cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return canvas


def _decode_three_camera_frames(
    video_paths: tuple[Path, Path, Path],
    video_metadata: tuple[dict[str, int | float], ...],
    frame_idx: int,
    final_idx: int,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode synchronized head/left/right frames and compose each into one image."""

    def decode(frame_idx_to_read: int) -> np.ndarray:
        frames = tuple(
            _decode_frame(
                path,
                frame_idx_to_read,
                float(metadata["fps"] or fps),
                int(metadata["height"]),
                int(metadata["width"]),
            )
            for path, metadata in zip(video_paths, video_metadata, strict=True)
        )
        return _compose_three_camera_frame(*frames)

    source = decode(frame_idx)
    if final_idx == frame_idx:
        return source.copy(), source.copy()
    return source, decode(final_idx)


def _to_chw_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert an RGB HWC uint8 frame to CHW uint8 without resizing or padding."""

    return torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).to(torch.uint8)


class EpisodeImageEditDataset(torch.utils.data.Dataset):
    """Map-style episode dataset for image-to-image SFT.

    Two on-disk layouts are supported, selected by ``dataset_format``:

    * ``"flat"`` — ``video/episode<N>.mp4`` plus ``instructions/episode<N>.json``
      (keyed lists of edit instructions).
    * ``"lerobot"`` — a LeRobot v2.1 dataset root. Videos come from the
      ``lerobot_video_key`` camera stream under
      ``videos/chunk-<C:03d>/<video_key>/episode_<N:06d>.mp4`` and the
      instruction for each episode is its ``tasks`` entry in
      ``meta/episodes.jsonl``. ``instruction_keys`` does not apply here.
    * ``"auto"`` (default) — picks ``"lerobot"`` when both ``meta/info.json``
      and ``meta/episodes.jsonl`` exist, otherwise ``"flat"``.

    The "first frame -> final frame" sampling and every emitted field are
    identical across layouts; only episode discovery and instruction loading
    differ. Subtask-segment targets (``target_mode="segment_final"``) apply to
    the LeRobot layout only.

    Args:
        dataset_dir: Dataset root.
        video_subdir: Flat layout: subdirectory with ``episode<N>.mp4`` files.
        instruction_subdir: Flat layout: subdirectory with ``episode<N>.json``.
        instruction_keys: Flat layout: JSON keys to sample instructions from.
            ``None`` uses all string or list-of-string fields.
        resize_resolution: Cosmos resolution bucket to resize to without padding.
            ``None`` keeps native frame size.
        include_final_as_input: Keep true for full frame coverage.
        first_frame_only: Emit just one sample per episode (frame 0 -> final
            frame). Use for a fixed 50-episode eval set instead of all frames.
        dataset_format: ``"auto"``, ``"flat"``, or ``"lerobot"``.
        lerobot_video_key: LeRobot camera feature whose video stream is used,
            e.g. ``"observation.images.cam_high"``.
        lerobot_meta_subdir: LeRobot metadata subdirectory (default ``"meta"``).
        target_mode: ``"episode_final"`` (default) targets each episode's last frame.
            ``"segment_final"`` targets the current segment's final frame, except that inputs in
            the last 15% of a non-final segment target the next segment's final frame. Tail frames
            in the final segment keep its own final frame as their goal. ``"segment_final"``
            requires the LeRobot layout.
        lerobot_segment_source: Where ``segment_final`` reads subtask segments — ``"auto"``
            (``meta/annotations.json``, then ``meta/legacy_action_annotations.json``, then
            ``meta/info.json['instruction_segments']``), ``"annotations"``,
            ``"legacy_action_annotations"``, ``"instruction_segments"``, ``"episode_final"``, or
            ``"auto_or_episode_final"``. The explicit ``"episode_final"`` source treats every
            unsegmented episode as one segment. ``"auto_or_episode_final"`` preserves available
            annotations and applies that one-segment fallback only to unannotated episodes. These
            options are useful for single-stage datasets whose task-completion frame is the
            subgoal reference; neither silently substitutes for missing annotations in
            ``"auto"`` mode.
        skip_mistake_segments: In ``segment_final`` mode, drop input frames whose containing
            segment is flagged ``is_mistake`` so the model never trains toward a bad subgoal.
        episodes_per_dataset: When set, deterministically select this many random episodes from
            this dataset root before building samples. Multi-root evaluation therefore selects the
            requested number independently for every task.
        episodes_per_task: When set, deterministically select this many episodes independently for
            each canonical task instruction in ``meta/episodes.jsonl``. This is intended for combined
            multi-task LeRobot roots whose outputs must remain grouped by task. Mutually exclusive
            with ``episodes_per_dataset``.
        max_episode_index: When set, exclude LeRobot episodes with a larger ``episode_index`` before
            bounded eval sampling. This supports evaluation against an earlier append-only snapshot
            of a combined dataset.
        frames_per_segment: In ``segment_final`` mode, emit this many random distinct input frames
            from each retained subtask segment. Takes precedence over ``frames_per_episode`` and
            is intended for subgoal evaluation, where every selected segment needs its own target.
        frames_per_episode: Eval sampling. When set, emit exactly this many random distinct input
            frames per episode (seeded per episode for reproducibility) instead of every frame.
            Takes precedence over ``first_frame_only``. ``None`` keeps the default all-frames /
            first-frame behavior.
        reasoner_subtask_target: When True (``segment_final`` only), the sample trains the reasoner
            to *predict* the target segment's checked CoT followed by the authoritative
            ``action_text``. For a source in a non-final segment's last 15%, this is the next
            segment; the final segment remains its own target. The text prompt fed to the model
            is the episode-level task **only**
            (never the subtask), and the target uses Cosmos's native
            ``<think>...</think>\\n{subtask}`` format after the assistant generation header. Frames
            from segments without both fields are excluded instead of receiving an action-only
            target. The returned ``num_prompt_tokens`` marks how many leading text tokens are
            prompt (unsupervised); the packer masks those from the CE loss. Leaves
            The subtask is a supervised output, never part of the conditioning prompt.
        reasoner_target_format: ``"cot_and_subtask"`` (default) requires checked CoT metadata and
            emits ``<think>...</think>\n{subtask}``. ``"subtask"`` emits the authoritative
            ``action_text`` alone for compatibility with checkpoints trained before CoT targets.
        next_subgoal_tail_fraction: Fraction of each non-final segment's trailing frames that target
            the next segment's final frame. Must be in ``[0, 1]``; defaults to ``0.15``. ``0``
            disables redirection. The final segment always targets its own final frame.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        video_subdir: str = "video",
        instruction_subdir: str = "instructions",
        video_glob: str = "episode*.mp4",
        instruction_keys: tuple[str, ...] | list[str] | None = None,
        seed: int = 42,
        tokenizer_config: Any | None = None,
        max_caption_tokens: int = 1024,
        cfg_dropout_rate: float = 0.0,
        resize_resolution: str | None = "256",
        include_final_as_input: bool = True,
        first_frame_only: bool = False,
        dataset_name: str = "episode_image_editing",
        metadata_num_workers: int = 16,
        dataset_format: str = "auto",
        lerobot_video_key: str = "observation.images.cam_high",
        use_three_camera: bool = True,
        three_camera_video_keys: str | tuple[str, ...] | list[str] | None = None,
        lerobot_meta_subdir: str = "meta",
        target_mode: str = "episode_final",
        lerobot_segment_source: str = "auto",
        skip_mistake_segments: bool = True,
        episodes_per_dataset: int | None = None,
        episodes_per_task: int | None = None,
        max_episode_index: int | None = None,
        frames_per_segment: int | None = None,
        frames_per_episode: int | None = None,
        reasoner_subtask_target: bool = False,
        reasoner_target_format: str = "cot_and_subtask",
        next_subgoal_tail_fraction: float = _DEFAULT_NEXT_SUBGOAL_TAIL_FRACTION,
    ) -> None:
        super().__init__()
        if tokenizer_config is None:
            raise ValueError("EpisodeImageEditDataset requires tokenizer_config for text tokenization")

        self.dataset_dir = Path(dataset_dir)
        self.video_dir = self.dataset_dir / video_subdir
        self.instruction_dir = self.dataset_dir / instruction_subdir
        self.video_glob = video_glob
        self.instruction_keys = None if instruction_keys is None else tuple(instruction_keys)
        self.seed = int(seed)
        self.max_caption_tokens = int(max_caption_tokens)
        self.cfg_dropout_rate = float(cfg_dropout_rate)
        self.resize_resolution = resize_resolution
        self.include_final_as_input = bool(include_final_as_input)
        self.first_frame_only = bool(first_frame_only)
        self.dataset_name = dataset_name
        self.metadata_num_workers = max(1, int(metadata_num_workers))
        self.lerobot_video_key = str(lerobot_video_key)
        self.use_three_camera = _as_bool(use_three_camera)
        self.three_camera_video_keys = (
            _normalize_three_camera_video_keys(self.lerobot_video_key, three_camera_video_keys)
            if self.use_three_camera
            else ()
        )
        if self.use_three_camera:
            (
                self.three_camera_concat_video_key,
                self.three_camera_target_hw,
                self.three_camera_aspect_ratio,
            ) = _three_camera_concat_spec(self.lerobot_video_key)
        else:
            self.three_camera_concat_video_key = ""
            self.three_camera_target_hw = (0, 0)
            self.three_camera_aspect_ratio = ""
        self.lerobot_meta_subdir = str(lerobot_meta_subdir)
        self.dataset_format = self._resolve_dataset_format(dataset_format)
        self.target_mode = str(target_mode).lower()
        if self.target_mode not in ("episode_final", "segment_final"):
            raise ValueError(f"target_mode must be 'episode_final' or 'segment_final', got {target_mode!r}")
        self.lerobot_segment_source = str(lerobot_segment_source).lower()
        self.skip_mistake_segments = bool(skip_mistake_segments)
        self.episodes_per_dataset = None if episodes_per_dataset is None else int(episodes_per_dataset)
        if self.episodes_per_dataset is not None and self.episodes_per_dataset < 1:
            raise ValueError(f"episodes_per_dataset must be >= 1 or None, got {episodes_per_dataset!r}")
        self.episodes_per_task = None if episodes_per_task is None else int(episodes_per_task)
        if self.episodes_per_task is not None and self.episodes_per_task < 1:
            raise ValueError(f"episodes_per_task must be >= 1 or None, got {episodes_per_task!r}")
        if self.episodes_per_dataset is not None and self.episodes_per_task is not None:
            raise ValueError("episodes_per_dataset and episodes_per_task are mutually exclusive")
        self.max_episode_index = None if max_episode_index is None else int(max_episode_index)
        if self.max_episode_index is not None and self.max_episode_index < 0:
            raise ValueError(f"max_episode_index must be >= 0 or None, got {max_episode_index!r}")
        self.frames_per_segment = None if frames_per_segment is None else int(frames_per_segment)
        if self.frames_per_segment is not None and self.frames_per_segment < 1:
            raise ValueError(f"frames_per_segment must be >= 1 or None, got {frames_per_segment!r}")
        if self.frames_per_segment is not None and self.target_mode != "segment_final":
            raise ValueError(
                f"frames_per_segment requires target_mode='segment_final', got target_mode={self.target_mode!r}"
            )
        self.reasoner_subtask_target = _as_bool(reasoner_subtask_target)
        self.reasoner_target_format = str(reasoner_target_format).strip().lower()
        if self.reasoner_target_format not in ("cot_and_subtask", "subtask"):
            raise ValueError(
                f"reasoner_target_format must be 'cot_and_subtask' or 'subtask', got {reasoner_target_format!r}"
            )
        if self.reasoner_subtask_target and self.target_mode != "segment_final":
            raise ValueError(
                "reasoner_subtask_target=True trains the reasoner to predict the per-segment subtask "
                "text, which only exists in target_mode='segment_final'; got "
                f"target_mode={self.target_mode!r}"
            )
        self.next_subgoal_tail_fraction = float(next_subgoal_tail_fraction)
        if not 0.0 <= self.next_subgoal_tail_fraction <= 1.0:
            raise ValueError(f"next_subgoal_tail_fraction must be in [0, 1], got {next_subgoal_tail_fraction!r}")
        self.frames_per_episode = None if frames_per_episode is None else int(frames_per_episode)
        if self.frames_per_episode is not None and self.frames_per_episode < 1:
            raise ValueError(f"frames_per_episode must be >= 1 or None, got {frames_per_episode!r}")
        if self.target_mode == "segment_final" and self.dataset_format != "lerobot":
            raise ValueError(
                "target_mode='segment_final' reads subtask segmentation from a LeRobot meta/ dir, so it "
                f"requires dataset_format='lerobot'; got dataset_format={self.dataset_format!r}"
            )

        self._processor = lazy_instantiate(tokenizer_config)
        self._tokenizer = self._processor.tokenizer
        self._metadata_cache_file = _metadata_cache_path(self.dataset_dir)
        self._metadata_cache = _load_metadata_cache(self._metadata_cache_file)
        self._metadata_cache_dirty = False
        self.episodes = self._select_eval_episodes(self._discover_episodes())
        if self._metadata_cache_dirty:
            _write_metadata_cache(self._metadata_cache_file, self._metadata_cache)
        self.samples = self._build_frame_index()
        if not self.samples:
            raise ValueError(f"No eligible frame samples found under {self.dataset_dir}")

        log.info(
            f"EpisodeImageEditDataset loaded {len(self.episodes)} episodes and "
            f"{len(self.samples)} frame samples from {self.dataset_dir}",
            rank0_only=True,
        )

    def _resolve_dataset_format(self, dataset_format: str) -> str:
        fmt = str(dataset_format).lower()
        if fmt not in ("auto", "flat", "lerobot"):
            raise ValueError(f"dataset_format must be 'auto', 'flat', or 'lerobot', got {dataset_format!r}")
        if fmt != "auto":
            return fmt
        meta_dir = self.dataset_dir / self.lerobot_meta_subdir
        if (meta_dir / "info.json").is_file() and (meta_dir / "episodes.jsonl").is_file():
            return "lerobot"
        return "flat"

    def _select_eval_episodes(self, episodes: list[_Episode]) -> list[_Episode]:
        """Choose a deterministic random subset of episodes for bounded evaluation."""

        if self.max_episode_index is not None:
            episodes = [episode for episode in episodes if episode.episode_id <= self.max_episode_index]
        if self.episodes_per_task is not None:
            indexes_by_task: dict[str, list[int]] = {}
            for index, episode in enumerate(episodes):
                task_name = episode.task_name or "unlabeled"
                indexes_by_task.setdefault(task_name, []).append(index)
            selected_indexes: set[int] = set()
            for task_offset, task_name in enumerate(sorted(indexes_by_task)):
                candidates = indexes_by_task[task_name]
                k = min(self.episodes_per_task, len(candidates))
                rng = random.Random(self.seed + task_offset * 1_000_003)
                selected_indexes.update(rng.sample(candidates, k))
            return [episode for index, episode in enumerate(episodes) if index in selected_indexes]
        if self.episodes_per_dataset is None or self.episodes_per_dataset >= len(episodes):
            return episodes
        rng = random.Random(self.seed)
        selected_indexes = set(rng.sample(range(len(episodes)), self.episodes_per_dataset))
        return [episode for index, episode in enumerate(episodes) if index in selected_indexes]

    def _discover_candidates_flat(self) -> list[_EpisodeCandidate]:
        if getattr(self, "use_three_camera", False):
            raise ValueError("use_three_camera=True requires a LeRobot dataset with three camera features")
        if not self.video_dir.is_dir():
            raise FileNotFoundError(f"Missing video directory: {self.video_dir}")
        if not self.instruction_dir.is_dir():
            raise FileNotFoundError(f"Missing instructions directory: {self.instruction_dir}")

        video_paths = sorted(self.video_dir.glob(self.video_glob), key=_episode_sort_key)
        candidates: list[_EpisodeCandidate] = []
        for video_path in video_paths:
            eid = _episode_id(video_path)
            instruction_path = self.instruction_dir / f"episode{eid}.json"
            if not instruction_path.is_file():
                raise FileNotFoundError(f"Missing instruction file for {video_path.name}: {instruction_path}")
            instructions = _load_instruction_set(instruction_path, self.instruction_keys)
            candidates.append(
                _EpisodeCandidate(
                    episode_id=eid,
                    video_path=video_path,
                    instructions=instructions,
                    instruction_path=instruction_path,
                    video_paths=(video_path,),
                )
            )
        return candidates

    def _discover_candidates_lerobot(self) -> list[_EpisodeCandidate]:
        meta_dir = self.dataset_dir / self.lerobot_meta_subdir
        info_path = meta_dir / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot dataset missing {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))

        features = info.get("features", {})
        # Three-camera mode consumes the offline derived stream. The raw camera keys remain
        # configurable only for the preprocessing utility; they are deliberately not decoded
        # in the training/evaluation workers.
        video_keys = (self.three_camera_concat_video_key,) if self.use_three_camera else (self.lerobot_video_key,)
        missing_keys = [key for key in video_keys if key not in features]
        if missing_keys:
            available = sorted(k for k in features if k.startswith("observation.images."))
            if self.use_three_camera:
                raise ValueError(
                    f"Offline three-camera stream {missing_keys!r} not found in {info_path}. "
                    "Run preprocess_episode_image_edit_three_camera.py before training/evaluation. "
                    f"Available camera streams: {available}"
                )
            raise ValueError(
                f"lerobot_video_key {missing_keys!r} not found in {info_path}. Available camera streams: {available}"
            )

        video_template = info.get("video_path")
        if not video_template:
            raise ValueError(f"{info_path} missing required 'video_path' template")
        # video_template e.g. videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
        chunks_size = int(info.get("chunks_size", 1000)) or 1000

        instructions_by_episode = _load_lerobot_instructions(meta_dir)
        task_names_by_episode = _task_names_from_annotations(meta_dir)
        segments_by_episode: dict[int, tuple[tuple[int, int, bool, str, str], ...]] = {}
        if self.target_mode == "segment_final":
            if self.lerobot_segment_source in ("episode_final", "auto_or_episode_final"):
                # An explicit opt-in fallback for single-stage datasets with no subtask
                # annotation. Use the metadata length where available; _prepare_episode_segments
                # later clamps this to the decodable video length. A very large end preserves the
                # same behavior for old LeRobot metadata that omits ``length``.
                fallback_segments = {
                    eid: ((0, length if length is not None else 2**63 - 1, False, "", ""),)
                    for eid, (_, length) in instructions_by_episode.items()
                }
                if self.lerobot_segment_source == "episode_final":
                    segments_by_episode = fallback_segments
                else:
                    # Preserve real multi-subtask annotations where they exist, and only then
                    # use the one-completion-frame fallback for a task/episode without them.
                    annotated = (
                        _segments_from_annotations(meta_dir)
                        or _segments_from_legacy_action_annotations(meta_dir)
                        or _segments_from_instruction_segments(meta_dir)
                    )
                    segments_by_episode = {
                        eid: annotated.get(eid, fallback_segments[eid]) for eid in instructions_by_episode
                    }
            else:
                segments_by_episode = _load_lerobot_segments(meta_dir, self.lerobot_segment_source)

        candidates: list[_EpisodeCandidate] = []
        for eid in sorted(instructions_by_episode):
            instructions, meta_length = instructions_by_episode[eid]
            video_paths = tuple(
                self.dataset_dir
                / video_template.format(
                    episode_chunk=eid // chunks_size,
                    video_key=video_key,
                    episode_index=eid,
                )
                for video_key in video_keys
            )
            for video_key, video_path in zip(video_keys, video_paths, strict=True):
                if not video_path.is_file():
                    raise FileNotFoundError(
                        f"LeRobot episode {eid}: expected {video_key} video at {video_path} "
                        f"(from video_path template {video_template!r})"
                    )
            video_path = video_paths[0]
            segments = segments_by_episode.get(eid, ()) if self.target_mode == "segment_final" else ()
            if self.target_mode == "segment_final" and not segments:
                raise ValueError(
                    f"LeRobot episode {eid}: no subtask segments found via lerobot_segment_source="
                    f"{self.lerobot_segment_source!r}; cannot build segment_final targets"
                )
            candidates.append(
                _EpisodeCandidate(
                    episode_id=eid,
                    video_path=video_path,
                    instructions=instructions,
                    instruction_path=meta_dir / "episodes.jsonl",
                    # ``episodes.jsonl`` is the authoritative source of task instructions. Some
                    # converted datasets carry a coarser/stale ``annotations.json.task_name``;
                    # using it here would collapse distinct tasks during per-task evaluation.
                    task_name=instructions[0] if instructions else task_names_by_episode.get(eid, ""),
                    meta_length=meta_length,
                    segments=segments,
                    video_paths=video_paths,
                    metadata_hints=tuple(
                        hint or {}
                        for hint in (
                            _lerobot_feature_metadata(info, {"length": meta_length}, key) for key in video_keys
                        )
                    ),
                )
            )
        if not candidates:
            raise ValueError(f"No LeRobot episodes discovered under {self.dataset_dir}")
        return candidates

    def _discover_episodes(self) -> list[_Episode]:
        if self.dataset_format == "lerobot":
            candidates = self._discover_candidates_lerobot()
        else:
            candidates = self._discover_candidates_flat()

        candidate_paths = {
            path: hint
            for cand in candidates
            for path, hint in zip(
                cand.video_paths or (cand.video_path,),
                cand.metadata_hints or ({},) * len(cand.video_paths or (cand.video_path,)),
                strict=True,
            )
        }
        missing = [
            path for path in candidate_paths if str(path.relative_to(self.dataset_dir)) not in self._metadata_cache
        ]
        if missing:
            log.info(
                f"Reading metadata for {len(missing)} videos with {self.metadata_num_workers} workers",
                rank0_only=True,
            )
            hinted = {
                path: hint
                for path, hint in candidate_paths.items()
                if path in missing
                and hint
                and all(field in hint for field in ("width", "height", "fps", "total_frames"))
            }
            for path, metadata in hinted.items():
                self._metadata_cache[str(path.relative_to(self.dataset_dir))] = metadata
            if hinted:
                self._metadata_cache_dirty = True
            missing = [path for path in missing if path not in hinted]
            if missing:
                with ThreadPoolExecutor(max_workers=self.metadata_num_workers) as executor:
                    for video_path, metadata in zip(
                        missing,
                        executor.map(_get_video_metadata, missing),
                        strict=True,
                    ):
                        cache_key = str(video_path.relative_to(self.dataset_dir))
                        self._metadata_cache[cache_key] = metadata
                self._metadata_cache_dirty = True

        episodes: list[_Episode] = []
        for cand in candidates:
            video_paths = cand.video_paths or (cand.video_path,)
            metadata_list = tuple(self._metadata_cache[str(path.relative_to(self.dataset_dir))] for path in video_paths)
            metadata = metadata_list[0]
            total_frames = min(int(item["total_frames"]) for item in metadata_list)
            if total_frames < 1:
                raise ValueError(f"{cand.video_path} has no frames")
            # When the layout supplies an authoritative frame count (LeRobot's
            # per-episode `length`), clamp to the smaller of it and ffprobe's count.
            # ffprobe nb_frames / duration*fps can overcount, which would make
            # final_idx point past the last decodable frame and hard-fail every
            # sample of the episode in _decode_two_frames.
            if cand.meta_length is not None and cand.meta_length >= 1:
                total_frames = min(total_frames, cand.meta_length)

            width = int(metadata["width"])
            height = int(metadata["height"])
            if self.use_three_camera:
                target_height, target_width = self.three_camera_target_hw
                aspect_ratio = self.three_camera_aspect_ratio
            else:
                aspect_ratio, target_width, target_height = _resolve_target_size(width, height, self.resize_resolution)
            segments, segment_ends = _prepare_episode_segments(cand.segments, total_frames, cand.episode_id)
            episodes.append(
                _Episode(
                    episode_id=cand.episode_id,
                    video_path=cand.video_path,
                    instruction_path=cand.instruction_path,
                    width=width,
                    height=height,
                    fps=float(metadata["fps"]),
                    total_frames=total_frames,
                    instructions=cand.instructions,
                    aspect_ratio=aspect_ratio,
                    target_width=target_width,
                    target_height=target_height,
                    video_paths=video_paths,
                    video_metadata=metadata_list,
                    task_name=cand.task_name,
                    segments=segments,
                    segment_ends=segment_ends,
                )
            )
        return episodes

    def _mistake_ranges(self, episode: _Episode) -> tuple[tuple[int, int], ...]:
        """``[start, end)`` ranges to exclude from inputs (mistake segments in segment mode)."""

        if self.target_mode != "segment_final" or not self.skip_mistake_segments:
            return ()
        return tuple((start, end) for start, end, mistake, _, _ in episode.segments if mistake)

    def _build_frame_index(self) -> list[tuple[int, int]]:
        samples: list[tuple[int, int]] = []
        for episode_idx, episode in enumerate(self.episodes):
            mistake_ranges = self._mistake_ranges(episode)
            # A segment-final sample is valid only when its input frame is actually covered by
            # an annotation.  Mapping a frame in an internal/trailing annotation gap to the
            # "nearest" segment end can put the target *before* the source frame and attach the
            # wrong reasoner label (common in recordings with an unannotated post-task tail).
            segment_ranges = (
                tuple((start, end) for start, end, _, _, _ in episode.segments)
                if self.target_mode == "segment_final"
                else ()
            )
            if self.target_mode == "segment_final" and not segment_ranges:
                continue

            def is_eligible(frame_idx: int) -> bool:
                if segment_ranges and not _frame_in_ranges(frame_idx, segment_ranges):
                    return False
                target_segment_index = _subgoal_target_segment_index(
                    episode.segments, frame_idx, self.next_subgoal_tail_fraction
                )
                if self.target_mode == "segment_final" and target_segment_index is None:
                    return False
                if (
                    target_segment_index is not None
                    and self.skip_mistake_segments
                    and episode.segments[target_segment_index][2]
                ):
                    return False
                if self.reasoner_subtask_target:
                    assert target_segment_index is not None
                    target_segment = episode.segments[target_segment_index]
                    if not target_segment[3].strip():
                        return False
                    if self.reasoner_target_format == "cot_and_subtask" and not target_segment[4].strip():
                        return False
                return not mistake_ranges or not _frame_in_ranges(frame_idx, mistake_ranges)

            last_input = episode.total_frames - 1 if self.include_final_as_input else episode.total_frames - 2
            # Subgoal eval: choose independent input frames inside every retained segment so each
            # prediction is compared with that segment's annotated final frame. This must precede
            # frames_per_episode, whose episode-wide sample can omit short segments entirely.
            if self.frames_per_segment is not None:
                for segment_index, (start, end, is_mistake, subtask, cot) in enumerate(episode.segments):
                    if self.skip_mistake_segments and is_mistake:
                        continue
                    if self.reasoner_subtask_target:
                        if not subtask.strip():
                            continue
                        if self.reasoner_target_format == "cot_and_subtask" and not cot.strip():
                            continue
                    segment_last_input = min(last_input, end - 1)
                    if segment_last_input < start:
                        continue
                    candidates = [
                        frame_idx for frame_idx in range(start, segment_last_input + 1) if is_eligible(frame_idx)
                    ]
                    if not candidates:
                        continue
                    k = min(self.frames_per_segment, len(candidates))
                    rng = random.Random(self.seed + episode.episode_id * 1_000_003 + segment_index)
                    for frame_idx in sorted(rng.sample(candidates, k)):
                        samples.append((episode_idx, frame_idx))
                continue
            # Eval sampling: N random distinct input frames per episode (seeded per episode so the
            # selection is reproducible across ranks/runs). Takes precedence over first_frame_only.
            if self.frames_per_episode is not None:
                if last_input < 0:
                    continue
                candidates = [f for f in range(last_input + 1) if is_eligible(f)]
                if not candidates:
                    continue
                rng = random.Random(self.seed + episode.episode_id)
                k = min(self.frames_per_episode, len(candidates))
                for frame_idx in sorted(rng.sample(candidates, k)):
                    samples.append((episode_idx, frame_idx))
                continue
            if self.first_frame_only:
                if episode.total_frames >= 1 and is_eligible(0):
                    samples.append((episode_idx, 0))
                continue
            if last_input < 0:
                continue
            for frame_idx in range(last_input + 1):
                if not is_eligible(frame_idx):
                    continue
                samples.append((episode_idx, frame_idx))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _rng_for_index(self, index: int) -> random.Random:
        return random.Random(self.seed + index * 1_000_003)

    def _tokenize_caption(self, caption: str) -> torch.Tensor:
        text_ids = tokenize_caption(caption, self._tokenizer, system_prompt=_SYSTEM_PROMPT_IMAGE_EDITING)
        if len(text_ids) > self.max_caption_tokens:
            log.warning(
                f"Episode image-edit caption has {len(text_ids)} tokens, truncating to {self.max_caption_tokens}",
                rank0_only=False,
            )
            text_ids = text_ids[: self.max_caption_tokens]
        return torch.tensor(text_ids, dtype=torch.long)

    def _tokenize_prompt_and_target(
        self, prompt_caption: str, cot: str, subtask_text: str
    ) -> tuple[torch.Tensor, int, str]:
        """Tokenize a reasoner sample: episode prompt + Cosmos think trace + subtask.

        ``prompt_caption`` is wrapped in the chat template (system + user + assistant generation
        header) exactly like :meth:`_tokenize_caption`, so its final tokens are the assistant turn
        opener. The target is tokenized as a plain continuation (no special tokens) and appended —
        these are the tokens the reasoner must learn to produce. Cosmos reasoner traces use the
        native ``<think>...</think>`` block followed by the final answer.
        Returns ``(token_ids, num_prompt_tokens)`` where ``num_prompt_tokens`` is the length of the
        prompt portion (everything the CE loss must ignore). The trailing EOS/target-stop token is
        appended by the packer, not here.
        """
        prompt_ids = tokenize_caption(prompt_caption, self._tokenizer, system_prompt=_SYSTEM_PROMPT_IMAGE_EDITING)
        if self.reasoner_target_format == "subtask":
            target_text = subtask_text.strip()
        else:
            target_text = f"<think>\n{cot.strip()}\n</think>\n{subtask_text.strip()}"
        target_ids = self._tokenizer.encode(target_text, add_special_tokens=False)
        # Keep the whole target; truncate only the prompt if the pair exceeds the cap so we never
        # drop the supervised subtask tokens.
        if len(prompt_ids) + len(target_ids) > self.max_caption_tokens:
            keep_prompt = max(1, self.max_caption_tokens - len(target_ids))
            log.warning(
                f"Episode image-edit reasoner sample has {len(prompt_ids) + len(target_ids)} tokens, "
                f"truncating prompt to {keep_prompt}",
                rank0_only=False,
            )
            prompt_ids = prompt_ids[:keep_prompt]
        token_ids = list(prompt_ids) + list(target_ids)
        return torch.tensor(token_ids, dtype=torch.long), len(prompt_ids), target_text

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)

        episode_idx, input_idx = self.samples[index]
        episode = self.episodes[episode_idx]
        rng = self._rng_for_index(index)

        if self.target_mode == "segment_final":
            source_seg_k = next(
                (
                    segment_index
                    for segment_index, (start, end, _, _, _) in enumerate(episode.segments)
                    if start <= input_idx < end
                ),
                -1,
            )
            if source_seg_k < 0:
                raise RuntimeError(
                    f"No annotated source segment for episode={episode.episode_id} frame={input_idx}; "
                    "the frame index should have excluded uncovered frames"
                )
            target = _segment_target_index(
                episode.segments,
                input_idx,
                episode.total_frames,
                self.next_subgoal_tail_fraction,
            )
            if target is None:
                raise RuntimeError(
                    f"No annotated subgoal target for episode={episode.episode_id} frame={input_idx}; "
                    "the frame index should have excluded uncovered frames"
                )
            target_seg_k, final_idx = target
            source_segment_start, source_segment_end = episode.segments[source_seg_k][:2]
            target_segment_start, target_segment_end = episode.segments[target_seg_k][:2]
            subtask_text = episode.segments[target_seg_k][3]
            cot_text = episode.segments[target_seg_k][4]
        else:
            source_seg_k = -1
            target_seg_k = -1
            source_segment_start = 0
            source_segment_end = episode.total_frames
            target_segment_start = 0
            target_segment_end = episode.total_frames
            final_idx = episode.total_frames - 1
            subtask_text, cot_text = "", ""
        episode_instruction = rng.choice(episode.instructions)
        if self.reasoner_subtask_target:
            # Reasoner training: prompt is the episode task ONLY (never the subtask), and the
            # checked CoT plus authoritative subtask is the supervised target. cfg dropout, if
            # enabled, drops the prompt text only.
            prompt_caption = (
                "" if self.cfg_dropout_rate > 0 and rng.random() < self.cfg_dropout_rate else episode_instruction
            )
            subtask = subtask_text.strip()
            cot = cot_text.strip()
            if subtask and (cot or self.reasoner_target_format == "subtask"):
                text_token_ids, num_prompt_tokens, reasoner_target_text = self._tokenize_prompt_and_target(
                    prompt_caption, cot, subtask
                )
                # ``ai_caption`` is the inference/generator input contract, so it must never carry
                # the supervised subtask. Training consumes ``text_token_ids`` for CE instead.
                caption = prompt_caption
            else:
                raise RuntimeError(
                    f"reasoner_subtask_target with format={self.reasoner_target_format!r} lacks "
                    f"required target metadata; episode={episode.episode_id} frame={input_idx}"
                )
        else:
            # Subtask annotations select the target frame only. The reasoner's sole text input is
            # always the episode-level task instruction from episodes.jsonl.
            caption = "" if self.cfg_dropout_rate > 0 and rng.random() < self.cfg_dropout_rate else episode_instruction
            text_token_ids = self._tokenize_caption(caption)
            # Every text token is prompt/conditioning (no reasoner CE target) in the legacy path.
            num_prompt_tokens = int(text_token_ids.numel())
            reasoner_target_text = ""

        source_frame, target_frame = _decode_two_frames(
            episode.video_path,
            frame_idx=input_idx,
            final_idx=final_idx,
            fps=episode.fps,
            target_height=episode.target_height,
            target_width=episode.target_width,
        )
        source_tensor = _to_chw_tensor(source_frame)
        target_tensor = _to_chw_tensor(target_frame)
        image_size = torch.tensor(
            [episode.target_height, episode.target_width, episode.target_height, episode.target_width],
            dtype=torch.float32,
        )

        return {
            "__key__": f"episode{episode.episode_id}_f{input_idx}_to_{final_idx}",
            "__url__": str(episode.video_path),
            "dataset_name": self.dataset_name,
            "task_name": episode.task_name,
            "images": [source_tensor, target_tensor],
            "image_size": [image_size, image_size.clone()],
            "ai_caption": caption,
            # Plain episode-level prompt for reasoner-generation evaluation.
            "reasoner_prompt": episode_instruction,
            "reasoner_target_text": reasoner_target_text,
            "selected_caption_type": "editing_instruction",
            "text_token_ids": text_token_ids,
            "num_prompt_tokens": int(num_prompt_tokens),
            "fps": episode.fps,
            "conditioning_fps": episode.fps,
            "num_frames": 2,
            "sequence_plan": SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=[],
            ),
            "episode_id": episode.episode_id,
            "input_frame_index": input_idx,
            "target_frame_index": final_idx,
            "source_segment_index": source_seg_k,
            "target_segment_index": target_seg_k,
            "source_segment_start_frame": source_segment_start,
            "source_segment_end_frame_exclusive": source_segment_end,
            "target_segment_start_frame": target_segment_start,
            "target_segment_end_frame_exclusive": target_segment_end,
            "num_segments": len(episode.segments) if self.target_mode == "segment_final" else 1,
            "n_orig_video_frames": episode.total_frames,
            "aspect_ratio": episode.aspect_ratio,
            "height": episode.target_height,
            "width": episode.target_width,
            "original_height": episode.height,
            "original_width": episode.width,
            "resize_resolution": self.resize_resolution,
        }


def _parse_dataset_dirs(dataset_dir: Any) -> list[str]:
    """Normalize ``dataset_dir`` to a list of dataset roots.

    Accepts a single path, a list/tuple of paths, or a string naming several roots
    separated by a comma or a colon (``os.pathsep``) — the latter so one
    ``EPISODE_IMAGE_EDIT_DATASET_PATH`` env var can point at multiple LeRobot datasets.
    """

    if isinstance(dataset_dir, (list, tuple)):
        raw = [str(d) for d in dataset_dir]
    else:
        raw = re.split(r"[,:]", str(dataset_dir))
    dirs = [p.strip() for p in raw if p and p.strip()]
    if not dirs:
        raise ValueError(f"dataset_dir resolved to no paths: {dataset_dir!r}")
    return dirs


def get_episode_image_edit_dataset(**kwargs: Any) -> torch.utils.data.Dataset:
    """LazyConfig-friendly factory supporting one or many dataset roots.

    ``dataset_dir`` may name a single root or several (a list, or a comma-/colon-separated
    string). One root returns a single :class:`EpisodeImageEditDataset` (unchanged path).
    Several roots return a :class:`torch.utils.data.ConcatDataset` pooling every episode and
    frame across the roots, so the ``MapDistributor`` shuffles/shards the combined index and
    training draws uniformly across datasets. Each root loads independently (its own videos,
    instructions, subtask segmentation, and metadata cache) while sharing the per-frame
    settings (``target_mode``, ``lerobot_video_key``, ``frames_per_episode``, ...).
    """

    dirs = _parse_dataset_dirs(kwargs.get("dataset_dir"))
    if len(dirs) == 1:
        return EpisodeImageEditDataset(**kwargs)

    kwargs.pop("dataset_dir")
    base_seed = int(kwargs.pop("seed", 42))
    base_name = str(kwargs.pop("dataset_name", "episode_image_editing"))
    subsets = [
        EpisodeImageEditDataset(
            dataset_dir=d,
            seed=base_seed + i,  # decorrelate per-dataset frame/instruction sampling
            dataset_name=f"{base_name}:{Path(d).name}",
            **kwargs,
        )
        for i, d in enumerate(dirs)
    ]
    total_episodes = sum(len(s.episodes) for s in subsets)
    total_samples = sum(len(s.samples) for s in subsets)
    log.info(
        f"EpisodeImageEditDataset: concatenated {len(subsets)} datasets "
        f"({', '.join(Path(d).name for d in dirs)}) -> {total_episodes} episodes, "
        f"{total_samples} frame samples",
        rank0_only=True,
    )
    return torch.utils.data.ConcatDataset(subsets)
