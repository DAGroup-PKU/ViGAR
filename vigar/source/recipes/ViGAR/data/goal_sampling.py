"""Recorded future goals with soft segment boundaries and occurrence-local RNG."""

import hashlib
import math
import random
from dataclasses import asdict, dataclass, field


GOAL_SAMPLING_VERSION = 2


def _number(value, name, *, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"goal_sampling.{name} must be finite and >= {minimum}")


@dataclass
class FutureGoalConfig:
    max_offset_seconds: float = 4.0
    min_horizon_multiple: float = 2.0

    def __post_init__(self):
        _number(self.max_offset_seconds, "future.max_offset_seconds")
        _number(self.min_horizon_multiple, "future.min_horizon_multiple", minimum=1.0)


@dataclass
class SegmentGoalConfig:
    boundary_jitter_seconds: float = 0.5
    max_future_endpoints: int = 2
    endpoint_rank_decay: float = 0.5

    def __post_init__(self):
        _number(self.boundary_jitter_seconds, "segment.boundary_jitter_seconds")
        if type(self.max_future_endpoints) is not int or self.max_future_endpoints < 1:
            raise ValueError("goal_sampling.segment.max_future_endpoints must be a positive integer")
        _number(self.endpoint_rank_decay, "segment.endpoint_rank_decay")
        if not 0 < self.endpoint_rank_decay <= 1:
            raise ValueError("goal_sampling.segment.endpoint_rank_decay must be in (0, 1]")


@dataclass
class GoalSamplingConfig:
    mode: str = "terminal"
    weights: dict[str, float] = field(default_factory=lambda: dict(terminal=0.30, segment=0.25, future=0.45))
    min_offset: str = "full_chunk"
    future: FutureGoalConfig = field(default_factory=FutureGoalConfig)
    segment: SegmentGoalConfig = field(default_factory=SegmentGoalConfig)
    fallback: dict[str, str] = field(default_factory=lambda: dict(segment="future", future="terminal"))
    eval_mode: str = "terminal"

    def __post_init__(self):
        for name, cls in (("future", FutureGoalConfig), ("segment", SegmentGoalConfig)):
            value = getattr(self, name)
            if isinstance(value, dict):
                try:
                    value = cls(**value)
                except TypeError as error:
                    raise ValueError(f"Invalid goal_sampling.{name} configuration") from error
                setattr(self, name, value)
            if not isinstance(value, cls):
                raise ValueError(f"goal_sampling.{name} must be a configuration mapping")
        if self.mode not in ("terminal", "mixture"):
            raise ValueError("goal_sampling.mode must be terminal or mixture")
        if self.min_offset != "full_chunk" or self.eval_mode != "terminal":
            raise ValueError("goal_sampling requires min_offset=full_chunk and eval_mode=terminal")
        if not isinstance(self.weights, dict) or set(self.weights) != {"terminal", "segment", "future"}:
            raise ValueError("goal_sampling.weights must specify terminal, segment and future")
        for name, value in self.weights.items():
            _number(value, f"weights.{name}")
        if not math.isclose(sum(self.weights.values()), 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError("goal_sampling.weights must sum to 1")
        if (
            not isinstance(self.fallback, dict)
            or set(self.fallback) != {"segment", "future"}
            or self.fallback["segment"] not in ("future", "terminal")
            or self.fallback["future"] != "terminal"
        ):
            raise ValueError("goal_sampling.fallback requires segment=future|terminal and future=terminal")

    def to_dict(self):
        return asdict(self)

    @property
    def terminal_only(self):
        """A degenerate mixture uses the legacy path, including RNG and resume."""
        return self.mode == "terminal" or (self.weights["segment"] == 0 and self.weights["future"] == 0)


def resolve_goal_sampling(value):
    if value is None:
        return GoalSamplingConfig()
    if isinstance(value, GoalSamplingConfig):
        # Copy and revalidate even when called directly, outside the recipe parser.
        value = value.to_dict()
    if not isinstance(value, dict):
        raise ValueError("goal_sampling must be a configuration mapping")
    try:
        return GoalSamplingConfig(**value)
    except TypeError as error:
        raise ValueError("Invalid goal_sampling configuration") from error


def _missing(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _seconds(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{context}: expected a finite timestamp")
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{context}: expected a finite timestamp") from error
    if not math.isfinite(result):
        raise ValueError(f"{context}: expected a finite timestamp")
    return result


def episode_goal_timing(annotation, length, fps, *, context):
    """Normalize absent optional annotations without relaxing structural validity."""
    segments = annotation.get("segments")
    if _missing(segments):
        segments = []
    if not isinstance(segments, list):
        raise ValueError(f"{context}: segments must be a list or absent")
    previous_end = 0.0
    endpoints = []
    for segment in segments:
        if not isinstance(segment, dict) or "start_time" not in segment or "end_time" not in segment:
            raise ValueError(f"{context}: segment requires start_time and end_time")
        lo = _seconds(segment["start_time"], f"{context}/segment/start_time")
        hi = _seconds(segment["end_time"], f"{context}/segment/end_time")
        if not previous_end <= lo < hi <= float(annotation["duration"]) + 1e-6:
            raise ValueError(f"{context}: overlapping, unordered or out-of-bounds segment")
        caption = segment.get("caption")
        captions = caption.get("overall") if isinstance(caption, dict) else None
        if not isinstance(captions, dict) or not all(
            isinstance(captions.get(lang), str) and captions[lang].strip() for lang in ("en",)
        ):
            raise ValueError(f"{context}: missing English segment caption")
        previous_end = hi
        endpoints.append(math.ceil(hi * fps - 1e-6) - 1)
    start_time, end_time = annotation.get("effective_start_time"), annotation.get("effective_end_time")
    start_time = 0.0 if _missing(start_time) else _seconds(start_time, f"{context}/effective_start_time")
    end_time = length / fps if _missing(end_time) else _seconds(end_time, f"{context}/effective_end_time")
    if not 0 <= start_time < end_time <= length / fps + 1e-6:
        raise ValueError(f"{context}: invalid effective interval")
    return (
        max(0, math.ceil(start_time * fps - 1e-6)),
        min(length, math.ceil(end_time * fps - 1e-6)),
        endpoints,
    )


def select_goal(config, *, start, end, fps, rate, horizon, endpoints, seed, occurrence, training):
    """Select once from legal row intervals; do not consume global RNG state."""
    terminal = end - 1
    source = requested = "terminal"
    goal, segment_index, reason = terminal, -1, ""
    if training and not config.terminal_only:
        digest = hashlib.sha256(f"vigar-goal-v{GOAL_SAMPLING_VERSION}:{seed}:{occurrence}".encode()).digest()
        rng = random.Random(digest)
        sources = ("terminal", "segment", "future")
        source = requested = rng.choices(sources, weights=[config.weights[key] for key in sources])[0]
        lower = min(start + horizon * rate, terminal)
        if source == "segment":
            jitter = config.segment.boundary_jitter_seconds * fps
            candidates = []
            for index, center in enumerate(endpoints):
                lo = max(lower, math.ceil(center - jitter - 1e-6))
                hi = min(terminal, math.floor(center + jitter + 1e-6))
                if lo <= hi:
                    candidates.append((index, lo, hi))
                    if len(candidates) == config.segment.max_future_endpoints:
                        break
            if candidates:
                weights = [config.segment.endpoint_rank_decay**j for j in range(len(candidates))]
                segment_index, lo, hi = rng.choices(candidates, weights=weights)[0]
                goal = rng.randint(lo, hi)
            else:
                source = config.fallback["segment"]
                reason = "no_eligible_segment"
        if source == "future":
            delay = max(config.future.max_offset_seconds, config.future.min_horizon_multiple * horizon * rate / fps)
            upper = min(terminal, math.floor(start + delay * fps + 1e-6))
            if lower <= upper:
                goal = rng.randint(lower, upper)
            else:
                source = config.fallback["future"]
                reason = "+".join(filter(None, (reason, "no_eligible_future")))
    return dict(
        goal_index=goal,
        goal_requested_source=requested,
        goal_source=source,
        goal_delay_seconds=(goal - start) / fps,
        goal_segment_index=segment_index,
        goal_fallback_reason=reason,
    )
