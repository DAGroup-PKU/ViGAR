"""Optional current-segment instructions on the recorded episode timeline."""

import math


def validate_text_conditioning(value):
    if value not in ("episode", "segment", "episode_segment"):
        raise ValueError("text_conditioning must be episode, segment or episode_segment")
    return value


def segment_text_intervals(annotation, fps):
    """Use validated annotations and the same half-open frame rounding as goals."""
    segments = annotation.get("segments")
    if segments is None or (isinstance(segments, str) and not segments.strip()):
        return []
    return [
        (
            math.ceil(float(segment["start_time"]) * fps - 1e-6),
            math.ceil(float(segment["end_time"]) * fps - 1e-6),
            segment["caption"]["overall"]["en"],
        )
        for segment in segments
    ]


def select_text(episode_text, intervals, frame, mode):
    """Select by current observation, independently of the sampled visual goal."""
    if mode != "episode":
        for start, end, caption in intervals:
            if start <= frame < end:
                if mode == "segment":
                    return caption
                return f"{episode_text}\nCurrent subtask: {caption}"
    return episode_text
