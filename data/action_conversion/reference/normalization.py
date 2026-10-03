"""0824 normalization primitives, retained with unchanged numerical behavior."""

import re
from typing import Dict

import numpy as np
import torch


_BOUNDS_99_CLIP_RE = re.compile(r"^bounds_99_clip([1-9][0-9]*)$")


def parse_bounds_99_clip_limit(norm_type: str) -> int | None:
    match = _BOUNDS_99_CLIP_RE.match(norm_type)
    if match is None:
        return None
    return int(match.group(1))


def dict_apply(func, d):
    """
    Apply a function to all values in a dictionary recursively.
    If the value is a dictionary, it will apply the function to its values.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            dict_apply(func, value)
        else:
            d[key] = func(value)
    return d


class Normalizer:
    def __init__(
        self,
        norm_stats: Dict[str, Dict[str, np.ndarray]],
        from_file: bool = False,
        data_type: str = None,
        norm_type: Dict[str, str] | None = None,
    ):
        if from_file:
            if data_type == "libero":
                norm_stats["state"]["mean"] = np.array(norm_stats["state"]["mean"][:8])
                norm_stats["state"]["std"] = np.array(norm_stats["state"]["std"][:8])
                norm_stats["actions"]["mean"] = np.array(norm_stats["actions"]["mean"][:7])
                norm_stats["actions"]["std"] = np.array(norm_stats["actions"]["std"][:7])
            elif data_type == "robotwin":
                norm_stats["observation.state"], norm_stats["action"] = {}, {}
                norm_stats["observation.state"]["q01"] = np.array(
                    norm_stats["observation.state.arm.position"]["q01"][:6]
                    + norm_stats["observation.state.effector.position"]["q01"][:1]
                    + norm_stats["observation.state.arm.position"]["q01"][6:]
                    + norm_stats["observation.state.effector.position"]["q01"][1:]
                )
                norm_stats["observation.state"]["q99"] = np.array(
                    norm_stats["observation.state.arm.position"]["q99"][:6]
                    + norm_stats["observation.state.effector.position"]["q99"][:1]
                    + norm_stats["observation.state.arm.position"]["q99"][6:]
                    + norm_stats["observation.state.effector.position"]["q99"][1:]
                )
                norm_stats["action"]["q01"] = np.array(
                    norm_stats["action.arm.position"]["q01"][:6]
                    + norm_stats["action.effector.position"]["q01"][:1]
                    + norm_stats["action.arm.position"]["q01"][6:]
                    + norm_stats["action.effector.position"]["q01"][1:]
                )
                norm_stats["action"]["q99"] = np.array(
                    norm_stats["action.arm.position"]["q99"][:6]
                    + norm_stats["action.effector.position"]["q99"][:1]
                    + norm_stats["action.arm.position"]["q99"][6:]
                    + norm_stats["action.effector.position"]["q99"][1:]
                )
            elif data_type == "robotwin_rep":
                norm_stats["observation.state"], norm_stats["action"] = {}, {}
                norm_stats["observation.state"]["q01"] = np.array(
                    norm_stats["observation.state.arm.position"]["q01"]
                    + norm_stats["observation.state.effector.position"]["q01"]
                )
                norm_stats["observation.state"]["q99"] = np.array(
                    norm_stats["observation.state.arm.position"]["q99"]
                    + norm_stats["observation.state.effector.position"]["q99"]
                )
                norm_stats["action"]["q01"] = np.array(
                    norm_stats["action.arm.position"]["q01"][:6]
                    + norm_stats["action.effector.position"]["q01"][:1]
                    + norm_stats["action.arm.position"]["q01"][6:]
                    + norm_stats["action.effector.position"]["q01"][1:]
                )
                norm_stats["action"]["q99"] = np.array(
                    norm_stats["action.arm.position"]["q99"][:6]
                    + norm_stats["action.effector.position"]["q99"][:1]
                    + norm_stats["action.arm.position"]["q99"][6:]
                    + norm_stats["action.effector.position"]["q99"][1:]
                )
            elif data_type == "customized":
                for key in norm_stats:
                    if isinstance(norm_stats[key], dict):
                        for sub_key in norm_stats[key]:
                            norm_stats[key][sub_key] = np.array(norm_stats[key][sub_key])
            self.norm_stats = norm_stats
        else:
            self.norm_stats = dict_apply(lambda x: np.asarray(x, dtype=np.float32), norm_stats)
        self.norm_type = norm_type or {}
        self.from_file = from_file

    @staticmethod
    def _stat_like(value, stat):
        if isinstance(value, torch.Tensor):
            return torch.as_tensor(stat, device=value.device, dtype=value.dtype)
        return np.asarray(stat, dtype=np.float32)

    @staticmethod
    def _require_stats(stats: Dict[str, np.ndarray], key: str, norm_type: str, required: tuple[str, ...]):
        missing = [name for name in required if name not in stats]
        if missing:
            raise KeyError(
                f"norm_stats for {key!r} are missing {missing} required by norm_type={norm_type!r}. "
                "Provide normalization statistics containing the required fields."
            )

    @staticmethod
    def _bounds_99_clip_limit(norm_type: str) -> int | None:
        return parse_bounds_99_clip_limit(norm_type)

    @staticmethod
    def _bounds_normalize(value, low, high):
        range_val = high - low
        if isinstance(range_val, torch.Tensor):
            degenerate = range_val < 1e-4
            safe_range = torch.where(degenerate, torch.ones_like(range_val), range_val + 1e-6)
            normalized = (value - low) / safe_range * 2.0 - 1.0
            return torch.where(degenerate, value, normalized)
        else:
            degenerate = range_val < 1e-4
            safe_range = np.where(degenerate, 1.0, range_val + 1e-6)
            normalized = (value - low) / safe_range * 2.0 - 1.0
            return np.where(degenerate, value, normalized)

    @staticmethod
    def _bounds_unnormalize(value, low, high):
        range_val = high - low
        if isinstance(range_val, torch.Tensor):
            degenerate = range_val < 1e-4
            safe_range = torch.where(degenerate, torch.ones_like(range_val), range_val + 1e-6)
            unnormalized = ((value + 1.0) / 2.0) * safe_range + low
            return torch.where(degenerate, value, unnormalized)
        else:
            degenerate = range_val < 1e-4
            safe_range = np.where(degenerate, 1.0, range_val + 1e-6)
            unnormalized = ((value + 1.0) / 2.0) * safe_range + low
            return np.where(degenerate, value, unnormalized)

    @staticmethod
    def _restore_degenerate(result, value, scale):
        if isinstance(scale, torch.Tensor):
            return torch.where(scale < 1e-4, value, result)
        return np.where(scale < 1e-4, value, result)

    def normalize(self, data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        normalized_data = {}
        for key, value in data.items():
            if key in self.norm_stats:
                norm_type = self.norm_type.get(key, "identity")
                stats = self.norm_stats[key]
                bounds_99_clip_limit = self._bounds_99_clip_limit(norm_type)
                if norm_type == "meanstd":
                    self._require_stats(stats, key, norm_type, ("mean", "std"))
                    mean = self._stat_like(value, stats["mean"])
                    std = self._stat_like(value, stats["std"])
                    normalized_value = self._restore_degenerate((value - mean) / (std + 1e-6), value, std)
                elif norm_type == "bounds_99_woclip":
                    self._require_stats(stats, key, norm_type, ("q01", "q99"))
                    low = self._stat_like(value, stats["q01"])
                    high = self._stat_like(value, stats["q99"])
                    normalized_value = self._bounds_normalize(value, low, high)
                elif bounds_99_clip_limit is not None:
                    self._require_stats(stats, key, norm_type, ("q01", "q99"))
                    low = self._stat_like(value, stats["q01"])
                    high = self._stat_like(value, stats["q99"])
                    normalized_value = self._bounds_normalize(value, low, high)
                    if isinstance(normalized_value, torch.Tensor):
                        normalized_value = torch.clamp(
                            normalized_value,
                            -float(bounds_99_clip_limit),
                            float(bounds_99_clip_limit),
                        )
                    else:
                        normalized_value = np.clip(
                            normalized_value,
                            -float(bounds_99_clip_limit),
                            float(bounds_99_clip_limit),
                        )
                    normalized_value = self._restore_degenerate(normalized_value, value, high - low)
                elif norm_type == "bounds_999_clip":
                    self._require_stats(stats, key, norm_type, ("q001", "q999"))
                    low = self._stat_like(value, stats["q001"])
                    high = self._stat_like(value, stats["q999"])
                    normalized_value = self._bounds_normalize(value, low, high)
                    if isinstance(normalized_value, torch.Tensor):
                        normalized_value = torch.clamp(normalized_value, -1.0, 1.0)
                    else:
                        normalized_value = np.clip(normalized_value, -1.0, 1.0)
                    normalized_value = self._restore_degenerate(normalized_value, value, high - low)
                elif norm_type == "std":
                    self._require_stats(stats, key, norm_type, ("std",))
                    std = self._stat_like(value, stats["std"])
                    normalized_value = self._restore_degenerate(value / (std + 1e-6), value, std)
                elif norm_type == "minmax":
                    self._require_stats(stats, key, norm_type, ("min", "max"))
                    min_val = self._stat_like(value, stats["min"])
                    max_val = self._stat_like(value, stats["max"])
                    normalized_value = self._bounds_normalize(value, min_val, max_val)
                elif norm_type == "identity":
                    normalized_value = value
                else:
                    raise ValueError(
                        f"Unknown normalization type: {norm_type}. Supported types are 'meanstd', "
                        "'bounds_99_woclip', 'bounds_99_clipX', 'bounds_999_clip', 'std', 'minmax', "
                        "and 'identity'."
                    )
                normalized_data[key] = normalized_value
            else:
                # If the key is not in norm_stats, we assume no normalization is needed
                normalized_data[key] = value
        return normalized_data

    def unnormalize(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Unnormalize the given data using stored normalization statistics.

        Args:
            data (Dict[str, np.ndarray]): Dictionary of normalized arrays to unnormalize.

        Returns:
            Dict[str, np.ndarray]: Dictionary of unnormalized arrays.
        """
        unnormalized_data = {}
        for key, value in data.items():
            if key in self.norm_stats:
                norm_type = self.norm_type.get(key, "identity")
                stats = self.norm_stats[key]
                bounds_99_clip_limit = self._bounds_99_clip_limit(norm_type)
                if norm_type == "meanstd":
                    self._require_stats(stats, key, norm_type, ("mean", "std"))
                    mean = self._stat_like(value, stats["mean"])
                    std = self._stat_like(value, stats["std"])
                    unnormalized_value = self._restore_degenerate(value * (std + 1e-6) + mean, value, std)
                elif norm_type == "bounds_98" or norm_type == "bounds_98_woclip":
                    self._require_stats(stats, key, norm_type, ("q02", "q98"))
                    low = self._stat_like(value, stats["q02"])
                    high = self._stat_like(value, stats["q98"])
                    unnormalized_value = self._bounds_unnormalize(value, low, high)
                elif norm_type == "bounds_99" or norm_type == "bounds_99_woclip":
                    self._require_stats(stats, key, norm_type, ("q01", "q99"))
                    low = self._stat_like(value, stats["q01"])
                    high = self._stat_like(value, stats["q99"])
                    unnormalized_value = self._bounds_unnormalize(value, low, high)
                elif bounds_99_clip_limit is not None:
                    self._require_stats(stats, key, norm_type, ("q01", "q99"))
                    low = self._stat_like(value, stats["q01"])
                    high = self._stat_like(value, stats["q99"])
                    unnormalized_value = self._bounds_unnormalize(value, low, high)
                elif norm_type == "bounds_999_clip":
                    self._require_stats(stats, key, norm_type, ("q001", "q999"))
                    low = self._stat_like(value, stats["q001"])
                    high = self._stat_like(value, stats["q999"])
                    unnormalized_value = self._bounds_unnormalize(value, low, high)
                elif norm_type == "std":
                    self._require_stats(stats, key, norm_type, ("std",))
                    std = self._stat_like(value, stats["std"])
                    unnormalized_value = self._restore_degenerate(value * (std + 1e-6), value, std)
                elif norm_type == "minmax":
                    self._require_stats(stats, key, norm_type, ("min", "max"))
                    min_val = self._stat_like(value, stats["min"])
                    max_val = self._stat_like(value, stats["max"])
                    unnormalized_value = self._bounds_unnormalize(value, min_val, max_val)
                elif norm_type == "identity":
                    unnormalized_value = value
                else:
                    raise ValueError(
                        f"Unknown normalization type: {norm_type}. Supported types are 'meanstd', "
                        "'bounds_98', 'bounds_98_woclip', 'bounds_99', 'bounds_99_woclip', "
                        "'bounds_99_clipX', 'bounds_999_clip', 'std', 'minmax', and 'identity'."
                    )
                unnormalized_data[key] = unnormalized_value
            else:
                # If no normalization was applied, return as-is
                unnormalized_data[key] = value
        return unnormalized_data
