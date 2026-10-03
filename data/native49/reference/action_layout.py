from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_REQUIRED_LAYOUT_KEYS = (
    "joint",
    "eef",
    "left_arm",
    "right_arm",
    "chassis_xy",
    "chassis_angle",
    "left_eef_pos",
    "right_eef_pos",
)


@dataclass(frozen=True)
class ActionLayout:
    joint: slice
    eef: slice
    left_arm: slice
    right_arm: slice
    chassis_xy: slice
    chassis_angle: int
    left_eef_pos: slice
    right_eef_pos: slice
    gripper_dims: tuple[int, ...]

    @property
    def chassis_x(self) -> int:
        return self.chassis_xy.start

    @property
    def chassis_y(self) -> int:
        return self.chassis_xy.start + 1

    @property
    def chassis_se2(self) -> slice:
        return slice(self.chassis_xy.start, self.chassis_angle + 1)

    @property
    def left_eef_quat(self) -> slice:
        return slice(self.left_eef_pos.stop, self.left_eef_pos.stop + 4)

    @property
    def right_eef_quat(self) -> slice:
        return slice(self.right_eef_pos.stop, self.right_eef_pos.stop + 4)

    @property
    def head_eef_pos(self) -> slice | None:
        # The 0803 layout appends head xyz+quaternion after the right EEF
        # gripper. Legacy 41-D layouts end at the right EEF gripper.
        start = self.right_eef_quat.stop + 1
        return slice(start, start + 3) if self.eef.stop >= start + 7 else None

    @property
    def head_eef_quat(self) -> slice | None:
        pos = self.head_eef_pos
        return None if pos is None else slice(pos.stop, pos.stop + 4)

    def to_config(self) -> dict[str, Any]:
        return {
            "joint": _slice_to_pair(self.joint),
            "eef": _slice_to_pair(self.eef),
            "left_arm": _slice_to_pair(self.left_arm),
            "right_arm": _slice_to_pair(self.right_arm),
            "chassis_xy": _slice_to_pair(self.chassis_xy),
            "chassis_angle": int(self.chassis_angle),
            "left_eef_pos": _slice_to_pair(self.left_eef_pos),
            "right_eef_pos": _slice_to_pair(self.right_eef_pos),
            "gripper_dims": list(self.gripper_dims),
        }


def _slice_to_pair(value: slice) -> list[int]:
    return [int(value.start), int(value.stop)]


def _as_mapping(config: Any) -> Mapping[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return config
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if hasattr(config, "__dict__"):
        return vars(config)
    raise TypeError(f"action_layout must be a mapping-like object, got {type(config).__name__}.")


def _resolve_slice(config: Mapping[str, Any], name: str) -> slice:
    raw = config.get(name)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"data.action_layout.{name} must be [start, stop], got {raw!r}.")
    start, stop = int(raw[0]), int(raw[1])
    if start < 0 or stop <= start:
        raise ValueError(f"data.action_layout.{name} must satisfy 0 <= start < stop, got {raw!r}.")
    return slice(start, stop)


def resolve_action_layout(config: Any = None) -> ActionLayout:
    if isinstance(config, ActionLayout):
        return config
    cfg = _as_mapping(config)
    missing = [key for key in _REQUIRED_LAYOUT_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"data.action_layout is missing required key(s): {missing}.")
    eef = _resolve_slice(cfg, "eef")
    default_gripper_dims = (7, 15, 33, 41) if eef.stop >= 49 else (7, 15, 32, 40)
    layout = ActionLayout(
        joint=_resolve_slice(cfg, "joint"),
        eef=eef,
        left_arm=_resolve_slice(cfg, "left_arm"),
        right_arm=_resolve_slice(cfg, "right_arm"),
        chassis_xy=_resolve_slice(cfg, "chassis_xy"),
        chassis_angle=int(cfg["chassis_angle"]),
        left_eef_pos=_resolve_slice(cfg, "left_eef_pos"),
        right_eef_pos=_resolve_slice(cfg, "right_eef_pos"),
        gripper_dims=tuple(int(dim) for dim in cfg.get("gripper_dims", default_gripper_dims)),
    )
    if layout.chassis_xy.stop - layout.chassis_xy.start != 2:
        raise ValueError(
            "data.action_layout.chassis_xy must contain exactly two dims "
            f"(x, y), got {layout.chassis_xy.start}:{layout.chassis_xy.stop}."
        )
    if layout.chassis_angle < 0:
        raise ValueError(f"data.action_layout.chassis_angle must be >= 0, got {layout.chassis_angle}.")
    for name, value in (("left_eef_pos", layout.left_eef_pos), ("right_eef_pos", layout.right_eef_pos)):
        if value.stop - value.start != 3:
            raise ValueError(
                f"data.action_layout.{name} must contain exactly three xyz dims, got {value.start}:{value.stop}."
            )
    if any(dim < 0 for dim in layout.gripper_dims) or len(set(layout.gripper_dims)) != len(layout.gripper_dims):
        raise ValueError(
            f"data.action_layout.gripper_dims must contain unique non-negative indices, got {layout.gripper_dims}."
        )
    return layout


def slice_stop(value: slice) -> int:
    return int(value.stop)


def max_required_dim(*items: slice | int) -> int:
    max_dim = 0
    for item in items:
        if isinstance(item, slice):
            max_dim = max(max_dim, int(item.stop))
        else:
            max_dim = max(max_dim, int(item) + 1)
    return max_dim
