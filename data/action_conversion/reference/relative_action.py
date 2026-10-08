"""Global <-> relative transforms for canonical 49D action vectors.

The policy is trained on actions expressed as *deltas* from the robot state
at the current observation time t. Observation joints stay in their native
coordinates. Left/right EEF state can either stay in the stored common/base
frame or be expressed relative to each observation's head-camera pose; the
active recipe uses the latter for a cross-embodiment task-space input. The
chassis portion of state is always zeroed because its absolute pose has no
intrinsic meaning (it drifts and is frame-dependent).

Canonical 49-D action layout (must match the 49-D dataset contract)::

    0-6   left arm joints        delta: action - state
    7     left gripper           absolute (unchanged)
    8-14  right arm joints       delta: action - state
    15    right gripper          absolute
    16-19 torso                  delta: action - state
    20-22 head                   delta: action - state
    23-25 chassis (x, y, theta)  SE(2) relative (dx_local, dy_local, dtheta)
    26-32 left EEF pose          anchor-camera xyz delta / relative quaternion
    33    left EEF gripper       absolute
    34-40 right EEF pose         anchor-camera xyz delta / relative quaternion
    41    right EEF gripper      absolute
    42-48 head EEF pose          relative xyz/quaternion

All transforms work for both NumPy arrays and PyTorch tensors. They accept
either a scalar state ``(49,)`` paired with a multi-step action ``(T, 49)``
or element-wise matched shapes ``(N, 49)`` / ``(N, 49)``; broadcasting
follows the usual NumPy rules on the trailing ``49`` axis.

Quaternion convention: (qx, qy, qz, qw). Relative quaternions are
canonicalized to the ``qw >= 0`` hemisphere so the training target for
small rotations is unique (avoids the q / -q antipodal ambiguity).

Left/right EEF xyz deltas use the NWU-style head-camera axes at the anchor
time. If ``q_WH(t)`` maps head-camera vectors into the stored common/base NWU
frame, the model target is ``R_WH(t)^T @ (p_action,W - p_state,W)``. The
translation of the head-camera pose cancels because both points are transformed
with the same anchor-time homogeneous transform. Head EEF xyz itself retains
the stored-frame delta representation.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from .action_layout import ActionLayout, resolve_action_layout


# ---------------------------------------------------------------------------
# Canonical 49D layout constants.
# ---------------------------------------------------------------------------

ACTION_DIM = 49

# Joint dims that participate in the delta transform: arms (0-6, 8-14),
# torso (16-19), and head (20-21). Scalar gripper commands at 7 and 15
# are intentionally excluded; they stay absolute.
JOINT_DELTA_DIMS: tuple[int, ...] = tuple(
    list(range(0, 7))  # left arm
    + list(range(8, 15))  # right arm
    + list(range(16, 20))  # torso
    + list(range(20, 23))  # head
)

# Action dims that are never transformed (pass through unchanged). These
# are the four gripper commands (joint-gripper and EEF-gripper on both arms).
ABSOLUTE_DIMS: tuple[int, ...] = (7, 15, 33, 41)

# Tolerance for ``|q| == 1`` on head-camera quaternions. Values are stored as
# float32 and converters round-trip them through rotation matrices, so an exact
# unit norm is unreachable; anything looser than this is a conversion bug.
HEAD_QUAT_NORM_TOL = 1e-3

ArrayLike = Union[np.ndarray, torch.Tensor]
STATE_ARM_EEF_COORDINATES = ("base", "head_camera")


# ---------------------------------------------------------------------------
# Type-dispatched helpers so the rest of the module stays framework-agnostic.
# ---------------------------------------------------------------------------


def _is_torch(x: ArrayLike) -> bool:
    return isinstance(x, torch.Tensor)


def _cos(x):
    return torch.cos(x) if _is_torch(x) else np.cos(x)


def _sin(x):
    return torch.sin(x) if _is_torch(x) else np.sin(x)


def _stack_last(components, like: ArrayLike) -> ArrayLike:
    if _is_torch(like):
        return torch.stack(components, dim=-1)
    return np.stack(components, axis=-1)


def _all_last(x: ArrayLike) -> ArrayLike:
    return x.all(dim=-1) if _is_torch(x) else np.all(x, axis=-1)


def _sum_last(x: ArrayLike) -> ArrayLike:
    return x.sum(dim=-1) if _is_torch(x) else np.sum(x, axis=-1)


# ---------------------------------------------------------------------------
# Quaternion operations (Hamilton product, xyzw convention).
# ---------------------------------------------------------------------------


def quat_conj(q: ArrayLike) -> ArrayLike:
    """Return the conjugate ``(-x, -y, -z, w)``; equals the inverse for unit quats."""
    return _stack_last([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], q)


def quat_mul(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Hamilton product ``a ⊗ b`` for xyzw quaternions; broadcasts on leading dims."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return _stack_last([x, y, z, w], a)


def quat_canonical(q: ArrayLike) -> ArrayLike:
    """Flip sign so ``qw >= 0`` (pick the shortest-rotation representative)."""
    if _is_torch(q):
        sign = torch.where(q[..., 3:4] < 0, -torch.ones_like(q[..., 3:4]), torch.ones_like(q[..., 3:4]))
        return q * sign
    sign = np.where(q[..., 3:4] < 0.0, -1.0, 1.0).astype(q.dtype)
    return q * sign


def quat_rotate(q: ArrayLike, vector: ArrayLike) -> ArrayLike:
    """Rotate xyz ``vector`` by unit xyzw quaternion ``q``.

    Leading dimensions follow NumPy/PyTorch broadcasting rules. Dataset
    validation is responsible for ensuring that valid quaternions are finite
    and normalized.
    """
    if _is_torch(q) != _is_torch(vector):
        raise TypeError("q and vector must use the same array type.")
    zero = vector[..., 0] * 0
    pure = _stack_last([vector[..., 0], vector[..., 1], vector[..., 2], zero], vector)
    return quat_mul(quat_mul(q, pure), quat_conj(q))[..., :3]


def check_anchor_head_quaternion(
    state: ArrayLike,
    state_valid: ArrayLike | None = None,
    action_layout: ActionLayout | dict | None = None,
    *,
    context: str = "",
) -> None:
    """Raise if a head-camera quaternion the dataset calls valid is not unit-norm.

    Hand xyz action targets are rotated by this quaternion, so an unnormalized
    value silently rescales every delta by ``|q|^2`` and an all-zero group
    written as a fake identity silently collapses the whole chunk to zero.
    Both are dataset bugs that must fail loudly rather than produce
    plausible-looking targets.

    Dimensions the dataset marks invalid are skipped: those are stored as zero
    on purpose, and :func:`propagate_relative_action_validity` already drops the
    hand xyz targets that depend on them.
    """
    layout = resolve_action_layout(action_layout)
    if layout.head_eef_quat is None:
        return
    quaternion = state[..., layout.head_eef_quat]
    deviation = abs(_sum_last(quaternion * quaternion) ** 0.5 - 1.0)
    if state_valid is not None:
        deviation = deviation * _all_last(state_valid[..., layout.head_eef_quat])
    worst = float(deviation.max())
    if worst > HEAD_QUAT_NORM_TOL:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}head EEF quaternion is marked valid but is not normalized "
            f"(max |norm(q) - 1| = {worst:.3e} > {HEAD_QUAT_NORM_TOL:.1e}). The 49D hand xyz "
            "action target is rotated by this quaternion, so an identity head-camera pose must "
            "be stored as (0, 0, 0, 1); an all-zero group requires state_dim_mask[42:49] = false."
        )


# ---------------------------------------------------------------------------
# Forward / inverse action-frame transforms.
# ---------------------------------------------------------------------------


def _copy(x: ArrayLike) -> ArrayLike:
    return x.clone() if _is_torch(x) else x.copy()


def resolve_state_arm_eef_coordinate(value: str) -> str:
    """Validate the configured coordinate frame for left/right EEF state."""
    coordinate = str(value)
    if coordinate not in STATE_ARM_EEF_COORDINATES:
        raise ValueError(f"state_arm_eef_coordinate must be one of {STATE_ARM_EEF_COORDINATES}, got {value!r}.")
    return coordinate


def transform_state_arm_eef_coordinate(
    state: ArrayLike,
    coordinate: str,
    action_layout: ActionLayout | dict | None = None,
) -> ArrayLike:
    """Express left/right EEF state in ``base`` or per-frame ``head_camera`` coordinates.

    The stored head EEF pose is ``T_WH`` (head camera in the common/base NWU
    frame), while each arm pose is ``T_WE``. For ``head_camera`` this applies
    ``T_HE = T_WH^-1 T_WE`` independently at every state-history frame. The
    head EEF pose itself is deliberately preserved because it remains the
    authoritative camera-frame anchor for action transforms and deployment.
    """
    coordinate = resolve_state_arm_eef_coordinate(coordinate)
    out = _copy(state)
    if coordinate == "base":
        return out

    layout = resolve_action_layout(action_layout)
    if layout.head_eef_pos is None or layout.head_eef_quat is None:
        raise ValueError("state_arm_eef_coordinate='head_camera' requires a head EEF xyz+quaternion pose.")

    head_pos = state[..., layout.head_eef_pos]
    q_world_from_head = state[..., layout.head_eef_quat]
    q_head_from_world = quat_conj(q_world_from_head)
    for pos, quat in (
        (layout.left_eef_pos, layout.left_eef_quat),
        (layout.right_eef_pos, layout.right_eef_quat),
    ):
        out[..., pos] = quat_rotate(q_head_from_world, state[..., pos] - head_pos)
        out[..., quat] = quat_canonical(quat_mul(q_head_from_world, state[..., quat]))
    return out


def propagate_state_arm_eef_validity(
    state_valid: ArrayLike,
    coordinate: str,
    action_layout: ActionLayout | dict | None = None,
) -> ArrayLike:
    """Propagate head-camera dependencies into transformed arm-EEF state masks."""
    coordinate = resolve_state_arm_eef_coordinate(coordinate)
    out = _copy(state_valid).to(torch.bool) if _is_torch(state_valid) else state_valid.astype(bool, copy=True)
    if coordinate == "base":
        return out

    layout = resolve_action_layout(action_layout)
    if layout.head_eef_pos is None or layout.head_eef_quat is None:
        raise ValueError("state_arm_eef_coordinate='head_camera' requires a head EEF xyz+quaternion pose.")

    head_pose = slice(layout.head_eef_pos.start, layout.head_eef_quat.stop)
    head_pose_valid = _all_last(out[..., head_pose])
    head_quat_valid = _all_last(out[..., layout.head_eef_quat])
    for pos, quat in (
        (layout.left_eef_pos, layout.left_eef_quat),
        (layout.right_eef_pos, layout.right_eef_quat),
    ):
        pos_valid = _all_last(out[..., pos]) & head_pose_valid
        quat_valid = _all_last(out[..., quat]) & head_quat_valid
        out[..., pos] = pos_valid[..., None]
        out[..., quat] = quat_valid[..., None]
    return out


def to_relative_action(
    state: ArrayLike,
    action: ArrayLike,
    action_layout: ActionLayout | dict | None = None,
) -> ArrayLike:
    """Convert a global-frame action (chunk) to the relative frame anchored at ``state``.

    ``state`` is the global-coordinate robot state at time t; ``action`` is
    either a single global-frame action ``(49,)`` / ``(N, 49)`` or an
    action chunk ``(T, 49)``. The returned tensor has the same shape and
    dtype as ``action``.
    """
    layout = resolve_action_layout(action_layout)
    out = _copy(action)

    # -- Joint delta (arms + torso + head): simple per-dim subtraction.
    joint_delta_dims = [d for d in range(layout.joint.start, layout.joint.stop) if d not in layout.gripper_dims]
    for d in joint_delta_dims:
        out[..., d] = action[..., d] - state[..., d]

    # -- Chassis SE(2): world -> body-frame at time t.
    x_t = state[..., layout.chassis_x]
    y_t = state[..., layout.chassis_y]
    theta_t = state[..., layout.chassis_angle]
    cos_t = _cos(theta_t)
    sin_t = _sin(theta_t)
    dx_g = action[..., layout.chassis_x] - x_t
    dy_g = action[..., layout.chassis_y] - y_t
    out[..., layout.chassis_x] = cos_t * dx_g + sin_t * dy_g
    out[..., layout.chassis_y] = -sin_t * dx_g + cos_t * dy_g
    out[..., layout.chassis_angle] = action[..., layout.chassis_angle] - theta_t

    # -- Left/right EEF position: delta expressed in the anchor head-camera
    # NWU axes. Legacy layouts without a head pose retain base-frame deltas.
    for pos in (layout.left_eef_pos, layout.right_eef_pos):
        delta_world = action[..., pos] - state[..., pos]
        if layout.head_eef_quat is None:
            out[..., pos] = delta_world
        else:
            q_world_from_camera = state[..., layout.head_eef_quat]
            out[..., pos] = quat_rotate(quat_conj(q_world_from_camera), delta_world)

    # The head pose defines the anchor camera frame; its own translation delta
    # remains in the stored common/base NWU frame.
    if layout.head_eef_pos is not None:
        out[..., layout.head_eef_pos] = action[..., layout.head_eef_pos] - state[..., layout.head_eef_pos]

    # -- EEF orientation (left, right): relative rotation q_rel = q_t^{-1} * q_a.
    quaternions = [layout.left_eef_quat, layout.right_eef_quat]
    if layout.head_eef_quat is not None:
        quaternions.append(layout.head_eef_quat)
    for quat in quaternions:
        q_t = state[..., quat]
        q_a = action[..., quat]
        q_rel = quat_mul(quat_conj(q_t), q_a)
        out[..., quat] = quat_canonical(q_rel)

    # Absolute gripper dims are left untouched by _copy.
    return out


def propagate_relative_action_validity(
    action_valid: ArrayLike,
    anchor_state_valid: ArrayLike,
    action_layout: ActionLayout | dict | None = None,
) -> ArrayLike:
    """Propagate validity through :func:`to_relative_action`.

    Delta dimensions depend on both the target action and the anchor state.
    Chassis SE(2), EEF xyz, and EEF quaternion slices are atomic groups, so
    one invalid component invalidates the entire group. Left/right EEF xyz
    additionally depends on the anchor head-camera pose that defines its axes.
    Absolute gripper dimensions depend only on the target action and therefore
    do not inherit anchor-state invalidity.

    The function supports both NumPy arrays and PyTorch tensors. A scalar
    anchor mask ``(49,)`` may be paired with an action-chunk mask ``(T, 49)``.
    """
    if _is_torch(action_valid) != _is_torch(anchor_state_valid):
        raise TypeError("action_valid and anchor_state_valid must use the same array type.")

    layout = resolve_action_layout(action_layout)
    out = _copy(action_valid).to(torch.bool) if _is_torch(action_valid) else action_valid.astype(bool, copy=True)
    anchor = anchor_state_valid.to(torch.bool) if _is_torch(anchor_state_valid) else anchor_state_valid.astype(bool)
    if anchor.ndim == out.ndim - 1:
        anchor = anchor[..., None, :]
    absolute = set(layout.gripper_dims)

    for dim in range(layout.joint.start, layout.joint.stop):
        if dim not in absolute:
            out[..., dim] &= anchor[..., dim]

    groups = [layout.chassis_se2, layout.left_eef_quat, layout.right_eef_quat]
    if layout.head_eef_pos is not None:
        groups.extend([layout.head_eef_pos, layout.head_eef_quat])
    for group in groups:
        valid = _all_last(out[..., group]) & _all_last(anchor[..., group])
        out[..., group] = valid[..., None]

    # Camera-frame hand translation requires the target hand xyz, anchor hand
    # xyz, and the complete anchor head-camera pose. Requiring the full pose
    # mirrors the homogeneous-transform contract even though its translation
    # cancels algebraically after subtraction.
    for group in (layout.left_eef_pos, layout.right_eef_pos):
        valid = _all_last(out[..., group]) & _all_last(anchor[..., group])
        if layout.head_eef_pos is not None:
            head_pose = slice(layout.head_eef_pos.start, layout.head_eef_quat.stop)
            valid &= _all_last(anchor[..., head_pose])
        out[..., group] = valid[..., None]
    return out


def to_global_action(
    state: ArrayLike,
    rel_action: ArrayLike,
    action_layout: ActionLayout | dict | None = None,
) -> ArrayLike:
    """Inverse of :func:`to_relative_action`. Reconstructs global-frame actions.

    Used at inference time to emit predictions that a controller can execute
    directly (joint targets in global coords, EEF poses in the robot base
    frame, chassis pose composed with the current global pose).
    """
    layout = resolve_action_layout(action_layout)
    out = _copy(rel_action)

    joint_delta_dims = [d for d in range(layout.joint.start, layout.joint.stop) if d not in layout.gripper_dims]
    for d in joint_delta_dims:
        out[..., d] = rel_action[..., d] + state[..., d]

    x_t = state[..., layout.chassis_x]
    y_t = state[..., layout.chassis_y]
    theta_t = state[..., layout.chassis_angle]
    cos_t = _cos(theta_t)
    sin_t = _sin(theta_t)
    dx_l = rel_action[..., layout.chassis_x]
    dy_l = rel_action[..., layout.chassis_y]
    out[..., layout.chassis_x] = cos_t * dx_l - sin_t * dy_l + x_t
    out[..., layout.chassis_y] = sin_t * dx_l + cos_t * dy_l + y_t
    out[..., layout.chassis_angle] = rel_action[..., layout.chassis_angle] + theta_t

    for pos in (layout.left_eef_pos, layout.right_eef_pos):
        delta = rel_action[..., pos]
        if layout.head_eef_quat is not None:
            q_world_from_camera = state[..., layout.head_eef_quat]
            delta = quat_rotate(q_world_from_camera, delta)
        out[..., pos] = delta + state[..., pos]

    if layout.head_eef_pos is not None:
        out[..., layout.head_eef_pos] = rel_action[..., layout.head_eef_pos] + state[..., layout.head_eef_pos]

    quaternions = [layout.left_eef_quat, layout.right_eef_quat]
    if layout.head_eef_quat is not None:
        quaternions.append(layout.head_eef_quat)
    for quat in quaternions:
        q_t = state[..., quat]
        q_rel = rel_action[..., quat]
        q_a = quat_mul(q_t, q_rel)
        out[..., quat] = quat_canonical(q_a)

    return out


def to_mixed_action(
    state: ArrayLike, rel_action: ArrayLike, action_layout: ActionLayout | dict | None = None
) -> ArrayLike:
    """Like :func:`to_global_action`, but keep configured chassis dims relative.

    Joints and end-effector dims are composed with ``state`` back into the global frame, while
    the chassis SE(2) block stays in the body-frame ``(dx_local, dy_local,
    dtheta)`` representation that the model emits. This is what an
    onboard controller typically wants: absolute joint targets and EEF
    poses, plus a chassis command expressed as an incremental motion
    relative to the robot's current heading.
    """
    layout = resolve_action_layout(action_layout)
    out = to_global_action(state, rel_action, action_layout=layout)
    out[..., layout.chassis_se2] = rel_action[..., layout.chassis_se2]
    return out


def zero_chassis_state(state: ArrayLike, action_layout: ActionLayout | dict | None = None) -> ArrayLike:
    """Return a copy of ``state`` with configured chassis dims set to zero.

    Joint and EEF dimensions stay in their original global-coordinate
    values; the model needs those as input to reason about kinematic
    feasibility and workspace limits.
    """
    layout = resolve_action_layout(action_layout)
    out = _copy(state)
    out[..., layout.chassis_se2] = 0.0
    return out
