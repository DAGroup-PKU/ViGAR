# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Startup and ongoing health checks for goal-conditioned training.

The callback is intentionally fail-fast.  It audits the optimizer's trainable
allowlist at train start, then checks the goal-role embedding gradient before
the first optimizer update.  It proves a real parameter update at the first
positive-learning-rate optimizer window, allowing a short zero-LR scheduler
startup.  The same gradient check also runs immediately after backward, before
callbacks such as finite-gradient sanitizers can modify the gradient.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
import wandb
from torch import nn

from cosmos_framework.utils import log, misc
from cosmos_framework.utils.callback import Callback


def _training_network(model: nn.Module) -> nn.Module:
    """Return the regular network, excluding a sibling EMA copy when present."""

    network = getattr(model, "net", model)
    if not isinstance(network, nn.Module):
        raise RuntimeError("goal-conditioning health check could not resolve model.net")
    return network


def _job_wandb_mode(config: Any) -> str | None:
    """Read ``job.wandb_mode`` from an object or mapping config."""

    if config is None:
        return None
    job = config.get("job") if isinstance(config, Mapping) else getattr(config, "job", None)
    if job is None:
        return None
    mode = job.get("wandb_mode") if isinstance(job, Mapping) else getattr(job, "wandb_mode", None)
    return str(mode).lower() if mode is not None else None


def _normalized_wandb_mode(value: Any) -> str | None:
    if value is None:
        return None
    value = getattr(value, "value", value)
    return str(value).lower()


def validate_online_wandb_run(config: Any, run: Any) -> None:
    """Require an initialized, positively identified online W&B run.

    W&B versions expose offline state either through ``run.offline`` or
    ``run.settings.mode`` (older releases use ``run._settings``).  If the job
    explicitly requests online mode, an absent or ambiguous run is unsafe: an
    authentication fallback must not silently turn a formal run into offline
    training.
    """

    if _job_wandb_mode(config) != "online":
        return
    if run is None:
        raise RuntimeError("job.wandb_mode='online' but wandb.run is not initialized")

    offline_flag = getattr(run, "offline", None)
    settings = getattr(run, "settings", None)
    if settings is None:
        settings = getattr(run, "_settings", None)
    settings_mode = None
    if settings is not None:
        settings_mode = settings.get("mode") if isinstance(settings, Mapping) else getattr(settings, "mode", None)
    normalized_mode = _normalized_wandb_mode(settings_mode)

    if offline_flag is True or normalized_mode in {"offline", "disabled", "dryrun"}:
        raise RuntimeError(
            "job.wandb_mode='online' but the initialized W&B run is offline "
            f"(run.offline={offline_flag!r}, settings.mode={normalized_mode!r})"
        )

    online_confirmed = offline_flag is False or normalized_mode == "online"
    if not online_confirmed:
        raise RuntimeError(
            "job.wandb_mode='online' but the initialized W&B run mode could not be confirmed "
            f"(run.offline={offline_flag!r}, settings.mode={normalized_mode!r})"
        )


def _validate_online_wandb_run_on_rank0(model: nn.Module, config: Any) -> None:
    """Validate rank-0's W&B run and make every distributed rank fail together."""

    if _job_wandb_mode(config) != "online":
        return

    local_error: str | None = None
    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        try:
            validate_online_wandb_run(config, wandb.run)
        except RuntimeError as error:
            local_error = str(error)

    if dist.is_available() and dist.is_initialized():
        network = _training_network(model)
        parameter = next(network.parameters(), None)
        if parameter is None:
            raise RuntimeError("cannot synchronize W&B health without a model parameter")
        device = misc.get_local_tensor_if_DTensor(parameter).device
        healthy = torch.tensor([int(local_error is None)], device=device, dtype=torch.int32)
        dist.all_reduce(healthy, op=dist.ReduceOp.MIN)
        if bool(healthy.item()):
            return
        if local_error is None:
            local_error = "rank 0 reported that the requested online W&B run is unavailable or offline"

    if local_error is not None:
        raise RuntimeError("W&B online health check failed: " + local_error)


def audit_trainable_parameter_allowlist(
    model: nn.Module,
    trainable_allowlist: Sequence[str],
    *,
    goal_parameter_suffix: str = "goal_vision_embed",
) -> tuple[str, nn.Parameter, list[str]]:
    """Validate that every trainable tensor matches the explicit allowlist.

    This mirrors the substring semantics used by the VFM optimizer.  Auditing
    the resulting ``requires_grad`` state after optimizer construction makes
    the Reasoner-freeze invariant observable instead of relying on config
    intent alone.
    """

    allowlist = tuple(str(pattern) for pattern in trainable_allowlist if str(pattern))
    if not allowlist:
        raise RuntimeError("goal-conditioning health check requires a non-empty trainable allowlist")

    network = _training_network(model)
    named_parameters = list(network.named_parameters())
    goal_matches = [(name, param) for name, param in named_parameters if name.endswith(goal_parameter_suffix)]
    if len(goal_matches) != 1:
        names = [name for name, _ in goal_matches]
        raise RuntimeError(
            f"expected exactly one regular-network parameter ending in {goal_parameter_suffix!r}; found {names}"
        )

    goal_name, goal_parameter = goal_matches[0]
    if not goal_parameter.requires_grad:
        raise RuntimeError(f"goal parameter {goal_name!r} is frozen")

    trainable_names = [name for name, param in named_parameters if param.requires_grad]
    unexpected = [name for name in trainable_names if not any(pattern in name for pattern in allowlist)]
    if unexpected:
        preview = ", ".join(unexpected[:20])
        remainder = len(unexpected) - min(20, len(unexpected))
        suffix = f" (+{remainder} more)" if remainder else ""
        raise RuntimeError(
            "trainable-parameter allowlist audit failed; unexpected trainable tensors: " + preview + suffix
        )

    if not any(pattern in goal_name for pattern in allowlist):
        raise RuntimeError(f"goal parameter {goal_name!r} is not covered by the trainable allowlist")

    return goal_name, goal_parameter, trainable_names


def _optimizer_parameters(optimizer: Any) -> Iterable[nn.Parameter]:
    """Yield parameters from a PyTorch optimizer or OptimizersContainer."""

    inner_optimizers = getattr(optimizer, "optimizers", None)
    optimizers = inner_optimizers if inner_optimizers is not None else (optimizer,)
    for inner in optimizers:
        for group in getattr(inner, "param_groups", ()):  # pragma: no branch - defensive for callback misuse
            yield from group.get("params", ())


def optimizer_contains_parameter(optimizer: Any, parameter: nn.Parameter) -> bool:
    """Return whether ``parameter`` is owned by any optimizer parameter group."""

    return any(candidate is parameter for candidate in _optimizer_parameters(optimizer))


def parameter_effective_learning_rate(optimizer: Any, parameter: nn.Parameter) -> float:
    """Return the learning rate of the unique group owning ``parameter``."""

    matching_groups: list[Mapping[str, Any]] = []
    inner_optimizers = getattr(optimizer, "optimizers", None)
    optimizers = inner_optimizers if inner_optimizers is not None else (optimizer,)
    for inner in optimizers:
        for group in getattr(inner, "param_groups", ()):
            if any(candidate is parameter for candidate in group.get("params", ())):
                matching_groups.append(group)

    if len(matching_groups) != 1:
        raise RuntimeError(
            "goal parameter must belong to exactly one optimizer parameter group; "
            f"found {len(matching_groups)}"
        )

    learning_rate = matching_groups[0].get("lr")
    if isinstance(learning_rate, torch.Tensor):
        if learning_rate.numel() != 1:
            raise RuntimeError("goal optimizer parameter-group learning rate must be scalar")
        learning_rate = learning_rate.detach().item()
    try:
        learning_rate = float(learning_rate)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"goal optimizer parameter-group learning rate is invalid: {learning_rate!r}"
        ) from error
    if not math.isfinite(learning_rate) or learning_rate < 0.0:
        raise RuntimeError(
            "goal optimizer parameter-group learning rate must be finite and non-negative; "
            f"got {learning_rate!r}"
        )
    return learning_rate


def validate_parameter_gradient(parameter: nn.Parameter, parameter_name: str) -> float:
    """Require a globally present, finite, non-zero gradient and return its L1 sum.

    For DTensor/FSDP parameters the check operates on local shards and reduces
    health flags plus the absolute sum across the default process group.  All
    ranks therefore pass or fail together.
    """

    local_parameter = misc.get_local_tensor_if_DTensor(parameter)
    device = local_parameter.device
    gradient = parameter.grad
    has_gradient = gradient is not None

    if has_gradient:
        local_gradient = misc.get_local_tensor_if_DTensor(gradient).detach()
        local_finite = bool(torch.isfinite(local_gradient).all().item())
        local_abs_sum = local_gradient.float().abs().sum()
    else:
        local_finite = True
        local_abs_sum = torch.zeros((), device=device, dtype=torch.float32)

    flags = torch.tensor(
        [int(has_gradient), int(local_finite)],
        device=device,
        dtype=torch.int32,
    )
    global_abs_sum = local_abs_sum.to(device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        dist.all_reduce(global_abs_sum, op=dist.ReduceOp.SUM)

    if not bool(flags[0].item()):
        raise RuntimeError(f"goal parameter {parameter_name!r} gradient is None on at least one rank")
    if not bool(flags[1].item()) or not bool(torch.isfinite(global_abs_sum).item()):
        raise RuntimeError(f"goal parameter {parameter_name!r} gradient is non-finite")

    abs_sum = float(global_abs_sum.item())
    if abs_sum == 0.0:
        raise RuntimeError(f"goal parameter {parameter_name!r} gradient is exactly zero")
    return abs_sum


def validate_parameter_update(
    parameter: nn.Parameter,
    snapshot: torch.Tensor,
    parameter_name: str,
) -> float:
    """Require a finite parameter and globally non-zero update from ``snapshot``."""

    current = misc.get_local_tensor_if_DTensor(parameter).detach().float()
    if current.shape != snapshot.shape:
        raise RuntimeError(
            f"goal parameter {parameter_name!r} local shape changed across optimizer step: "
            f"{tuple(snapshot.shape)} -> {tuple(current.shape)}"
        )

    local_finite = torch.isfinite(current).all().to(dtype=torch.int32).reshape(1)
    local_abs_delta = (current - snapshot.to(device=current.device, dtype=current.dtype)).abs().sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_finite, op=dist.ReduceOp.MIN)
        dist.all_reduce(local_abs_delta, op=dist.ReduceOp.SUM)

    if not bool(local_finite.item()) or not bool(torch.isfinite(local_abs_delta).item()):
        raise RuntimeError(f"goal parameter {parameter_name!r} became non-finite after optimizer step")
    abs_delta = float(local_abs_delta.item())
    if abs_delta == 0.0:
        raise RuntimeError(f"goal parameter {parameter_name!r} did not update at positive effective LR")
    return abs_delta


def _global_max_gpu_memory_gib(parameter: nn.Parameter) -> tuple[float, float]:
    """Return all-rank max allocated/reserved CUDA memory in GiB."""

    device = misc.get_local_tensor_if_DTensor(parameter).device
    if device.type == "cuda":
        gib = float(1024**3)
        local_memory = torch.tensor(
            [
                torch.cuda.max_memory_allocated(device) / gib,
                torch.cuda.max_memory_reserved(device) / gib,
            ],
            device=device,
            dtype=torch.float32,
        )
    else:
        local_memory = torch.zeros(2, device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_memory, op=dist.ReduceOp.MAX)
    return float(local_memory[0].item()), float(local_memory[1].item())


class GoalConditioningHealthCheck(Callback):
    """Fail fast unless goal conditioning is trainable and receives gradient."""

    def __init__(
        self,
        trainable_allowlist: Sequence[str],
        goal_parameter_suffix: str = "goal_vision_embed",
        grad_accum_steps: int = 1,
        max_consecutive_nonfinite_windows: int = 100,
        max_update_check_optimizer_steps: int = 3,
    ) -> None:
        super().__init__()
        if grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
        if max_consecutive_nonfinite_windows < 0:
            raise ValueError(
                "max_consecutive_nonfinite_windows must be >= 0, "
                f"got {max_consecutive_nonfinite_windows}"
            )
        if max_update_check_optimizer_steps < 1:
            raise ValueError(
                "max_update_check_optimizer_steps must be >= 1, "
                f"got {max_update_check_optimizer_steps}"
            )
        self.trainable_allowlist = tuple(trainable_allowlist)
        self.goal_parameter_suffix = goal_parameter_suffix
        self.grad_accum_steps = int(grad_accum_steps)
        self.max_consecutive_nonfinite_windows = int(max_consecutive_nonfinite_windows)
        self.max_update_check_optimizer_steps = int(max_update_check_optimizer_steps)
        self._goal_name: str | None = None
        self._goal_parameter: nn.Parameter | None = None
        self._completed = False
        self._saw_backward = False
        self._backward_calls = 0
        self._awaiting_update = False
        self._goal_snapshot: torch.Tensor | None = None
        self._goal_grad_abs_sum: float | None = None
        self._first_goal_grad_abs_sum: float | None = None
        self._optimizer_windows_seen = 0
        self._window_nonfinite_loss_flag: torch.Tensor | None = None
        self._consecutive_nonfinite_windows = 0

    def on_train_start(self, model: nn.Module, iteration: int = 0) -> None:
        del iteration
        _validate_online_wandb_run_on_rank0(model, getattr(self, "config", None))
        goal_name, goal_parameter, trainable_names = audit_trainable_parameter_allowlist(
            model,
            self.trainable_allowlist,
            goal_parameter_suffix=self.goal_parameter_suffix,
        )
        self._goal_name = goal_name
        self._goal_parameter = goal_parameter
        log.info(
            "[GoalConditioningHealthCheck] trainable allowlist PASS: "
            f"{len(trainable_names)} tensors; goal={goal_name}; Reasoner remains frozen"
        )

    def _resolved_goal_parameter(self) -> tuple[str, nn.Parameter]:
        if self._goal_name is None or self._goal_parameter is None:
            raise RuntimeError("goal-conditioning health check ran before on_train_start")
        return self._goal_name, self._goal_parameter

    def on_before_backward(self, model: nn.Module, loss: torch.Tensor, iteration: int = 0) -> None:
        del model, iteration
        # Keep this entirely on-device.  The optimizer-window callback performs
        # the sole collective and host synchronization for all microbatches.
        nonfinite = (~torch.isfinite(loss.detach()).all()).to(dtype=torch.int32).reshape(1)
        if self._window_nonfinite_loss_flag is None:
            self._window_nonfinite_loss_flag = nonfinite
        else:
            self._window_nonfinite_loss_flag = torch.maximum(
                self._window_nonfinite_loss_flag.to(device=nonfinite.device),
                nonfinite,
            )

    def on_after_backward(self, model: nn.Module, iteration: int = 0) -> None:
        del model, iteration
        if self._completed:
            return
        self._backward_calls += 1
        if self._backward_calls % self.grad_accum_steps != 0:
            return
        goal_name, goal_parameter = self._resolved_goal_parameter()
        # Check the accumulated gradient after the final microbatch, but before
        # finite-gradient sanitizers or clipping can mask a bad backward pass.
        validate_parameter_gradient(goal_parameter, goal_name)
        self._saw_backward = True

    def on_before_optimizer_step(
        self,
        model: nn.Module,
        optimizer: Any,
        scheduler: Any,
        grad_scaler: Any,
        iteration: int = 0,
    ) -> None:
        del model, scheduler, grad_scaler
        goal_name, goal_parameter = self._resolved_goal_parameter()
        local_parameter = misc.get_local_tensor_if_DTensor(goal_parameter)

        if self._window_nonfinite_loss_flag is None:
            nonfinite_flag = torch.zeros(1, device=local_parameter.device, dtype=torch.int32)
        else:
            nonfinite_flag = self._window_nonfinite_loss_flag.to(device=local_parameter.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(nonfinite_flag, op=dist.ReduceOp.MAX)
        window_nonfinite = bool(nonfinite_flag.item())
        self._window_nonfinite_loss_flag = None
        if window_nonfinite:
            self._consecutive_nonfinite_windows += 1
            log.warning(
                "[GoalConditioningHealthCheck] non-finite loss observed on at least one rank "
                f"for optimizer window at iteration={iteration} "
                f"(consecutive={self._consecutive_nonfinite_windows})"
            )
        else:
            self._consecutive_nonfinite_windows = 0

        if (
            self.max_consecutive_nonfinite_windows > 0
            and self._consecutive_nonfinite_windows >= self.max_consecutive_nonfinite_windows
        ):
            raise RuntimeError(
                "Training unstable: non-finite loss on at least one rank for "
                f"{self._consecutive_nonfinite_windows} consecutive optimizer windows "
                f"at iteration {iteration}"
            )

        if self._completed:
            return
        if not self._saw_backward:
            raise RuntimeError("first optimizer step was reached before a goal-gradient backward check")

        if not optimizer_contains_parameter(optimizer, goal_parameter):
            raise RuntimeError(f"goal parameter {goal_name!r} is missing from optimizer parameter groups")

        abs_sum = validate_parameter_gradient(goal_parameter, goal_name)
        self._saw_backward = False
        self._optimizer_windows_seen += 1
        if self._first_goal_grad_abs_sum is None:
            self._first_goal_grad_abs_sum = abs_sum

        effective_lr = parameter_effective_learning_rate(optimizer, goal_parameter)
        positive_lr = torch.tensor(
            [int(effective_lr > 0.0)],
            device=local_parameter.device,
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(positive_lr, op=dist.ReduceOp.SUM)
            positive_rank_count = int(positive_lr.item())
            world_size = dist.get_world_size()
            if positive_rank_count not in {0, world_size}:
                raise RuntimeError("goal optimizer learning-rate positivity differs across ranks")
            has_positive_lr = positive_rank_count == world_size
        else:
            has_positive_lr = bool(positive_lr.item())

        if not has_positive_lr:
            self._awaiting_update = False
            self._goal_snapshot = None
            self._goal_grad_abs_sum = None
            log.warning(
                "[GoalConditioningHealthCheck] goal gradient PASS but effective LR is zero; "
                "deferring parameter-update proof "
                f"(optimizer_window={self._optimizer_windows_seen}/"
                f"{self.max_update_check_optimizer_steps}, iteration={iteration})"
            )
            return

        self._goal_snapshot = local_parameter.detach().float().clone()
        self._goal_grad_abs_sum = abs_sum
        self._awaiting_update = True

    def on_before_zero_grad(
        self,
        model: nn.Module,
        optimizer: Any,
        scheduler: Any,
        iteration: int = 0,
    ) -> None:
        del optimizer, scheduler
        if self._completed:
            return
        if not self._awaiting_update:
            if self._optimizer_windows_seen == 0:
                raise RuntimeError("goal-conditioning update check ran before an optimizer window")
            if self._optimizer_windows_seen >= self.max_update_check_optimizer_steps:
                raise RuntimeError(
                    "could not prove a goal-parameter update within "
                    f"{self.max_update_check_optimizer_steps} optimizer windows because its effective LR remained zero"
                )
            return
        if self._goal_snapshot is None or self._goal_grad_abs_sum is None:
            raise RuntimeError("goal-conditioning update check is missing its pre-optimizer parameter snapshot")

        goal_name, goal_parameter = self._resolved_goal_parameter()
        update_abs_sum = validate_parameter_update(goal_parameter, self._goal_snapshot, goal_name)
        grad_abs_sum = self._first_goal_grad_abs_sum
        if grad_abs_sum is None:
            raise RuntimeError("goal-conditioning update check lost the first-window gradient statistic")
        self._completed = True
        self._awaiting_update = False
        self._goal_snapshot = None

        if _job_wandb_mode(getattr(self, "config", None)) == "online":
            _validate_online_wandb_run_on_rank0(model, getattr(self, "config", None))
            max_allocated_gib, max_reserved_gib = _global_max_gpu_memory_gib(goal_parameter)
            if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                wandb.log(
                    {
                        "health/goal_grad_abs_sum": grad_abs_sum,
                        "health/goal_update_abs_sum": update_abs_sum,
                        "health/max_gpu_memory_allocated_gib": max_allocated_gib,
                        "health/max_gpu_memory_reserved_gib": max_reserved_gib,
                    },
                    step=iteration,
                )

        log.success(
            "[GoalConditioningHealthCheck] first optimizer-step/update PASS: "
            f"{goal_name}.grad is present, finite, and non-zero; parameter updated "
            f"(global_grad_abs_sum={grad_abs_sum:.6e}, "
            f"global_update_abs_sum={update_abs_sum:.6e}, iteration={iteration})"
        )
