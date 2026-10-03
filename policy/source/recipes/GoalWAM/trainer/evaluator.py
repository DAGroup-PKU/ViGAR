"""Reproducibility artifacts and offline evaluation for the current 0824 contract.

All model computation uses native Cosmos methods. These helpers observe tensors,
save/load native DCP states, and evaluate recorded data; they contain no alternate
network, flow-matching objective, optimizer, or inference solver.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import random
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from recipes.GoalWAM.data.data_loader import (
    collate_samples,
    evaluation_loader,
)

from .evaluation_budget import resolve_evaluation_budget


def json_default(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=json_default) + "\n")


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        if hasattr(value, "to_local"):
            value = value.to_local()
        return value.detach().cpu().clone()
    if dataclasses.is_dataclass(value):
        return {f.name: cpu_tree(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if hasattr(value, "sha256") and hasattr(value, "norm_type"):
        return dict(sha256=value.sha256, norm_type=value.norm_type)
    if isinstance(value, dict):
        return {str(k): cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def module_counters(model):
    return {
        name: {
            key: copy.deepcopy(getattr(module, key))
            for key in ("accum_image_sample_counter", "accum_video_sample_counter")
            if hasattr(module, key)
        }
        for name, module in model.named_modules()
    }


@contextmanager
def evaluation_state(model):
    state = rng_state()
    modes = [(module, module.training) for module in model.modules()]
    counters = [
        (module, key, copy.deepcopy(getattr(module, key)))
        for module in model.modules()
        for key in ("accum_image_sample_counter", "accum_video_sample_counter")
        if hasattr(module, key)
    ]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        restore_rng(state)
        for module, mode in modes:
            module.training = mode
        for module, key, value in counters:
            setattr(module, key, value)


def load_base(model, path, output):
    from cosmos_framework.inference.model import (
        _DiffusersHuggingFaceStorageReader,
        _DiffusersLoadPlanner,
    )

    path = Path(path)
    state = get_model_state_dict(model.net)
    fresh = (
        "action2llm",
        "llm2action",
        "action_modality_embed",
        "action_pos_embed",
        "goal_vision_embed",
    )
    skipped = [key for key in state if any(part in key for part in fresh)]
    target = {key: value for key, value in state.items() if key not in skipped}
    # Validate the complete required mapping before touching any tensor.
    planner = _DiffusersLoadPlanner(path)
    _, mapped = planner._build_remapped_state_dict(target)
    missing = set(target) - mapped
    if missing:
        raise ValueError(f"Base checkpoint is missing required tensors: {sorted(missing)}")
    dcp.load(target, storage_reader=_DiffusersHuggingFaceStorageReader(path), planner=planner)
    set_model_state_dict(model.net, state)
    if model.config.ema.enabled:
        model.net_ema_worker.copy_to(src_model=model.net, tgt_model=model.net_ema)
    if dist.get_rank() == 0:
        write_json(
            Path(output) / "base_mapping.json",
            dict(
                source=str(path),
                loaded=sorted(mapped),
                initialized_fresh=sorted(skipped),
                required_missing=[],
            ),
        )


def optimizer_master_state(model, optimizer, *, initialize=False):
    """FusedAdam keeps FP32 master parameters outside Optimizer.state_dict()."""
    names = {id(p): name for name, p in model.named_parameters()}
    state = {}
    for opt in optimizer.optimizers:
        if not getattr(opt, "master_weights", False):
            continue
        if opt.param_groups_master is None:
            if not initialize:
                continue
            opt.param_groups_master = [
                {"params": [p.detach().clone().float() for p in group["params"]]} for group in opt.param_groups
            ]
        for group, masters in zip(opt.param_groups, opt.param_groups_master, strict=True):
            for param, master in zip(group["params"], masters["params"], strict=True):
                state[names[id(param)]] = master
    return state


def save_checkpoint(model, optimizer, scheduler, loader, iteration, path, *, mark_complete=True):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    dcp.save(get_model_state_dict(model), checkpoint_id=path / "model")
    if iteration:
        dcp.save(optimizer.state_dict(), checkpoint_id=path / "optimizer")
        masters = optimizer_master_state(model, optimizer)
        if masters:
            dcp.save(masters, checkpoint_id=path / "optimizer_masters")
    state = dict(
        iteration=iteration,
        rng=rng_state(),
        sampler=loader.state_dict(),
        scheduler=scheduler.state_dict(),
        world_size=dist.get_world_size(),
        module_counters=module_counters(model),
    )
    torch.save(state, path / f"rank_{dist.get_rank():02d}.pt")
    dist.barrier()
    if mark_complete and dist.get_rank() == 0:
        write_json(
            path / "complete.json",
            dict(iteration=iteration, world_size=dist.get_world_size()),
        )


def load_checkpoint(model, optimizer, scheduler, loader, path, *, resume=True):
    path = Path(path)
    if not (path / "complete.json").exists():
        raise ValueError(f"Incomplete reference checkpoint: {path}")
    state = get_model_state_dict(model)
    metadata = dcp.FileSystemReader(path / "model").read_metadata()
    if set(metadata.state_dict_metadata) != set(state):
        raise ValueError("Checkpoint model tensor names do not exactly match the configured model")
    for name, tensor in state.items():
        stored = metadata.state_dict_metadata[name]
        if tensor.shape != stored.size or tensor.dtype != stored.properties.dtype:
            raise ValueError(f"Checkpoint tensor shape/dtype mismatch: {name}")
    dcp.load(state, checkpoint_id=path / "model")
    # OmniMoT requires strict=False at this API boundary and returns its own
    # incompatibility report. Enforce strictness explicitly on that report.
    incompatible = set_model_state_dict(model, state, options=StateDictOptions(strict=False))
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Checkpoint model mapping is incomplete: {incompatible}")
    if resume:
        runtime = torch.load(path / f"rank_{dist.get_rank():02d}.pt", weights_only=False, map_location="cpu")
        if runtime["world_size"] != dist.get_world_size():
            raise ValueError("Exact resume requires the original rank topology")
        if runtime["iteration"]:
            optim = optimizer.state_dict()
            dcp.load(optim, checkpoint_id=path / "optimizer")
            optimizer.load_state_dict(optim)
            # Optimizer.load_state_dict casts moments to the BF16 parameter
            # dtype before FusedAdam casts them back to FP32. Restore the
            # original checkpoint tensors to avoid irreversible rounding.
            names = {id(p): name for name, p in model.named_parameters()}
            for opt in optimizer.optimizers:
                if not getattr(opt, "master_weights", False):
                    continue
                for group in opt.param_groups:
                    for param in group["params"]:
                        for key in ("exp_avg", "exp_avg_sq"):
                            name = (
                                names[id(param)].replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
                            )
                            value = optim[f"state.{name}.{key}"]
                            if value.dtype != torch.float32:
                                raise ValueError("Native FusedAdam checkpoint moments must be FP32")
                            opt.state[param][key] = value
            masters = optimizer_master_state(model, optimizer, initialize=True)
            if masters:
                dcp.load(masters, checkpoint_id=path / "optimizer_masters")
        scheduler.load_state_dict(runtime["scheduler"])
        loader.load_state_dict(runtime["sampler"])
        modules = dict(model.named_modules())
        for name, counters in runtime.get("module_counters", {}).items():
            for key, value in counters.items():
                setattr(modules[name], key, value)
        restore_rng(runtime["rng"])
    return runtime["iteration"] if resume else 0


def parameter_manifest(model):
    result = {}
    for name, param in model.net.named_parameters():
        value = cpu_tree(param).contiguous()
        result[name] = dict(
            shape=list(param.shape),
            local_shape=list(value.shape),
            dtype=str(param.dtype),
            trainable=param.requires_grad,
            sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        )
    return result


class Trace:
    """Observe native inputs/predictions without changing RNG or arithmetic."""

    def __init__(self, model):
        self.model, self.data = model, {}
        self.original = model._compute_losses

        def observed(**kwargs):
            self.data = cpu_tree(kwargs)
            return self.original(**kwargs)

        model._compute_losses = observed

    def close(self):
        self.model._compute_losses = self.original


def selected_parameters(model, *, gradients=False):
    prefixes = (
        "goal_vision_embed",
        "action2llm",
        "llm2action",
        "moe_gen",
        "vae2llm",
        "llm2vae",
    )
    result, seen = {}, set()
    for name, param in model.net.named_parameters():
        group = next((p for p in prefixes if p in name), None)
        if group is None or group in seen:
            continue
        value = param.grad if gradients else param
        if value is not None:
            result[name] = cpu_tree(value).flatten()[:4096]
            seen.add(group)
    return result


def selected_optimizer_state(model, optimizer):
    selected = set(selected_parameters(model))
    names = {id(p): name for name, p in model.net.named_parameters()}
    masters = optimizer_master_state(model, optimizer)
    result = {}
    for opt in optimizer.optimizers:
        for group in opt.param_groups:
            for param in group["params"]:
                name = names[id(param)]
                if name not in selected or not opt.state.get(param):
                    continue
                values = dict(opt.state[param])
                values["master"] = masters[f"net.{name}"]
                result[name] = {
                    key: cpu_tree(
                        value.to_local().flatten()[:4096] if hasattr(value, "to_local") else value.flatten()[:4096]
                    )
                    for key, value in values.items()
                }
                result[name]["step"] = cpu_tree(group["step"])
    return result


def inference_sample(sample):
    result = dict(sample)
    plan = sample["sequence_plan"]
    if list(plan.condition_frame_indexes_action) != [0] or list(plan.condition_frame_indexes_vision) != [0]:
        raise ValueError("Only the current state and current rollout observation may condition generation")
    action = torch.zeros_like(sample["action"])
    action[0] = sample["action"][0]
    rollout = torch.zeros_like(sample["video"][-1])
    rollout[:, 0] = sample["video"][-1][:, 0]
    result["action"] = action
    result["video"] = [*sample["video"][:-1], rollout]
    return result


def wandb_evaluation_tables(output, rows, iteration, weights, max_rows):
    """Build per-robot W&B tables from the bounded local visualization artifacts."""
    if max_rows <= 0:
        return {}

    import wandb

    output = Path(output)
    selected = [row for row in rows if row.get("media")][:max_rows]
    columns = [
        "step",
        "dataset",
        "episode",
        "window_start",
        "task_text",
        "conditioning",
        "gt_pred_future_frames",
        "gt_pred_future_video",
        "left_eef_3d",
        "right_eef_3d",
        "chassis_bev",
    ]

    def image(media, name):
        filename = media.get(name)
        return wandb.Image(str(output / filename)) if filename else None

    tables = {}
    for robot in sorted({row["robot_type"] for row in selected}):
        table_rows = []
        for row in selected:
            if row["robot_type"] != robot:
                continue
            media = row["media"]
            video = media.get("future_video")
            table_rows.append(
                [
                    iteration,
                    row["dataset"],
                    row["episode"],
                    row["start"],
                    row["caption"],
                    image(media, "conditioning"),
                    image(media, "future_frames"),
                    wandb.Video(str(output / video), format="mp4") if video else None,
                    image(media, "left_eef_3d"),
                    image(media, "right_eef_3d"),
                    image(media, "chassis_bev"),
                ]
            )
        tables[f"eval/{robot}/{weights}/generation_table"] = wandb.Table(columns=columns, data=table_rows)
    return tables


@torch.no_grad()
def evaluate(
    model,
    dataset,
    output,
    iteration,
    count=32,
    visual_count=8,
    weights="regular",
    *,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=1,
    selection_seed=20260824,
    gen_num_steps=5,
    gen_guidance=3.0,
    gen_shift=5.0,
    loss_seed=8000,
    gen_seed=9000,
    record_artifacts=True,
    eval_loss_per_rank=None,
    generation_per_rank=None,
    generation_wandb_max=None,
):
    from cosmos_framework.data.vfm.action.action_processing import ActionProcessingRecord
    from cosmos_framework.model.vfm.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig
    from cosmos_framework.utils import misc

    from recipes.GoalWAM.data.relative_action import to_global_action

    from .metrics import aggregate_metrics, image_metrics, physical_action_metrics
    from .visualization import save_sample_visuals, write_gallery

    output = Path(output) / f"iter_{iteration:09d}" / weights
    output.mkdir(parents=True, exist_ok=True)
    rank, world = dist.get_rank(), dist.get_world_size()
    budget = resolve_evaluation_budget(
        world,
        count=count,
        visual_count=visual_count,
        eval_loss_per_rank=eval_loss_per_rank,
        generation_per_rank=generation_per_rank,
        generation_wandb_max=generation_wandb_max,
    )
    visual_count = budget.visual_max
    raw = dataset._dataset
    loader = evaluation_loader(
        dataset,
        budget.windows_per_rank * world,
        rank,
        world,
        seed=selection_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    rows = []
    iterator = None
    with evaluation_state(model):
        scope = model.ema_scope(context="0824_eval", is_cpu=True) if weights == "ema" else nullcontext()
        with scope:
            try:
                iterator = iter(loader)
                for local_index, item in enumerate(iterator):
                    sample_id, index = item["sample_id"], item["index"]
                    physical, sample = item["physical"], item["sample"]
                    metrics = {}
                    record = dict(
                        sample_id=sample_id,
                        flat_index=index,
                        dataset=physical["dataset"],
                        robot_type=physical["robot_type"],
                        episode=physical["episode_index"],
                        start=physical["start"],
                        caption=item["caption"],
                        metrics=metrics,
                    )
                    losses = None
                    if local_index < budget.loss_per_rank:
                        seed_all(loss_seed + sample_id)
                        trace = Trace(model) if record_artifacts else None
                        rng = rng_state() if record_artifacts else None
                        try:
                            losses, loss = model.training_step(
                                misc.to(collate_samples([sample]), device="cuda"), iteration
                            )
                        finally:
                            if trace:
                                trace.close()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite GoalWAM evaluation loss")
                        metrics.update(
                            loss=float(loss),
                            action_fm_loss=float(losses["flow_matching_loss_action"]),
                            video_fm_loss=float(losses["flow_matching_loss_vision"]),
                            action_fm_weighted=10 * float(losses["flow_matching_loss_action"]),
                            video_fm_weighted=10 * float(losses["flow_matching_loss_vision"]),
                        )
                        if record_artifacts:
                            torch.save(
                                dict(trace=trace.data, outputs=cpu_tree(losses), loss=cpu_tree(loss), rng=rng),
                                output / f"sample_{sample_id:03d}_forward.pt",
                            )
                    if local_index >= budget.generation_per_rank:
                        rows.append(record)
                        print(json.dumps(record), flush=True)
                        continue
                    seed_all(gen_seed + sample_id)
                    sampler = UniPCSampler(cfg=UniPCSamplerConfig(), tensor_kwargs=model.tensor_kwargs)
                    denoise_calls = []
                    original_denoise = model.denoise

                    def observed_denoise(*args, **kwargs):
                        result = original_denoise(*args, **kwargs)
                        denoise_calls.append(dict(input=cpu_tree(kwargs), output=cpu_tree(result)))
                        return result

                    if record_artifacts:
                        model.denoise = observed_denoise
                    inference = inference_sample(sample)
                    inference["action_processing_record"] = ActionProcessingRecord(49, None)
                    try:
                        generated = model.generate_samples_from_batch(
                            misc.to(collate_samples([inference]), device="cuda"),
                            sampler=sampler,
                            seed=[gen_seed + sample_id],
                            num_steps=gen_num_steps,
                            guidance=gen_guidance,
                            shift=gen_shift,
                        )
                    finally:
                        model.denoise = original_denoise
                    normalized = generated["action"][0][1:].float().cpu()
                    mask = physical["relative_valid"]
                    if normalized.shape != (48, 49) or not torch.isfinite(normalized).all():
                        raise ValueError("Invalid generated action chunk")
                    normalizer = raw.normalizers[physical["robot_type"]]
                    relative = normalizer.denormalize_action(normalized).masked_fill(~mask, 0)
                    absolute = to_global_action(physical["anchor_state"], relative, action_layout=raw.layout)
                    absolute = absolute.masked_fill(~mask, 0)
                    latent = generated["vision"][0]
                    decoded = model.decode(latent.unsqueeze(0) if latent.ndim == 4 else latent).float().cpu()
                    if decoded.ndim == 5:
                        decoded = decoded[0]
                    target_video = item["target_video"]
                    if decoded.shape != target_video.shape or not torch.isfinite(decoded).all():
                        raise ValueError(f"Invalid decoded future video: {decoded.shape}")
                    rgb = ((decoded.clamp(-1, 1) + 1) * 127.5).round().byte()
                    metrics.update(
                        physical_action_metrics(
                            relative.numpy(), physical["relative_actions"].numpy(), mask.numpy(), raw.layout
                        )
                    )
                    metrics.update(image_metrics(rgb, target_video, item["pixel_mask"], item["camera_boxes"]))
                    if losses is not None:
                        # This diagnostic compares generation with the loss forward;
                        # it is defined only for windows included in both budgets.
                        metrics["current_latent_max_abs"] = float(
                            (latent[:, :, 0].float().cpu() - cpu_tree(losses["x0"][-1][:, :, 0]).float()).abs().max()
                        )
                    entry = raw.entries[physical["dataset"]]
                    fps = entry["fps"] / entry["rate"] / raw.video_stride
                    record["sampling"] = dict(
                        solver="UniPC",
                        steps=gen_num_steps,
                        guidance=gen_guidance,
                        shift=gen_shift,
                        seed=gen_seed + sample_id,
                    )
                    if sample_id < visual_count:
                        record["media"] = save_sample_visuals(
                            output,
                            sample_id,
                            prediction=relative.numpy(),
                            target=physical["relative_actions"].numpy(),
                            mask=mask.numpy(),
                            layout=raw.layout,
                            rgb=rgb,
                            target_video=target_video,
                            goal=sample["video"][0],
                            caption=item["caption"],
                            fps=fps,
                        )
                    rows.append(record)
                    artifact = dict(
                        relative=relative,
                        normalized=normalized,
                        absolute=absolute,
                        mask=mask,
                        target_relative=physical["relative_actions"],
                        target_absolute=physical["absolute_actions"],
                        anchor_state=physical["anchor_state"],
                        record=record,
                    )
                    if record_artifacts or sample_id < visual_count:
                        artifact.update(
                            latent=cpu_tree(latent),
                            decoded=decoded,
                            rgb=rgb,
                            target_video=target_video,
                            pixel_mask=item["pixel_mask"],
                            camera_boxes=item["camera_boxes"],
                        )
                    if record_artifacts:
                        artifact["denoise_calls"] = denoise_calls
                    torch.save(artifact, output / f"sample_{sample_id:03d}_generation.pt")
                    print(json.dumps(record), flush=True)
            finally:
                # Explicitly release bounded evaluation prefetch, even after an exception.
                if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                    iterator._shutdown_workers()
    write_json(output / f"metrics_rank_{rank:02d}.json", rows)
    gathered = [None] * world
    dist.all_gather_object(gathered, rows)
    all_rows = sorted(sum(gathered, []), key=lambda row: row["sample_id"])
    summary = aggregate_metrics(all_rows)
    if rank == 0:
        write_json(output / "metrics.json", summary)
        write_gallery(output, all_rows, iteration, weights)
    dist.barrier()
    return summary
