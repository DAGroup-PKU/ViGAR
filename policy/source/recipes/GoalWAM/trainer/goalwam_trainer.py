"""VeOmni lifecycle with the native GoalWAM RF update and checkpoint semantics."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from cosmos_framework.utils import misc
from cosmos_framework.utils.callback import CallBackGroup
from cosmos_framework.utils.vfm.parallelism import ParallelDims
from omegaconf import OmegaConf

from veomni.models.loader import get_model_class, get_model_config
from veomni.trainer.base import BaseTrainer
from veomni.trainer.callbacks import TrainerState
from veomni.trainer.callbacks.trace_callback import WandbTraceCallback

from ..data.data_collator import collate_samples
from ..data.data_loader import StatefulWindowLoader
from ..data.dataset import LeRobotPolicyDataset, LeRobotPolicySFTDataset
from .callbacks import UpdateAudit
from .checkpoint_bundle import copy_assets, finish_bundle, read_bundle, save_portable_config
from .evaluator import (
    Trace,
    cpu_tree,
    evaluate,
    json_default,
    load_base,
    load_checkpoint,
    parameter_manifest,
    rng_state,
    save_checkpoint,
    seed_all,
    selected_optimizer_state,
    selected_parameters,
    wandb_evaluation_tables,
    write_json,
)


class GoalWAMTrainer(BaseTrainer):
    def __init__(self, args):
        if args.train.exp_name:
            checkpoint = args.train.checkpoint
            if os.path.basename(os.path.normpath(checkpoint.output_dir)) != args.train.exp_name:
                checkpoint.output_dir = os.path.join(checkpoint.output_dir, args.train.exp_name)
            if args.train.wandb.name is None:
                args.train.wandb.name = args.train.exp_name
        super().__init__(args)

    def _setup(self):
        super()._setup()
        self.output = Path(self.args.train.checkpoint.output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        # VeOmni's generic setup disables BF16 reduced-precision GEMM reduction.
        # Restore the backend contract measured in the original native environment.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        misc.set_random_seed(seed=self.args.train.seed, by_rank=True)

    def _build_model(self):
        args = self.args
        self.model_config = get_model_config(args.model.config_path)
        self.native_config = self.model_config.runtime_config(
            base_checkpoint=args.model.base_checkpoint,
            vae_path=args.model.vae_path,
            tokenizer_path=args.model.tokenizer_path,
            shard_degree=args.train.world_size,
            checkpoint_dir=args.model.config_path if self.model_config.bundle_assets else None,
        )
        self.native_config.trainer.grad_accum_iter = args.train.gradient_accumulation_steps
        self.native_config.trainer.max_iter = args.train.max_steps
        if args.data.include_tail_windows:
            tokenizer = self.native_config.model.config.tokenizer
            tokenizer.encode_exact_durations = sorted(set(tokenizer.encode_exact_durations or []) | {1, 5, 9, 13})
        opt = args.train.optimizer
        self.native_config.optimizer.lr = opt.lr
        self.native_config.optimizer.betas = opt.betas
        self.native_config.optimizer.eps = opt.eps
        self.native_config.optimizer.weight_decay = opt.weight_decay
        for key in self.native_config.optimizer.lr_multipliers:
            if key != "goal_vision_embed":
                self.native_config.optimizer.lr_multipliers[key] = opt.action_lr_multiplier
        if opt.goal_lr_multiplier is not None:
            self.native_config.optimizer.lr_multipliers["goal_vision_embed"] = opt.goal_lr_multiplier
        if opt.disable_weight_decay_for_1d_params is not None:
            self.native_config.optimizer.disable_weight_decay_for_1d_params = opt.disable_weight_decay_for_1d_params
        for key, value in dict(
            cycle_lengths=opt.cycle_length,
            warm_up_steps=opt.warmup_steps,
            f_start=opt.f_start,
            f_max=opt.f_max,
            f_min=opt.f_min,
        ).items():
            self.native_config.scheduler[key] = [value]
        self.native_config.trainer.callbacks.grad_clip.clip_norm = opt.max_grad_norm
        self.native_config.trainer.callbacks.goal_conditioning_health.grad_accum_steps = (
            args.train.gradient_accumulation_steps
        )
        # VeOmni owns the W&B run; keep the native framework from creating a
        # second run for the same process.
        self.native_config.job.wandb_mode = "disabled"
        self.model = get_model_class(self.model_config)(
            self.model_config,
            native_config=self.native_config,
            defer_network_init=True,
        )
        self.core = self.model.core
        # The VeOmni logger below owns step numbering and metric names.
        self.core.log_enc_time_every_n = 0

    def _freeze_model_module(self):
        # Native optimizer construction selects and audits the trainable allowlist,
        # after the network exists at _build_parallelized_model.
        pass

    def _build_model_assets(self):
        self.tokenizer = self.core.vlm_tokenizer
        self.processor = self.core.vlm_processor
        self.model_assets = [self.model_config, self.tokenizer]

    def _build_data_transform(self):
        self.data_transform = LeRobotPolicySFTDataset

    def _dataset(self, path, split, names=None, *, training=False):
        raw = LeRobotPolicyDataset(
            path,
            self.args.data.norm_stat_files,
            split=split,
            training=training,
            include_tail_windows=training and self.args.data.include_tail_windows,
            goal_sampling=self.args.data.goal_sampling,
            goal_image_composition=self.args.data.goal_image_composition,
            seed=self.args.train.seed,
            resolution=self.args.data.resolution,
            norm_type=self.args.data.norm_type,
            dataset_names=names,
            assume_native_astribot_basis=self.args.data.assume_native_astribot_basis,
            img_size=self.args.data.img_size,
            img_size_buckets=self.args.data.img_size_buckets,
            enable_cameras=self.args.data.enable_cameras,
            supervise_head_eef=self.args.data.supervise_head_eef,
            supervise_arm_head_torso=self.args.data.supervise_arm_head_torso,
            state_arm_eef_coordinate=self.args.data.state_arm_eef_coordinate,
            action_layout=self.args.data.action_layout,
            include_robot_type_text_context=self.args.data.include_robot_type_text_context,
            text_conditioning=self.args.data.text_conditioning,
            parquet_cache_dir=self.args.data.parquet_cache_dir,
        )
        return LeRobotPolicySFTDataset(
            raw,
            tokenizer_config=self.native_config.model.config.vlm_config.tokenizer,
            cfg_dropout_rate=self.args.data.caption_dropout_rate if training else 0.0,
        )

    def _build_dataset(self):
        self.train_dataset = self._dataset(
            self.args.data.train_path, self.args.data.split, self.args.data.dataset_names, training=True
        )

    def _build_collate_fn(self):
        self.collate_fn = collate_samples

    def _build_dataloader(self):
        args = self.args
        self.train_dataloader = StatefulWindowLoader(
            self.train_dataset,
            rank=args.train.global_rank,
            world_size=args.train.world_size,
            seed=args.train.seed,
            batch_size=args.train.micro_batch_size,
            num_workers=args.data.dataloader.num_workers,
            prefetch_factor=args.data.dataloader.prefetch_factor,
            fixed_sample_count=args.data.fixed_sample_count,
            accumulation_steps=args.train.gradient_accumulation_steps,
        )

    def _build_parallelized_model(self):
        # Keep the native (replicate=1, shard=world) two-dimensional mesh and
        # exact wrap boundaries. VeOmni owns this construction; no generic wrapper follows.
        self.native_parallel_dims = ParallelDims(
            enable_inference_mode=False,
            world_size=dist.get_world_size(),
            dp_shard=dist.get_world_size(),
            cfgp=1,
            cp=1,
        )
        self.native_parallel_dims.build_meshes(device_type="cuda")
        self.model.initialize_network(self.native_parallel_dims)
        self.model.train()
        self.core.on_train_start(torch.preserve_format)

    def _build_optimizer(self):
        self.optimizers, self.lr_schedulers = self.core.init_optimizer_scheduler(
            self.native_config.optimizer,
            self.native_config.scheduler,
        )

    def _build_lr_scheduler(self):
        # The native optimizer/scheduler containers are built together above.
        pass

    def _build_training_context(self):
        self.model_fwd_context = nullcontext
        self.model_bwd_context = nullcontext
        self.grad_scaler = torch.amp.GradScaler("cuda", **self.native_config.trainer.grad_scaler_args)

    def _init_callbacks(self):
        self.state = TrainerState()
        self.callbacks = CallBackGroup(self.native_config, self)
        self.audit = UpdateAudit(record=self.args.train.record_artifacts)
        self.wandb_callback = WandbTraceCallback(self)

    def forward_backward_step(self, batch, iteration):
        self.callbacks.on_before_forward(iteration=iteration)
        result = self.model(batch, iteration=iteration)
        self.callbacks.on_after_forward(iteration=iteration)
        self.callbacks.on_before_backward(self.core, result.loss, iteration=iteration)
        self.grad_scaler.scale(result.loss / self.args.train.gradient_accumulation_steps).backward()
        self.core.on_after_backward()
        self.callbacks.on_after_backward(self.core, iteration=iteration)
        return result.native_outputs, result.loss

    def train_step(self, batch, iteration, micro):
        outputs, loss = self.forward_backward_step(batch, iteration)
        if micro + 1 == self.args.train.gradient_accumulation_steps:
            self.callbacks.on_before_optimizer_step(
                self.core, self.optimizers, self.lr_schedulers, self.grad_scaler, iteration=iteration
            )
            self.grad_scaler.step(self.optimizers)
            self.grad_scaler.update()
            self.lr_schedulers.step()
            self.callbacks.on_before_zero_grad(self.core, self.optimizers, self.lr_schedulers, iteration=iteration)
            self.core.on_before_zero_grad(self.optimizers, self.lr_schedulers, iteration=iteration)
            self.optimizers.zero_grad(set_to_none=True)
        return outputs, loss

    def _save(self, iteration):
        path = self.output / f"checkpoints/iter_{iteration:09d}"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
        dist.barrier()
        bundled = bool(self.args.model.model_path)
        save_checkpoint(
            self.core,
            self.optimizers,
            self.lr_schedulers,
            self.train_dataloader,
            iteration,
            path,
            mark_complete=not bundled,
        )
        if dist.get_rank() == 0:
            # Export the resolved config beside native regular+EMA tensors.
            self.model_config.cosmos = OmegaConf.to_container(self.native_config, resolve=True)
            if bundled:
                copy_assets(
                    path,
                    vae=self.args.model.vae_path,
                    tokenizer=self.args.model.tokenizer_path,
                    backbone_config=self.native_config.model.config.vlm_config.model_instance.config.base_config.json_file,
                )
                save_portable_config(self.model_config, path)
            else:
                self.model_config.save_pretrained(path)
            write_json(path / "data_contract.json", self.train_dataset._dataset.selection_record())
            write_json(path / "recipe_config.json", {**asdict(self.args), "image_preprocessing_version": 2})
            if bundled:
                finish_bundle(
                    path,
                    kind="training",
                    iteration=iteration,
                    world_size=dist.get_world_size(),
                    provenance=self.model_config.provenance,
                )
        dist.barrier()

    def _evaluate(self, iteration):
        args = self.args
        budget = args.evaluation_budget
        eval_weights = list(self.args.train.eval_weights)
        for index, weights in enumerate(eval_weights):
            summary = evaluate(
                self.core,
                self.eval_dataset,
                self.output / "evaluation",
                iteration,
                self.args.data.eval_count,
                weights=weights,
                eval_loss_per_rank=budget.loss_per_rank,
                generation_per_rank=budget.generation_per_rank,
                generation_wandb_max=budget.visual_max,
                num_workers=args.data.eval_num_workers,
                pin_memory=args.data.eval_pin_memory,
                prefetch_factor=args.data.eval_prefetch_factor,
                selection_seed=args.data.eval_seed,
                gen_num_steps=args.train.gen_num_steps,
                gen_guidance=args.train.gen_guidance,
                gen_shift=args.train.gen_shift,
                loss_seed=args.train.eval_loss_seed,
                gen_seed=args.train.gen_seed,
                record_artifacts=args.train.record_artifacts,
            )
            metrics = {
                f"eval/{robot}/{weights}/{key}": value
                for robot, values in summary["by_robot"].items()
                for key, value in values.items()
            }
            wandb_payload = {}
            if dist.get_rank() == 0 and args.train.wandb.enable:
                evaluation_output = self.output / "evaluation" / f"iter_{iteration:09d}" / weights
                wandb_payload = wandb_evaluation_tables(
                    evaluation_output,
                    summary["samples"],
                    iteration,
                    weights,
                    budget.visual_max,
                )
            self._log_metrics(
                iteration,
                metrics,
                wandb_payload=wandb_payload,
                commit=index == len(eval_weights) - 1,
            )
        self._last_eval_iteration = iteration

    def _log_metrics(self, iteration, metrics, *, wandb_payload=None, commit=True):
        if dist.get_rank() == 0:
            row = dict(step=iteration, **metrics)
            with (self.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
            if self.args.train.wandb.enable:
                import wandb

                wandb.log(dict(metrics) | dict(wandb_payload or {}), step=iteration, commit=commit)

    def _initialize_runtime(self):
        args = self.args
        rank = dist.get_rank()
        if args.model.native_checkpoint:
            initial_bundle = bool(args.model.model_path) and read_bundle(args.model.model_path)["kind"] == "initial"
            # Standalone evaluation must not restore a training optimizer or
            # sampler (their grouping, dropout or rank topology may differ).
            resume = args.model.resume_training and not initial_bundle and not args.train.evaluation_only
            iteration = load_checkpoint(
                self.core,
                self.optimizers,
                self.lr_schedulers,
                self.train_dataloader,
                args.model.native_checkpoint,
                resume=resume,
            )
            if not resume:
                seed_all(42000 + rank)
            if args.train.evaluation_only:
                marker = Path(args.model.native_checkpoint) / "complete.json"
                iteration = int(json.loads(marker.read_text())["iteration"])
        else:
            load_base(self.core, args.model.base_checkpoint, self.output)
            seed_all(42000 + rank)
            iteration = 0
            self._save(0)
        self.state.global_step = iteration
        write_json(self.output / f"parameters_initial_rank_{rank:02d}.json", parameter_manifest(self.core))
        write_json(
            self.output / f"backend_rank_{rank:02d}.json",
            dict(
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                matmul_allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                matmul_allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
            ),
        )
        if rank == 0:
            selection = self.train_dataset._dataset.selection_record()
            selection["fixed_indices"] = self.train_dataloader.sampler.indices
            write_json(self.output / "training_selection.json", selection)
            OmegaConf.save(self.native_config, self.output / "native_config.yaml")
            write_json(self.output / "recipe_config.json", {**asdict(args), "image_preprocessing_version": 2})
        self.eval_dataset = None
        if args.evaluation_budget.windows_per_rank:
            self.eval_dataset = self._dataset(args.data.eval_path, args.data.eval_split, args.data.eval_dataset_names)
            if rank == 0:
                write_json(self.output / "evaluation_selection.json", self.eval_dataset._dataset.selection_record())
        return iteration

    def train(self):
        args, rank = self.args, dist.get_rank()
        iteration = self._initialize_runtime()
        if args.train.initialize_only:
            if args.model.native_checkpoint:
                self._save(iteration)
            return
        self.wandb_callback.on_train_begin(self.state)
        if args.evaluation_budget.windows_per_rank and (args.train.eval_initial or args.train.evaluation_only):
            self._evaluate(iteration)
        if args.train.evaluation_only:
            self.wandb_callback.on_train_end(self.state)
            return
        self.callbacks.on_train_start(self.core, iteration=iteration)
        self.callbacks._callbacks.insert(0, self.audit)
        trace = Trace(self.core) if args.train.record_artifacts else None
        iterator = iter(self.train_dataloader)
        self.optimizers.zero_grad(set_to_none=True)
        detailed_log_context = (
            (self.output / f"losses_rank_{rank:02d}.jsonl").open("a")
            if args.train.record_artifacts
            else nullcontext(None)
        )
        with detailed_log_context as detailed_log:
            while iteration < args.train.max_steps:
                step_started = time.perf_counter()
                data_wait_time = 0.0
                step_metrics = torch.zeros(3, device="cuda", dtype=torch.float64)
                for micro in range(args.train.gradient_accumulation_steps):
                    data_started = time.perf_counter()
                    batch = next(iterator)
                    data_wait_time += time.perf_counter() - data_started
                    if trace:
                        started = time.monotonic()
                        sample_ids = {
                            key: cpu_tree(batch[key])
                            for key in (
                                "dataset_name",
                                "episode_index",
                                "window_start_frame",
                                "goal_frame_index",
                                "goal_requested_source",
                                "goal_source",
                                "goal_delay_seconds",
                                "goal_segment_index",
                                "goal_fallback_reason",
                            )
                            if key in batch
                        }
                        before_rng = rng_state()
                        fixture = cpu_tree(batch)
                    lr_used = [
                        [float(group["lr"]) for group in opt.param_groups] for opt in self.optimizers.optimizers
                    ]
                    self.audit.gradients, self.audit.before, self.audit.clip_norms = {}, {}, {}
                    batch = misc.to(batch, device="cuda")
                    self.callbacks.on_training_step_start(self.core, batch, iteration=iteration)
                    outputs, loss = self.train_step(batch, iteration, micro)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite GoalWAM training loss")
                    step_metrics += (
                        torch.stack(
                            [
                                loss.detach().double(),
                                outputs["flow_matching_loss_action"].detach().double(),
                                outputs["flow_matching_loss_vision"].detach().double(),
                            ]
                        )
                        / args.train.gradient_accumulation_steps
                    )
                    if trace:
                        row = dict(
                            iteration=iteration,
                            micro=micro,
                            rank=rank,
                            loss=float(loss),
                            action_loss=float(outputs["flow_matching_loss_action"]),
                            video_loss=float(outputs["flow_matching_loss_vision"]),
                            lr_used=lr_used,
                            gradient_norm=(
                                float(self.audit.clip_norms["global_norm"]) if self.audit.clip_norms else None
                            ),
                            lr=[
                                [float(group["lr"]) for group in opt.param_groups]
                                for opt in self.optimizers.optimizers
                            ],
                            ema_beta=self.core.ema_beta(iteration),
                            sample_ids=sample_ids,
                            seconds=time.monotonic() - started,
                        )
                        torch.save(
                            dict(
                                inputs=fixture,
                                rng=before_rng,
                                native_trace=trace.data,
                                outputs=cpu_tree(outputs),
                                selected_gradients=self.audit.gradients,
                                clip_norms=self.audit.clip_norms,
                                parameters_before=self.audit.before,
                                parameters_after=selected_parameters(self.core),
                                optimizer_after=selected_optimizer_state(self.core, self.optimizers),
                                sampler=self.train_dataloader.state_dict(),
                            ),
                            self.output / f"step_{iteration:03d}_micro_{micro:02d}_rank_{rank:02d}.pt",
                        )
                        line = json.dumps(row, default=json_default, allow_nan=False)
                        detailed_log.write(line + "\n")
                        detailed_log.flush()
                    self.callbacks.on_training_step_end(self.core, batch, outputs, loss, iteration=iteration)
                iteration += 1
                self.state.global_step = iteration
                dist.all_reduce(step_metrics, op=dist.ReduceOp.SUM)
                step_metrics /= dist.get_world_size()
                torch.cuda.synchronize()
                # Reuse the timing collective; data wait is host time blocked
                # in next(iterator), summed over this rank's microbatches.
                # Maxima can come from different ranks and are not additive.
                step_times = torch.tensor(
                    [time.perf_counter() - step_started, data_wait_time], device="cuda", dtype=torch.float64
                )
                dist.all_reduce(step_times, op=dist.ReduceOp.MAX)
                checkpoint_due = iteration % args.train.checkpoint_every == 0 or iteration == args.train.max_steps
                self._log_metrics(
                    iteration,
                    dict(
                        zip(
                            ("train/loss", "train/action_fm_loss", "train/video_fm_loss"),
                            step_metrics.tolist(),
                            strict=True,
                        )
                    )
                    | {
                        "train/action_fm_weighted": 10 * float(step_metrics[1]),
                        "train/video_fm_weighted": 10 * float(step_metrics[2]),
                        "train/gradient_norm": float(self.audit.clip_norms["global_norm"]),
                        "train/lr": lr_used[0][0],
                        "train/step_time": float(step_times[0]),
                        "train/data_wait_time": float(step_times[1]),
                    },
                    commit=not (checkpoint_due and bool(args.evaluation_budget.windows_per_rank)),
                )
                if checkpoint_due:
                    self._save(iteration)
                    if args.evaluation_budget.windows_per_rank:
                        self._evaluate(iteration)
        if trace:
            trace.close()
        self.audit.close()
        write_json(self.output / f"parameters_final_rank_{rank:02d}.json", parameter_manifest(self.core))
        if args.evaluation_budget.windows_per_rank and getattr(self, "_last_eval_iteration", None) != iteration:
            self._evaluate(iteration)
        if rank == 0:
            modules = {
                name: str(module.__file__)
                for name, module in sorted(sys.modules.items())
                if name.startswith(("cosmos_framework", "recipes.GoalWAM")) and getattr(module, "__file__", None)
            }
            if any("/repos/GoalWAM/" in path for path in modules.values()):
                raise RuntimeError("Migration imported code from the original GoalWAM checkout")
            write_json(self.output / "loaded_runtime_modules.json", modules)
        self.wandb_callback.on_train_end(self.state)
        dist.barrier()
