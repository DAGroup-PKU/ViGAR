"""Recipe arguments using VeOmni's nested configuration parser."""

import os
from dataclasses import dataclass, field
from typing import Literal

from veomni.arguments import (
    DataArguments,
    ModelArguments,
    OptimizerConfig,
    TrainingArguments,
    VeOmniArguments,
    WandbConfig,
)

from ..data.dataset import ACTION_LAYOUT, manifest_paths
from ..data.goal_sampling import GoalSamplingConfig, resolve_goal_sampling
from ..data.images import CAMERA_KEYS, validate_goal_image_composition
from ..data.resolution import validate_buckets
from ..data.text_conditioning import validate_text_conditioning
from .checkpoint_bundle import local_asset, read_bundle
from .evaluation_budget import resolve_evaluation_budget


@dataclass
class GoalWAMModelArguments(ModelArguments):
    base_checkpoint: str | None = None
    vae_path: str | None = None
    native_checkpoint: str | None = None
    resume_training: bool = True

    def __post_init__(self):
        if self.model_path:
            from pathlib import Path

            root = Path(self.model_path).resolve()
            for value in (self.config_path, self.native_checkpoint):
                if value is not None and Path(value).resolve() != root:
                    raise ValueError(
                        "model_path selects both configuration and weights; conflicting paths are unsupported"
                    )
            bundle = read_bundle(root)
            expected = {
                "vae_path": str(local_asset(root, bundle["assets"]["vae"])),
                "tokenizer_path": str(local_asset(root, bundle["assets"]["tokenizer"])),
                "base_checkpoint": str(root),
            }
            for name, path in expected.items():
                value = getattr(self, name)
                if value is not None and Path(value).resolve() != Path(path):
                    raise ValueError(f"{name} must come from the selected checkpoint bundle")
                setattr(self, name, path)
            self.model_path = self.config_path = self.native_checkpoint = str(root)
        else:
            # Explicit legacy/conversion path. Ordinary recipes use model_path only.
            self.base_checkpoint = self.base_checkpoint or "/path/to/vigar/assets"
            self.vae_path = self.vae_path or "/path/to/vigar/Wan2.2_VAE.pth"
        super().__post_init__()


@dataclass
class GoalWAMDataArguments(DataArguments):
    train_path: str | dict[str, str] = field(metadata={"help": "Source name -> training manifest YAML path."})
    eval_path: str | dict[str, str] | None = field(
        default=None, metadata={"help": "Source name -> evaluation manifest YAML path."}
    )
    text_keys: str = "annotation.task.command.en"
    text_conditioning: Literal["episode", "segment", "episode_segment"] = "episode"
    norm_stat_files: dict[str, str] = field(default_factory=dict)
    parquet_cache_dir: str | None = None
    # Only historical parity profiles opt into episode holdout. Ordinary recipes
    # select their populations entirely through train_path/eval_path.
    split: Literal["train", "eval", "all"] = "all"
    eval_split: Literal["train", "eval", "all"] = "all"
    dataset_names: list[str] | None = None
    eval_dataset_names: list[str] | None = None
    fixed_sample_count: int | None = None
    eval_count: int = 32
    eval_num_workers: int = 0
    eval_pin_memory: bool = False
    eval_prefetch_factor: int = 1
    eval_visual_count: int = 8
    eval_seed: int = 20260824
    norm_type: str = "meanstd"
    # Training only; zero preserves historical migration/compatibility profiles.
    caption_dropout_rate: float = 0.0
    # Training only; evaluation retains the historical full-window population.
    include_tail_windows: bool = False
    goal_sampling: GoalSamplingConfig = field(default_factory=GoalSamplingConfig)
    # Goal only; current/future cameras are selected by enable_cameras.
    goal_image_composition: Literal["multi_view"] = "multi_view"
    img_size: list[int] | None = None
    img_size_buckets: list[list[int]] = field(default_factory=list)
    resolution: str = "384x320"
    enable_cameras: list[str] = field(default_factory=lambda: ["head", "left", "right"])
    supervise_head_eef: bool = False
    supervise_arm_head_torso: bool = True
    state_arm_eef_coordinate: Literal["base", "head_camera"] = "head_camera"
    include_robot_type_text_context: bool = False
    action_layout: dict = field(
        default_factory=lambda: {k: list(v) if isinstance(v, list) else v for k, v in ACTION_LAYOUT.items()}
    )
    assume_native_astribot_basis: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.goal_sampling = resolve_goal_sampling(self.goal_sampling)
        validate_text_conditioning(self.text_conditioning)
        validate_goal_image_composition(self.goal_image_composition, self.enable_cameras)
        self.img_size_buckets = [list(size) for size in validate_buckets(self.img_size_buckets, self.img_size)]
        manifest_paths(self.train_path)
        if self.eval_path is not None:
            manifest_paths(self.eval_path)
        if self.text_keys not in {"annotation.task.command.en", "ai_caption"}:
            raise ValueError("GoalWAM text comes from annotation.task.command.en (internal alias: ai_caption)")
        if not 0.0 <= self.caption_dropout_rate <= 1.0:
            raise ValueError("data.caption_dropout_rate must be in [0, 1]")
        if (
            self.eval_count < 0
            or self.eval_visual_count < 0
            or self.eval_num_workers < 0
            or self.eval_prefetch_factor < 1
        ):
            raise ValueError("Evaluation counts/workers must be nonnegative and prefetch_factor positive")
        if self.img_size is not None and (len(self.img_size) != 2 or any(v < 2 or v % 2 for v in self.img_size)):
            raise ValueError("img_size must contain even positive HEIGHT WIDTH values")
        if (
            not self.enable_cameras
            or set(self.enable_cameras) - CAMERA_KEYS.keys()
            or len(set(self.enable_cameras)) != len(self.enable_cameras)
        ):
            raise ValueError("enable_cameras must select unique torso/head/left/right cameras")


@dataclass
class GoalWAMOptimizerConfig(OptimizerConfig):
    type: Literal["FusedAdam"] = "FusedAdam"
    lr: float = 1e-5
    weight_decay: float = 0.05
    betas: list[float] = field(default_factory=lambda: [0.9, 0.99])
    eps: float = 1e-8
    action_lr_multiplier: float = 5.0
    # None retains the checkpoint's historical optimizer grouping.
    goal_lr_multiplier: float | None = None
    disable_weight_decay_for_1d_params: bool | None = None
    lr_decay_style: str = "native_linear"
    cycle_length: int = 100000
    warmup_steps: int = 0
    f_start: float = 0.0
    f_max: float = 0.4
    f_min: float = 0.0


@dataclass
class GoalWAMTrainingArguments(TrainingArguments):
    optimizer: GoalWAMOptimizerConfig = field(default_factory=GoalWAMOptimizerConfig)
    wandb: WandbConfig = field(default_factory=lambda: WandbConfig(project="goalwam"))
    exp_name: str | None = None
    eval_initial: bool = False
    evaluation_only: bool = False
    initialize_only: bool = False
    record_artifacts: bool = False
    checkpoint_every: int = 20
    eval_weights: list[str] = field(default_factory=lambda: ["regular", "ema"])
    # None retains the legacy data.eval_count / data.eval_visual_count limits.
    eval_loss_per_rank: int | None = None
    generation_per_rank: int | None = None
    generation_wandb_max: int | None = None
    gen_num_steps: int = 5
    gen_guidance: float = 3.0
    gen_shift: float = 5.0
    eval_loss_seed: int = 8000
    gen_seed: int = 9000

    def __post_init__(self):
        if self.exp_name:
            checkpoint = self.checkpoint
            if os.path.basename(os.path.normpath(checkpoint.output_dir)) != self.exp_name:
                checkpoint.output_dir = os.path.join(checkpoint.output_dir, self.exp_name)
            if self.wandb.name is None:
                self.wandb.name = self.exp_name
        super().__post_init__()
        acc = self.accelerator
        if any(size != 1 for size in (acc.tp_size, acc.pp_size, acc.cp_size, acc.ulysses_size, acc.ep_size)):
            raise ValueError("GoalWAM initially supports pure data parallelism only")
        if acc.dp_replicate_size != 1 or acc.dp_shard_size != self.world_size:
            raise ValueError("GoalWAM requires one native FSDP shard group over all ranks")
        if self.dyn_bsz:
            raise ValueError(
                "GoalWAM dyn_bsz is unsupported: native multimodal packing uses fixed samples and sample-mean RF losses; variable token batches require a sampler/reduction adaptation"
            )
        if self.enable_mixed_precision:
            raise ValueError(
                "Leave enable_mixed_precision=false: GoalWAM already uses native BF16 FSDP with FP32 optimizer moments/master weights; VeOmni's generic mixed-precision wrapper is not used"
            )
        if self.enable_compile or self.train_architecture != "full":
            raise ValueError(
                "GoalWAM requires fixed sample batches, native BF16, no compilation, and full training mode"
            )
        if self.max_steps is None or self.max_steps < 0 or self.checkpoint_every < 1:
            raise ValueError("Provide nonnegative train.max_steps and positive checkpoint_every")
        if not self.eval_weights or any(weight not in {"regular", "ema"} for weight in self.eval_weights):
            raise ValueError("eval_weights must select regular and/or ema")
        if self.gen_num_steps < 1 or self.gen_shift <= 0 or self.gen_guidance < 0:
            raise ValueError("Generation steps/shift must be positive and guidance nonnegative")
        opt = self.optimizer
        if opt.goal_lr_multiplier is not None and not 0 < opt.goal_lr_multiplier < float("inf"):
            raise ValueError("optimizer.goal_lr_multiplier must be finite and positive")
        if opt.type != "FusedAdam" or opt.lr_decay_style != "native_linear":
            raise ValueError("GoalWAM preserves native FusedAdam and the native_linear scheduler")
        if len(opt.betas) != 2 or any(not 0 <= beta < 1 for beta in opt.betas):
            raise ValueError("optimizer.betas must contain two values in [0,1)")
        if (
            opt.cycle_length <= opt.warmup_steps
            or opt.warmup_steps < 0
            or opt.lr <= 0
            or opt.eps <= 0
            or opt.action_lr_multiplier <= 0
        ):
            raise ValueError("Invalid native optimizer/scheduler settings")


@dataclass
class VeOmniGoalWAMArguments(VeOmniArguments):
    model: GoalWAMModelArguments = field(default_factory=GoalWAMModelArguments)
    data: GoalWAMDataArguments = field(default_factory=GoalWAMDataArguments)
    train: GoalWAMTrainingArguments = field(default_factory=GoalWAMTrainingArguments)

    @property
    def evaluation_budget(self):
        return resolve_evaluation_budget(
            self.train.world_size,
            count=self.data.eval_count,
            visual_count=self.data.eval_visual_count,
            eval_loss_per_rank=self.train.eval_loss_per_rank,
            generation_per_rank=self.train.generation_per_rank,
            generation_wandb_max=self.train.generation_wandb_max,
        )

    def __post_init__(self):
        super().__post_init__()
        if self.evaluation_budget.windows_per_rank and self.data.eval_path is None:
            raise ValueError("Set data.eval_path explicitly when evaluation is enabled; no training-data fallback")
