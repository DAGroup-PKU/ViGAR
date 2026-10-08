"""Online current-observation preprocessing and native ViGAR bundle inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from recipes.ViGAR.data.action_layout import resolve_action_layout
from recipes.ViGAR.data.dataset import (
    ACTION_LAYOUT,
    ROBOT_DOMAINS,
    ContractNormalizer,
    LeRobotPolicySFTDataset,
    validate_mask,
    validate_values,
)
from recipes.ViGAR.data.images import (
    compose_cameras,
    compose_goal_image,
    validate_goal_image_composition,
)
from recipes.ViGAR.data.relative_action import (
    check_anchor_head_quaternion,
    propagate_relative_action_validity,
    propagate_state_arm_eef_validity,
    to_global_action,
    transform_state_arm_eef_coordinate,
    zero_chassis_state,
)
from recipes.ViGAR.data.resolution import assign_bucket, validate_buckets
from recipes.ViGAR.trainer.checkpoint_bundle import read_bundle

from ..common.geometry import ACTION_MASK, ROBOT, STATE_MASK
from .checkpoint import load_serving_weights
from .sampling import resolve_sampling


VEOMNI_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_KEYS = (
    "norm_type",
    "img_size",
    "img_size_buckets",
    "resolution",
    "enable_cameras",
    "goal_image_composition",
    "supervise_head_eef",
    "supervise_arm_head_torso",
    "state_arm_eef_coordinate",
    "include_robot_type_text_context",
    "action_layout",
)
CONTRACT_DEFAULTS = dict(
    norm_type="meanstd",
    img_size=None,
    img_size_buckets=[],
    resolution="384x320",
    enable_cameras=["head", "left", "right"],
    goal_image_composition="multi_view",
    supervise_head_eef=False,
    supervise_arm_head_torso=True,
    state_arm_eef_coordinate="head_camera",
    include_robot_type_text_context=False,
    action_layout=ACTION_LAYOUT,
)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else VEOMNI_ROOT / path


def reshard_network(network):
    from torch.distributed._composable.fsdp import FSDPModule

    for module in network.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def load_settings(checkpoint, recipe=None):
    checkpoint = Path(checkpoint).resolve()
    bundle = read_bundle(checkpoint)
    recorded = checkpoint / "recipe_config.json"
    saved = json.loads(recorded.read_text()) if recorded.exists() else None
    supplied = yaml.safe_load(Path(recipe).read_text()) if recipe else None
    for value in (saved, supplied):
        if value is not None:
            value["data"] = {**CONTRACT_DEFAULTS, **value["data"]}
    if saved is None and supplied is None:
        raise ValueError("An initial bundle has no training data settings; supply --recipe")
    settings = supplied or saved
    if saved and supplied:
        for key in CONTRACT_KEYS:
            if saved["data"].get(key) != supplied["data"].get(key):
                raise ValueError(f"Recipe disagrees with trained checkpoint data.{key}")
    data = settings["data"]
    data["_image_preprocessing_version"] = saved.get("image_preprocessing_version", 1) if saved else 2
    if data["_image_preprocessing_version"] not in (1, 2):
        raise ValueError("Unsupported checkpoint image preprocessing version")
    validate_buckets(data["img_size_buckets"], data["img_size"])
    validate_goal_image_composition(data["goal_image_composition"], data["enable_cameras"])
    if not data.get("img_size"):
        raise ValueError("Simulation requires a letterbox-trained checkpoint with data.img_size")
    if set(data.get("enable_cameras", [])) - {"head", "left", "right"}:
        raise ValueError("RoboTwin supplies head/left/right cameras only")
    normalizer = ContractNormalizer(
        resolve_path(data["norm_stat_files"][ROBOT]), data["norm_type"], data["state_arm_eef_coordinate"]
    )
    contract_path = checkpoint / "data_contract.json"
    if contract_path.exists():
        expected = json.loads(contract_path.read_text())["normalizers"][ROBOT]
        if expected["sha256"] != normalizer.sha256 or expected["norm_type"] != normalizer.norm_type:
            raise ValueError("Simulation statistics differ from this checkpoint's training contract")
    return settings, normalizer, bundle


class ObservationProcessor:
    def __init__(self, data, normalizer, tokenizer_config=None):
        self.data, self.normalizer = data, normalizer
        self.goal_image_composition = validate_goal_image_composition(
            data.get("goal_image_composition", "multi_view"), data["enable_cameras"]
        )
        self.layout = resolve_action_layout(data.get("action_layout", ACTION_LAYOUT))
        self.buckets = validate_buckets(data.get("img_size_buckets"), data["img_size"])
        if self.layout != resolve_action_layout(ACTION_LAYOUT):
            raise ValueError("Simulation requires the canonical 49-D layout")
        self.raw_config = SimpleNamespace(
            img_size=data["img_size"], resolution=data.get("resolution", "384x320"), normalizers={ROBOT: normalizer}
        )
        self.sft = LeRobotPolicySFTDataset(self.raw_config, tokenizer_config=tokenizer_config)

    def raw_sample(self, request):
        resolve_sampling(request.get("sampling"))
        if request.get("robot_type") != ROBOT or not isinstance(request.get("instruction"), str):
            raise ValueError("Expected RoboTwin robot_type and instruction string")
        state = torch.as_tensor(request["state"], dtype=torch.float32).clone()
        sm = validate_mask(request["state_valid_mask"], context="simulator state mask").clone()
        am = validate_mask(request["action_valid_mask"], context="simulator action mask").clone()
        if state.shape != (49,) or sm.shape != (49,) or am.shape != (49,):
            raise ValueError("Expected 49-D simulator state/masks")
        if not torch.equal(sm, torch.from_numpy(STATE_MASK)) or not torch.equal(am, torch.from_numpy(ACTION_MASK)):
            raise ValueError("Unexpected aloha-agilex availability masks")
        validate_values(state, sm, context="simulator state")
        # This policy bridge uses the stored RoboTwin common frame, not the
        # simulator world as an alternate common frame.
        if not torch.allclose(state[42:49], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), atol=1e-6, rtol=0):
            raise ValueError("Canonical RoboTwin head pose must be identity")
        check_anchor_head_quaternion(state, sm, self.layout, context="simulator")
        coordinate = self.data["state_arm_eef_coordinate"]
        model_state = transform_state_arm_eef_coordinate(state, coordinate, self.layout)
        model_sm = propagate_state_arm_eef_validity(sm, coordinate, self.layout)
        relative_mask = propagate_relative_action_validity(am.expand(48, -1).clone(), sm, self.layout)
        if not self.data.get("supervise_head_eef", False):
            model_sm[42:49] = False
            relative_mask[:, 42:49] = False
        if not self.data.get("supervise_arm_head_torso", True):
            model_sm[self.layout.joint] = False
            relative_mask[:, self.layout.joint] = False
        model_state = zero_chassis_state(model_state, self.layout).masked_fill(~model_sm, 0)
        normalized = self.normalizer.normalize(model_state, "observation.state", model_sm)
        img_size = self.data["img_size"]
        bucket_id = 0
        if self.buckets:
            # RoboTwin's canonical camera is head. An explicit source_hw can
            # carry canonical metadata when a request uses other native sizes.
            source_hw = request.get("source_hw")
            if source_hw is None:
                head = np.asarray(request["images"].get("head"))
                if head.ndim != 3 or head.shape[-1] != 3:
                    raise ValueError("Bucketed serving requires head RGB or canonical source_hw")
                source_hw = head.shape[:2]
            bucket_id, img_size = assign_bucket(source_hw, self.buckets)
        canvases = []
        for field in ("images", "goal_images"):
            views = {}
            for camera in self.data["enable_cameras"]:
                value = np.asarray(request[field].get(camera))
                if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8 or min(value.shape[:2]) < 1:
                    if camera in ("left", "right"):
                        continue
                    raise ValueError("Expected decoded uint8 RGB images")
                views[camera] = torch.from_numpy(value.copy()).permute(2, 0, 1)[None]
            if field == "images":
                canvas, _, _ = compose_cameras(views, img_size, self.data["enable_cameras"])
            else:
                canvas, _, _ = compose_goal_image(
                    views, img_size, self.data["enable_cameras"], self.goal_image_composition
                )
            canvases.append(canvas.permute(1, 0, 2, 3).contiguous())
        if "goal_canvas_sha256" in request:
            import hashlib

            rgb = canvases[1][:, 0].permute(1, 2, 0).numpy()
            if hashlib.sha256(rgb.tobytes()).hexdigest() != request["goal_canvas_sha256"]:
                raise ValueError("Composed goal pixels differ from the saved generated goal")
        video = torch.zeros(3, 13, *canvases[0].shape[-2:], dtype=torch.uint8)
        video[:, :1] = canvases[0]
        caption = request["instruction"]
        if self.data.get("include_robot_type_text_context", False):
            caption = f"<robot_type>{ROBOT}</robot_type> {caption}"
        sample = dict(
            action=torch.cat([normalized[None], torch.zeros(48, 49)]),
            action_valid_mask=torch.cat([model_sm[None], relative_mask]),
            video=video,
            goal_frame=canvases[1],
            ai_caption=caption,
            mode="policy",
            viewpoint="concat_view",
            fps=torch.tensor(7.5),
            conditioning_fps=torch.tensor(7.5),
            action_fps=torch.tensor(30.0),
            domain_id=torch.tensor(ROBOT_DOMAINS[ROBOT]),
            **({"target_hw": torch.tensor(img_size), "image_bucket_id": bucket_id} if self.buckets else {}),
        )
        return sample, state, relative_mask

    def prepare(self, request):
        from cosmos_framework.data.vfm.action.action_processing import ActionProcessingRecord

        sample, state, mask = self.raw_sample(request)
        prepared = self.sft.prepare_sample(sample, ROBOT)
        # Keep model outputs normalized, then invert with the physical anchor.
        prepared["action_processing_record"] = ActionProcessingRecord(49, None)
        return prepared, state, mask


class ViGARSimulationPolicy:
    def __init__(self, checkpoint, recipe=None, *, weights="ema", output=None, decode_video=False):
        import torch.distributed as dist
        from cosmos_framework.utils import misc
        from cosmos_framework.utils.vfm.parallelism import ParallelDims

        from veomni.models.loader import get_model_class, get_model_config

        self.settings, normalizer, self.bundle = load_settings(checkpoint, recipe)
        self.weights, self.output, self.decode_video = weights, Path(output) if output else None, decode_video
        self.request_count = 0
        config = get_model_config(str(checkpoint))
        assets = self.bundle["assets"]
        native = config.runtime_config(
            base_checkpoint=checkpoint,
            vae_path=Path(checkpoint) / assets["vae"],
            tokenizer_path=Path(checkpoint) / assets["tokenizer"],
            shard_degree=dist.get_world_size(),
            checkpoint_dir=checkpoint,
        )
        if weights == "ema" and not native.model.config.ema.enabled:
            raise ValueError("EMA serving requires an EMA-enabled checkpoint")
        # Serving uses one immutable weight tree. Load the selected checkpoint
        # tensors directly into net, avoiding regular + FP32 EMA allocation.
        native.model.config.ema.enabled = False
        fast = os.environ.get("VIGAR_FAST_INFERENCE") == "1"
        if fast:
            # Opt-in speedup: static block compile with CUDA graphs over the language model and
            # fused projections after loading. BF16 rounding differs from eager serving.
            os.environ.setdefault("VIGAR_LANGUAGE_COMPILE_GRANULARITY", "block")
            os.environ.setdefault("VIGAR_CUDA_GRAPH_PAD_ALIGNMENT", "16")
            compile_config = native.model.config.compile
            compile_config.enabled, compile_config.use_cuda_graphs = True, True
            compile_config.compile_dynamic, compile_config.compiled_region = False, "language"
            import torch._dynamo

            # Static shapes recompile once per packed sample length (one per instruction length).
            torch._dynamo.config.recompile_limit = 128
            torch._dynamo.config.accumulated_recompile_limit = 8 * torch._dynamo.config.recompile_limit
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Compiled graphs cannot enter the NATTEN deterministic-mode context, so fast serving leaves it off.
        torch.use_deterministic_algorithms(not fast, warn_only=True)
        misc.set_random_seed(seed=42, by_rank=True)
        self.model = get_model_class(config)(config, native_config=native, defer_network_init=True)
        parallel = ParallelDims(
            enable_inference_mode=False, world_size=dist.get_world_size(), dp_shard=dist.get_world_size(), cfgp=1, cp=1
        )
        parallel.build_meshes(device_type="cuda")
        self.model.initialize_network(parallel)
        self.core = self.model.core
        self.core.on_train_start(torch.preserve_format)
        loading = load_serving_weights(self.core, checkpoint, weights)
        fused = None
        if fast:
            from cosmos_framework.utils.inference_fusions import fuse_attention_qkv, fuse_dense_swiglu

            fused = dict(gate_up=fuse_dense_swiglu(self.core), qkv=fuse_attention_qkv(self.core))
            print(f"Policy fast inference: fused {fused}", flush=True)
        torch.cuda.empty_cache()
        self.core.eval()
        self.processor = ObservationProcessor(
            self.settings["data"], normalizer, native.model.config.vlm_config.tokenizer
        )
        if self.output and dist.get_rank() == 0:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / "policy_contract.json").write_text(
                json.dumps(
                    dict(
                        checkpoint=str(Path(checkpoint).resolve()),
                        weights=weights,
                        serving_weights="selected once at startup",
                        weight_loading=loading,
                        fast_inference=fused,
                        data=self.settings["data"],
                        norm_sha256=normalizer.sha256,
                        horizon=48,
                        observation_history=1,
                        goal="same-seed expert terminal RGB",
                        simulator_frame="SAPIEN world / native tool / wxyz",
                        model_frame="head camera NWU / Astribot S1 tool / xyzw",
                    ),
                    indent=2,
                )
                + "\n"
            )

    @torch.no_grad()
    def infer(self, request):
        import torch.distributed as dist
        from cosmos_framework.model.vfm.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig
        from cosmos_framework.utils import misc

        from recipes.ViGAR.data.data_collator import collate_samples
        from recipes.ViGAR.trainer.evaluator import seed_all

        sample, state, mask = self.processor.prepare(request)
        train = self.settings["train"]
        sampling = resolve_sampling(request.get("sampling"), train)
        seed = int(request.get("generation_seed", train.get("gen_seed", 9000)))
        seed_all(seed)
        sampler = UniPCSampler(cfg=UniPCSamplerConfig(), tensor_kwargs=self.core.tensor_kwargs)
        try:
            generated = self.core.generate_samples_from_batch(
                misc.to(collate_samples([sample]), device="cuda"),
                sampler=sampler,
                seed=[seed],
                **sampling,
            )
        finally:
            # Match native ema_scope's FSDP cleanup without restoring weights.
            reshard_network(self.core.net)
        normalized = generated["action"][0][1:].float().cpu()
        if normalized.shape != (48, 49) or not torch.isfinite(normalized).all():
            raise ValueError("Invalid generated 48x49 chunk")
        relative = self.processor.normalizer.denormalize_action(normalized).masked_fill(~mask, 0)
        absolute = to_global_action(state, relative, self.processor.layout).masked_fill(~mask, 0)
        artifact = dict(
            normalized=normalized.numpy(),
            relative=relative.numpy(),
            absolute=absolute.numpy(),
            valid=mask.numpy(),
            anchor_state=state.numpy(),
        )
        if dist.get_rank() == 0 and self.output:
            directory = self.output / f"request_{self.request_count:06d}"
            directory.mkdir()
            np.savez_compressed(directory / "actions.npz", **artifact)
            (directory / "request.json").write_text(
                json.dumps(
                    dict(
                        instruction=request["instruction"],
                        seed=seed,
                        sampling=sampling,
                        artifact_context=request.get("artifact_context"),
                    )
                )
                + "\n"
            )
            if self.decode_video:
                import imageio.v2 as imageio

                latent = generated["vision"][0]
                decoded = self.core.decode(latent[None] if latent.ndim == 4 else latent).float().cpu()
                if decoded.ndim == 5:
                    decoded = decoded[0]
                rgb = ((decoded.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).numpy()
                imageio.mimwrite(directory / "predicted_future.mp4", rgb, fps=7.5, macro_block_size=1)
        self.request_count += 1
        return artifact
