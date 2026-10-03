"""Serializable native configuration and the fixed 49D data contract."""

import copy

from transformers import PretrainedConfig


class GoalWAMConfig(PretrainedConfig):
    model_type = "goalwam"

    def __init__(
        self,
        cosmos=None,
        action_horizon=48,
        action_dim=49,
        padded_action_dim=64,
        video_stride=4,
        observation_history=1,
        provenance=None,
        bundle_assets=None,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["GoalWAMPolicy"])
        super().__init__(**kwargs)
        if (action_horizon, action_dim, padded_action_dim, video_stride, observation_history) != (48, 49, 64, 4, 1):
            raise ValueError(
                "GoalWAM requires the frozen 48-step / 49-D / 64-padded / stride-4 / current-only contract"
            )
        self.cosmos = copy.deepcopy(cosmos or {})
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.padded_action_dim = padded_action_dim
        self.video_stride = video_stride
        self.observation_history = observation_history
        self.provenance = copy.deepcopy(provenance or {})
        self.bundle_assets = copy.deepcopy(bundle_assets or {})

    def runtime_config(self, *, base_checkpoint, vae_path, tokenizer_path, shard_degree, checkpoint_dir=None):
        from pathlib import Path

        import cosmos_framework
        from omegaconf import OmegaConf

        backbone_config = (
            Path(cosmos_framework.__file__).parent / "model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
        )
        if self.bundle_assets:
            if checkpoint_dir is None:
                raise ValueError("Bundled GoalWAM configuration requires its checkpoint directory")
            root = Path(checkpoint_dir).resolve()
            resolved = {}
            for name, relative in self.bundle_assets.items():
                path = (root / relative).resolve()
                if Path(relative).is_absolute() or not path.is_relative_to(root) or not path.exists():
                    raise ValueError(f"Invalid bundled asset: {relative}")
                resolved[name] = path
            vae_path, tokenizer_path, backbone_config = (
                resolved["vae"],
                resolved["tokenizer"],
                resolved["backbone_config"],
            )
        native = copy.deepcopy(self.cosmos)
        model = native["model"]["config"]
        model["parallelism"]["data_parallel_shard_degree"] = shard_degree
        model["tokenizer"]["vae_path"] = str(Path(vae_path).resolve())
        model["vlm_config"]["tokenizer"]["pretrained_model_name"] = str(Path(tokenizer_path).resolve())
        model["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"] = str(backbone_config)
        native["checkpoint"]["load_path"] = str(Path(base_checkpoint).resolve())
        # Preserve callback order from the executed source runner, independent of JSON key order.
        callbacks = native["trainer"]["callbacks"]
        native["trainer"]["callbacks"] = {
            key: callbacks[key] for key in ("grad_clip", "low_precision", "goal_conditioning_health")
        }
        return OmegaConf.create(native, flags={"allow_objects": True})
