"""VeOmni model API around the unchanged native Cosmos3-Nano computation."""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_vigar import ViGARConfig


@dataclass
class ViGAROutput(ModelOutput):
    loss: torch.Tensor | None = None
    native_outputs: dict[str, Any] | None = None


class ViGARPolicy(PreTrainedModel):
    config_class = ViGARConfig
    base_model_prefix = "core"
    main_input_name = "batch"

    def __init__(self, config: ViGARConfig, *, native_config=None, defer_network_init=False):
        super().__init__(config)
        from cosmos_framework.model.vfm.omni_mot_model import OmniMoTModel
        from omegaconf import OmegaConf

        self.native_config = native_config or OmegaConf.create(config.cosmos, flags={"allow_objects": True})
        self.core = OmniMoTModel(self.native_config.model.config, defer_network_init=defer_network_init)
        # Native construction owns initialization. HF post_init would reinitialize weights.

    def initialize_network(self, parallel_dims=None):
        self.core.initialize_network(parallel_dims)

    def forward(self, batch, iteration=0):
        outputs, loss = self.core.training_step(batch, iteration)
        return ViGAROutput(loss=loss, native_outputs=outputs)

    def generate_action_and_video(self, batch, **kwargs):
        return self.core.generate_samples_from_batch(batch, **kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise ValueError(
            "ViGAR uses native DCP tensors. Use the recipe's explicit checkpoint import/evaluation entrypoint."
        )
