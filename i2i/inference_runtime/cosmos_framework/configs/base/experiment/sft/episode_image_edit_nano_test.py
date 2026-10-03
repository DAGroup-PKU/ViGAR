# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from omegaconf import OmegaConf

from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.configs.base.experiment.sft.episode_image_edit_nano import episode_image_edit_nano


def test_episode_image_edit_reasoner_invariants() -> None:
    model_config = episode_image_edit_nano.model.config

    assert model_config.reasoner_stop_gradient is True
    assert OmegaConf.select(model_config, "predict_text_tokens", default=False) is False
    assert episode_image_edit_nano.optimizer.keys_to_select == [
        "moe_gen",
        "time_embedder",
        "vae2llm",
        "llm2vae",
    ]


def test_reasoner_is_frozen_by_default() -> None:
    assert OmniMoTModelConfig().reasoner_stop_gradient is True
