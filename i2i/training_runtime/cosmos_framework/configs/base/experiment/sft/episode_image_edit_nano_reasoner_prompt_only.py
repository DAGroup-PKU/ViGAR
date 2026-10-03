# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reasoner-cotraining variant where the GENERATOR is conditioned on the prompt-only reasoner K/V.

Extends :mod:`episode_image_edit_nano_reasoner` (subgoal i2i + reasoner CE + LoRA + stop-gradient).
The only difference is ``reasoner_gen_prompt_only=True``: the generator's cross-attention to the
reasoner (und) K/V is restricted to the leading conditioning prefix (current-frame causal VAE
tokens + ``num_prompt_tokens`` episode-prompt tokens), dropping the teacher-forced subtask target
tokens (+ EOS/SOG).

Motivation: in the base reasoner recipe the whole ``[prompt | subtask | EOS | SOG]`` block is one
causal split and the generator attends to all of it, so the generator is teacher-forced on the
ground-truth subtask — which it cannot see at inference (the reasoner must generate it). Here the
generator only ever sees the source+prompt K/V, while the reasoner CE loss still trains on the
subtask in the SAME forward. Because the reasoner is causal and the source+prompt prefix precedes
the subtask, its K/V is independent of the subtask.
"""

import copy

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from cosmos_framework.configs.base.experiment.sft.episode_image_edit_nano_reasoner import (
    episode_image_edit_nano_reasoner,
)

cs = ConfigStore.instance()


episode_image_edit_nano_reasoner_prompt_only = copy.deepcopy(episode_image_edit_nano_reasoner)
episode_image_edit_nano_reasoner_prompt_only.job.name = "episode_image_edit_nano_reasoner_prompt_only"

OmegaConf.set_struct(episode_image_edit_nano_reasoner_prompt_only, False)
_mc = episode_image_edit_nano_reasoner_prompt_only.model.config
# Generator attends to the prompt-prefix und K/V only (no teacher-forced subtask). Reasoner CE
# (predict_text_tokens=True) + stop-gradient + LoRA are inherited unchanged from the base recipe.
_mc.reasoner_gen_prompt_only = True
OmegaConf.set_struct(episode_image_edit_nano_reasoner_prompt_only, True)

cs.store(
    group="experiment",
    package="_global_",
    name="episode_image_edit_nano_reasoner_prompt_only",
    node=episode_image_edit_nano_reasoner_prompt_only,
)
