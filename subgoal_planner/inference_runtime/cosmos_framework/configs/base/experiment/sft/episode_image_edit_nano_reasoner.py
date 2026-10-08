# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Nano episode image-edit SFT that ALSO trains the reasoner on subtask prompts.

Extends :mod:`episode_image_edit_nano` (subgoal i2i, ``target_mode=segment_final``): in addition to
the generator flow-matching loss, the reasoner (understanding tower) is trained with a next-token
cross-entropy loss to predict the target subtask instruction (``action_steps[].action_text`` from
``meta/annotations.json``) given the current frame + the episode-level task. In the last 15% of a
non-final segment, the target is the next segment rather than the nearly completed current one;
the final segment continues to target its own final frame. Two losses:

  * Reasoner CE — LoRA on the reasoner attention projections (``q/k/v/o_proj``), with the base
    embedding/output matrix frozen. ``predict_text_tokens=True`` makes the network emit ``ce_preds``;
    the dataset
    (``reasoner_subtask_target=True``, set via ``EPISODE_IMAGE_EDIT_REASONER_TARGET=1``) feeds the
    current frame as a causal VAE-token prefix, the episode task as prompt, and the subtask as the
    supervised target (source/prompt positions have no CE labels; prompt labels are masked).
  * Generator flow-matching — unchanged; the generator still attends to the reasoner.

``reasoner_stop_gradient=True`` detaches the reasoner K/V feeding the generator's attention, so the
flow-matching loss never updates the reasoner — the reasoner is trained purely by its CE loss.

Trainable surface (base reasoner weights otherwise frozen):
  * ``lora_`` adapters on the reasoner attention (``lora_target_modules="q_proj,k_proj,v_proj,o_proj"``).
  * Generator pathway ``moe_gen`` / ``vae2llm`` / ``llm2vae`` / ``time_embedder`` (full, as before).
  * The tied ``embed_tokens`` / ``lm_head`` matrix remains frozen; updating that full matrix made
    the reasoner tuning surface hundreds of millions of parameters larger than the LoRA adapters.

The optimizer groups and checkpoint knobs live in
``toml/sft_config/episode_image_edit_nano_reasoner.toml``; the reasoner/LoRA model config fields are
set here because they are recipe-level architecture choices.
"""

import copy

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from cosmos_framework.configs.base.experiment.sft.episode_image_edit_nano import episode_image_edit_nano

cs = ConfigStore.instance()


episode_image_edit_nano_reasoner = copy.deepcopy(episode_image_edit_nano)
episode_image_edit_nano_reasoner.job.name = "episode_image_edit_nano_reasoner"

# NANO_MODEL_CONFIG does not carry the reasoner/LoRA fields, so relax struct to add them; they merge
# into OmniMoTModelConfig (which declares them) at Hydra compose time.
OmegaConf.set_struct(episode_image_edit_nano_reasoner, False)
_mc = episode_image_edit_nano_reasoner.model.config
# Reasoner text supervision + stop-gradient.
_mc.predict_text_tokens = True
_mc.reasoner_ce_weight = 1.0
_mc.reasoner_stop_gradient = True
# Keep CE gradients on reasoner LoRA only; ``vae2llm`` is updated by flow matching through the
# generator optimizer and is merely used as a detached feature projection for reasoner input.
_mc.reasoner_detach_source_vision_projection = True
# LoRA on the reasoner attention; co-train only the generator pathway outside the adapters.
_mc.lora_enabled = True
_mc.lora_rank = 16
_mc.lora_alpha = 32
_mc.lora_target_modules = "q_proj,k_proj,v_proj,o_proj"
_mc.lora_cotrain_keys = "moe_gen,vae2llm,llm2vae,time_embedder"
OmegaConf.set_struct(episode_image_edit_nano_reasoner, True)

cs.store(
    group="experiment",
    package="_global_",
    name="episode_image_edit_nano_reasoner",
    node=episode_image_edit_nano_reasoner,
)
