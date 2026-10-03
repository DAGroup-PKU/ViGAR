# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Regression tests for source-frame conditioning of reasoner CE."""

import torch

from cosmos_framework.data.vfm.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.vfm.sequence_packing.types import SequencePlan
from cosmos_framework.model.vfm.mot.attention import _build_gen_prompt_only_kv_meta
from cosmos_framework.model.vfm.utils.data_and_condition import GenerationDataClean


def _pack(*, target_value: float = 1.0):
    source = torch.zeros(1, 4, 1, 4, 4)
    target = torch.full((1, 4, 1, 4, 4), target_value)
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=True,
        x0_tokens_vision=[source, target],
        num_vision_items_per_sample=[2],
    )
    return pack_input_sequence(
        sequence_plans=[SequencePlan(has_text=True, has_vision=True)],
        input_text_indexes=[[10, 11, 12]],  # two prompt tokens, then one supervised target token
        gen_data_clean=data,
        input_timesteps=torch.tensor([0.5]),
        special_tokens={"eos_token_id": 99, "start_of_generation": 100, "end_of_generation": 101},
        latent_patch_size=2,
        position_embedding_type="unified_3d_mrope",
        input_num_prompt_tokens=[2],
        record_gen_visible_prompt=True,
    )


def test_source_vision_and_text_are_one_causal_sequence() -> None:
    packed = _pack()

    # 4 source patches + 3 raw text tokens + EOS + SOG form one causal sequence.
    # The ordinary [source,target] generator vision items remain a separate full sequence.
    assert packed.split_lens == [9, 8]
    assert packed.attn_modes == ["causal", "full"]
    assert packed.reasoner_vision.sequence_indexes.tolist() == [0, 1, 2, 3]
    assert packed.text_indexes.tolist() == [4, 5, 6, 7, 8]
    assert packed.vision.sequence_indexes.tolist() == list(range(9, 17))

    # The first supervised target is predicted at text position 5. Since source positions 0:4
    # precede it in the same causal split, every CE query can attend to the current frame.
    assert packed.ce_loss_indexes.tolist() == [4, 5, 6]
    assert packed.label_ids.tolist() == [-100, 12, 99]
    assert max(packed.reasoner_vision.sequence_indexes.tolist()) < packed.ce_loss_indexes[1].item()


def test_prompt_only_generator_sees_source_prefix_but_not_target_text() -> None:
    packed = _pack()

    # Visible causal prefix = four source patches + two prompt tokens. The supervised target token,
    # EOS, and SOG are excluded from generator cross-attention.
    assert packed.gen_visible_und_lens == [6]
    gather, offsets, max_kv = _build_gen_prompt_only_kv_meta(
        attn_modes=packed.attn_modes,
        split_lens=packed.split_lens,
        sample_lens=packed.sample_lens,
        gen_visible_und_lens=packed.gen_visible_und_lens,
        cp_world_size=1,
        device=torch.device("cpu"),
    )
    assert gather.tolist() == list(range(0, 6)) + list(range(9, 17))
    assert offsets.tolist() == [0, 14]
    assert max_kv == 14


def test_reasoner_prefix_never_uses_generated_target() -> None:
    first = _pack(target_value=1.0)
    second = _pack(target_value=7.0)

    # Changing only the generated target must not alter the reasoner's source prefix.
    assert torch.equal(first.reasoner_vision.tokens[0], second.reasoner_vision.tokens[0])
    assert not torch.equal(first.vision.tokens[1], second.vision.tokens[1])


def test_single_item_generation_sample_has_no_source_prefix() -> None:
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=True,
        x0_tokens_vision=[torch.zeros(1, 4, 1, 4, 4)],
        num_vision_items_per_sample=None,
    )
    packed = pack_input_sequence(
        sequence_plans=[SequencePlan(has_text=True, has_vision=True)],
        input_text_indexes=[[10, 11]],
        gen_data_clean=data,
        input_timesteps=torch.tensor([0.5]),
        special_tokens={"eos_token_id": 99, "start_of_generation": 100, "end_of_generation": 101},
    )

    assert packed.reasoner_vision is None
