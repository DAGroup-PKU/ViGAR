# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch

from cosmos_framework.model.attention import (
    attention,
    merge_attentions,
    multi_dimensional_attention_varlen,
)
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.vfm.utils.memory import KVToStore, MemoryValue


class SplitInfo:
    def __init__(
        self,
        split_lens: list[int],
        attn_modes: list[str],
        sample_lens: list[int],
        actual_len: int,
        is_three_way: bool = False,
        vision_token_shapes: list[tuple[int, int, int]] | None = None,
        action_token_shapes: list[tuple[int, ...]] | None = None,
        num_action_tokens_per_supertoken: int = 0,
        null_action_supertokens: bool = False,
        reasoner_stop_gradient: bool = False,
        gen_kv_gather_index: torch.Tensor | None = None,
        gen_kv_offsets: torch.Tensor | None = None,
        gen_max_kv: int | None = None,
    ):
        """
        Actual len is the actual non-padded length of the packed sequence.
        It's used to trim split_lens, attn_modes and sample_lens, which may
        be padded to max sequence length by upstream packers.
        """
        assert sum(sample_lens) == sum(split_lens), (
            f"Sum of new sample lens {sum(sample_lens)} is not equal to sum of new split lens {sum(split_lens)}"
        )

        max_causal_len = 0
        max_full_len = 0
        for split_len, attn_mode in zip(split_lens, attn_modes):
            if attn_mode == "causal":
                max_causal_len = max(max_causal_len, split_len)
            elif attn_mode == "full":
                max_full_len = max(max_full_len, split_len)

        self.max_causal_len = max_causal_len
        self.max_full_len = max_full_len
        self.max_sample_len = max(sample_lens)

        self.split_lens = split_lens
        self.attn_modes = attn_modes
        self.sample_lens = sample_lens

        self.is_three_way = is_three_way
        self.vision_token_shapes = vision_token_shapes
        self.action_token_shapes = action_token_shapes
        self.num_action_tokens_per_supertoken = num_action_tokens_per_supertoken
        self.null_action_supertokens = null_action_supertokens
        # When True, the generator ("full") queries attend to a DETACHED copy of the reasoner
        # ("causal") K/V, so the flow-matching loss does not backprop into the reasoner tower. The
        # reasoner self-attention keeps grad-carrying K/V (its CE loss still trains it).
        self.reasoner_stop_gradient = reasoner_stop_gradient
        # Optional generator->reasoner cross-attention restriction (reasoner_gen_prompt_only). When
        # set, the generator's "full" queries attend to a GATHERED subset of the joint K/V —
        # per sample, source+prompt prefix und tokens + all gen tokens (the teacher-forced subtask und
        # tokens are dropped). gen_kv_gather_index indexes into get_all_seq(...); gen_kv_offsets are
        # the matching per-sample cu-seqlens; gen_max_kv is max(np_s + G_s). None => unrestricted
        # (legacy: generator attends to the whole sample via sample_offsets).
        self.gen_kv_gather_index = gen_kv_gather_index
        self.gen_kv_offsets = gen_kv_offsets
        self.gen_max_kv = gen_max_kv


AttentionMaskType = SplitInfo


_dotproduct_attention_cache = {}


from cosmos_framework.data.vfm.sequence_packing.natten import (
    generate_natten_metadata,
    generate_temporal_causal_natten_metadata,
)
from cosmos_framework.data.vfm.sequence_packing.runtime import (
    SequencePack,
    from_mode_splits,
    get_all_seq,
    get_causal_seq,
    get_full_only_seq,
    sequence_pack_from_packed_sequence,
)


def _detached_und_kv_pack(pack: SequencePack) -> SequencePack:
    """Shallow-copy a K/V pack with the reasoner (causal/und) portion detached.

    The generator ("full") queries reconstruct their K/V via ``get_all_seq``, which scatters
    ``causal_seq`` (und) and ``full_only_seq`` (gen). Detaching only ``causal_seq`` makes the
    generator's attention read the reasoner K/V as a constant (stop-gradient), while the generator's
    own K/V still carries gradient. The reasoner's self-attention uses the original (non-copied)
    pack, so its CE gradient is untouched. Any cached ``all_seq`` is dropped so ``get_all_seq``
    recomputes from the detached ``causal_seq``.
    """
    detached = dict(pack)
    detached.pop("all_seq", None)
    detached["causal_seq"] = pack["causal_seq"].detach()
    return detached


def two_way_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    reasoner_stop_gradient: bool = False,
    gen_kv_gather_index: torch.Tensor | None = None,
    gen_kv_offsets: torch.Tensor | None = None,
    gen_max_kv: int | None = None,
) -> SequencePack:
    """
    Performs two-way attention with causal and full attention.

    When ``reasoner_stop_gradient`` is True, the generator ("full") queries attend to a detached
    copy of the reasoner ("causal") K/V so the flow-matching loss cannot update the reasoner tower.

    When ``gen_kv_gather_index`` is provided (reasoner_gen_prompt_only), the generator's queries
    attend to a GATHERED subset of the joint K/V — per sample, source+prompt prefix und tokens + all
    gen tokens — instead of the whole sample. This drops the teacher-forced subtask und tokens from
    the generator's view while leaving the reasoner self-attention (``causal_res``) untouched.
    """

    causal_q, causal_q_offsets = get_causal_seq(packed_query_states)
    causal_k, causal_k_offsets = get_causal_seq(packed_key_states)
    causal_v, _ = get_causal_seq(packed_value_states)
    full_q, full_q_offsets = get_full_only_seq(packed_query_states)

    sample_offsets = packed_query_states["sample_offsets"]

    use_dont_care_mask = causal_q_offsets is causal_k_offsets

    # NOTE: cosmos_framework attention is BSHD in, BSHD out
    causal_res = attention(
        causal_q.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_k.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        cumulative_seqlen_Q=causal_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_causal_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
    )  # [1,N_und,heads,head_dim]

    # [1,N_und,heads,head_dim] -> [N_und,heads,head_dim] -> [N_und,heads*head_dim]
    causal_out = causal_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_und,heads*head_dim]

    # Generator queries attend to und+gen K/V. Under stop-gradient, the und portion of that K/V is
    # detached so flow-matching gradient never reaches the reasoner tower.
    key_pack_for_gen = _detached_und_kv_pack(packed_key_states) if reasoner_stop_gradient else packed_key_states
    value_pack_for_gen = _detached_und_kv_pack(packed_value_states) if reasoner_stop_gradient else packed_value_states

    gen_key = get_all_seq(key_pack_for_gen)  # [1?,N_all,heads,head_dim] -> [N_all,...]
    gen_value = get_all_seq(value_pack_for_gen)
    if gen_kv_gather_index is not None:
        # Restrict the generator's key/value set to (prompt und + gen) per sample. The gathered
        # tensors stay contiguous per sample, so flash varlen with gen_kv_offsets is well-formed.
        gen_key = gen_key[gen_kv_gather_index]
        gen_value = gen_value[gen_kv_gather_index]
        gen_kv_cu = gen_kv_offsets
        gen_kv_max = gen_max_kv
    else:
        gen_kv_cu = sample_offsets
        gen_kv_max = packed_query_states["max_sample_len"]

    full_res = attention(
        full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
        gen_key.unsqueeze(0),  # [1,N_kv,heads,head_dim]
        gen_value.unsqueeze(0),  # [1,N_kv,heads,head_dim]
        cumulative_seqlen_Q=full_q_offsets,
        cumulative_seqlen_KV=gen_kv_cu,
        max_seqlen_Q=packed_query_states["max_full_len"],
        max_seqlen_KV=gen_kv_max,
    )  # [1,N_full,heads,head_dim]

    # [1,N_full,heads,head_dim] -> [N_full,heads,head_dim] -> [N_full,heads*head_dim]
    full_out = full_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_full,heads*head_dim]

    out_all = from_mode_splits(causal_out, full_out, packed_query_states)
    return out_all


def three_way_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    natten_metadata: dict | None,
    attention_meta: SplitInfo | None = None,
    reasoner_stop_gradient: bool = False,
) -> SequencePack:
    """
    Performs three-way attention, with understanding and generations attentions fully decomposed,
    and allows sparsity / multi-dimensional masking in the generation tower.

    When attention_meta is provided with null_action_supertokens=True, zeros V for the first
    num_action_tokens_per_supertoken tokens of each sample's GEN sequence (null action
    supertokens for temporal causal training). The metadata encodes is_causal=(True, False):
    causal across T supertokens, full within each supertoken S.

    NOTE: the three-way decomposition is only done so we can handle sparsity in the gen tower,
    but a KEY assumption is that the "full" tokens all correspond to the same modality!
    We should be careful when extending this to beyond t2i and t2v.
    """

    causal_q, causal_q_offsets = get_causal_seq(packed_query_states)
    causal_k, causal_k_offsets = get_causal_seq(packed_key_states)
    causal_v, _ = get_causal_seq(packed_value_states)
    full_q, full_q_offsets = get_full_only_seq(packed_query_states)
    full_k, full_k_offsets = get_full_only_seq(packed_key_states)
    full_v, _ = get_full_only_seq(packed_value_states)

    sample_offsets = packed_query_states["sample_offsets"]

    if attention_meta is not None and attention_meta.null_action_supertokens:
        # Zero V for the first num_action_tokens_per_supertoken tokens of each
        # sample's GEN sequence (null action supertokens at t=0).
        # out_i = Σ_j softmax(QKᵀ/√d)_j · V_j — terms with V_j=0 contribute exactly 0 to the output,
        # regardless of attention weights. Softmax mass is still allocated to these positions (not
        # redistributed), so this differs from hard key masking, but the output contribution is 0.
        full_v = full_v.clone()
        starts = full_q_offsets[:-1].long()  # [B]
        null_positions = (
            starts.unsqueeze(1) + torch.arange(attention_meta.num_action_tokens_per_supertoken, device=starts.device)
        ).reshape(-1)
        full_v[null_positions] = 0

    use_dont_care_mask = causal_q_offsets is causal_k_offsets

    # NOTE: cosmos_framework attention is BSHD in, BSHD out
    causal_res = attention(
        causal_q.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_k.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        cumulative_seqlen_Q=causal_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_causal_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
    )  # [1,N_und,heads,head_dim]
    # [1,N_und,heads,head_dim] -> [N_und,heads,head_dim] -> [N_und,heads*head_dim]
    causal_out = causal_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_und,heads*head_dim]

    # If there's no metadata, it's a dense layer
    if natten_metadata is None:
        full_sa, full_sa_lse = attention(
            full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_k.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_v.unsqueeze(0),  # [1,N_full,heads,head_dim]
            cumulative_seqlen_Q=full_q_offsets,
            cumulative_seqlen_KV=full_k_offsets,
            max_seqlen_Q=packed_query_states["max_full_len"],
            max_seqlen_KV=packed_query_states["max_full_len"],
            return_lse=True,
        )  # full_sa: [1,N_full,heads,head_dim], full_sa_lse: [1,N_full,heads]
    else:
        assert natten_metadata is not None
        full_sa, full_sa_lse = multi_dimensional_attention_varlen(
            full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_k.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_v.unsqueeze(0),  # [1,N_full,heads,head_dim]
            metadata=natten_metadata,
            return_lse=True,
        )  # full_sa: [1,N_full,heads,head_dim], full_sa_lse: [1,N_full,heads]

    # Generator→reasoner cross-attention. Under stop-gradient, detach the reasoner (causal) K/V so
    # flow-matching gradient never reaches the reasoner tower; its self-attention above keeps the
    # grad-carrying causal_k/causal_v.
    ca_k = causal_k.detach() if reasoner_stop_gradient else causal_k
    ca_v = causal_v.detach() if reasoner_stop_gradient else causal_v
    full_ca, full_ca_lse = attention(
        full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
        ca_k.unsqueeze(0),  # [1,N_und,heads,head_dim]
        ca_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        cumulative_seqlen_Q=full_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_full_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        return_lse=True,
    )  # full_ca: [1,N_full,heads,head_dim], full_ca_lse: [1,N_full,heads]

    assert full_sa.shape == full_ca.shape
    full_res, _ = merge_attentions(
        outputs=[full_sa, full_ca], lse_tensors=[full_sa_lse, full_ca_lse], torch_compile=False
    )  # [1,N_full,heads,head_dim]

    # [1,N_full,heads,head_dim] -> [N_full,heads,head_dim] -> [N_full,heads*head_dim]
    full_out = full_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_full,heads*head_dim]

    out_all = from_mode_splits(causal_out, full_out, packed_query_states)
    return out_all


def dispatch_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_mask: SplitInfo,
    natten_metadata: dict | None = None,
    memory_value: MemoryValue | None = None,
) -> tuple[SequencePack, KVToStore | None]:
    assert memory_value is None, "Base dispatch_attention does not handle MemoryValue"
    reasoner_stop_gradient = bool(getattr(attention_mask, "reasoner_stop_gradient", False))
    gen_kv_gather_index = getattr(attention_mask, "gen_kv_gather_index", None)
    gen_kv_offsets = getattr(attention_mask, "gen_kv_offsets", None)
    gen_max_kv = getattr(attention_mask, "gen_max_kv", None)
    if isinstance(attention_mask, SplitInfo) and attention_mask.is_three_way:
        if gen_kv_gather_index is not None:
            raise NotImplementedError(
                "reasoner_gen_prompt_only (generator prompt-only cross-attention) is only implemented "
                "for two_way attention; this recipe uses joint_attn_implementation='two_way'."
            )
        output = three_way_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            natten_metadata=natten_metadata,
            attention_meta=attention_mask,
            reasoner_stop_gradient=reasoner_stop_gradient,
        )
    elif isinstance(attention_mask, SplitInfo):
        output = two_way_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            reasoner_stop_gradient=reasoner_stop_gradient,
            gen_kv_gather_index=gen_kv_gather_index,
            gen_kv_offsets=gen_kv_offsets,
            gen_max_kv=gen_max_kv,
        )
    else:
        raise TypeError(f"Unsupported attention metadata: {type(attention_mask)}")
    return output, None


def _build_gen_prompt_only_kv_meta(
    *,
    attn_modes: list[str],
    split_lens: list[int],
    sample_lens: list[int],
    gen_visible_und_lens: list[int],
    cp_world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the generator's restricted-KV gather index + cu-seqlens (reasoner_gen_prompt_only).

    Per sample the generator attends to (source+prompt-prefix und tokens) + (all gen tokens), dropping the
    teacher-forced subtask und tokens. Returns ``(gather_index, gen_kv_offsets, gen_max_kv)`` where
    ``gather_index`` indexes into ``get_all_seq(...)`` (packed order: each sample is [und..., gen...]).

    Built from python ints (mirrors ``_compute_mode_indices_and_offsets`` in runtime.py) so the
    torch.compile behavior matches the existing packing metadata.
    """
    assert cp_world_size == 1, "reasoner_gen_prompt_only is incompatible with context parallel (cp>1)"
    num_samples = len(sample_lens)
    assert len(gen_visible_und_lens) == num_samples, (
        f"gen_visible_und_lens ({len(gen_visible_und_lens)}) must be 1:1 with samples ({num_samples})"
    )

    # Per-sample und (causal) token count. Splits are packed per sample (text/causal then vision/full)
    # and never cross a sample boundary, so we walk splits and attribute each to its sample.
    und_len_per_sample = [0] * num_samples
    pos = 0
    sample_idx = 0
    sample_end = sample_lens[0] if num_samples > 0 else 0
    for split_len, mode in zip(split_lens, attn_modes):
        while sample_idx < num_samples - 1 and pos >= sample_end:
            sample_idx += 1
            sample_end += sample_lens[sample_idx]
        if mode == "causal":
            und_len_per_sample[sample_idx] += split_len
        pos += split_len

    gather: list[int] = []
    offsets: list[int] = [0]
    running = 0
    gen_max_kv = 0
    sample_start = 0
    for s in range(num_samples):
        u_s = und_len_per_sample[s]
        g_s = sample_lens[s] - u_s
        np_s = gen_visible_und_lens[s]
        assert 0 <= np_s <= u_s, f"sample {s}: prompt len {np_s} out of und range [0,{u_s}]"
        # prompt-prefix und tokens
        gather.extend(range(sample_start, sample_start + np_s))
        # all gen tokens (und block precedes gen block within the sample)
        gather.extend(range(sample_start + u_s, sample_start + u_s + g_s))
        cnt = np_s + g_s
        running += cnt
        offsets.append(running)
        gen_max_kv = max(gen_max_kv, cnt)
        sample_start += sample_lens[s]

    gather_index = torch.tensor(gather, dtype=torch.long, device=device)
    gen_kv_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    return gather_index, gen_kv_offsets, int(gen_max_kv)


def build_packed_sequence(
    joint_attn_implementation: str,
    *,
    packed_sequence: torch.Tensor,
    attn_modes: list[str],
    split_lens: list[int],
    sample_lens: list[int],
    packed_und_token_indexes: torch.LongTensor,
    packed_gen_token_indexes: torch.LongTensor,
    num_heads: int,
    head_dim: int,
    num_layers: int,
    token_shapes: list[tuple[int, int, int]] | None = None,
    natten_parameter_list: list | None = None,
    block_size: int = 128,
    is_image_batch: bool = False,
    cp_world_size: int = 1,
    video_temporal_causal: bool = False,
    skip_natten_metadata: bool = False,
    vision_token_shapes: list[tuple[int, int, int]] | None = None,
    action_token_shapes: list[tuple[int, ...]] | None = None,
    num_action_tokens_per_supertoken: int = 0,
    null_action_supertokens: bool = False,
    pad_for_cuda_graphs: bool = False,
    reasoner_stop_gradient: bool = False,
    gen_visible_und_lens: list[int] | None = None,
) -> tuple[SequencePack, AttentionMaskType, list | None]:
    """
    Build the model input pack and attention meta for joint attention.
    Returns a tuple: (input_pack, attention_meta).
    """
    device = packed_sequence.device
    natten_metadata_list = None
    if joint_attn_implementation == "two_way":
        gen_kv_gather_index = None
        gen_kv_offsets = None
        gen_max_kv = None
        if gen_visible_und_lens is not None:
            gen_kv_gather_index, gen_kv_offsets, gen_max_kv = _build_gen_prompt_only_kv_meta(
                attn_modes=attn_modes,
                split_lens=split_lens,
                sample_lens=sample_lens,
                gen_visible_und_lens=gen_visible_und_lens,
                cp_world_size=cp_world_size,
                device=device,
            )
        attention_meta = SplitInfo(
            split_lens=split_lens,
            attn_modes=attn_modes,
            sample_lens=sample_lens,
            actual_len=int(packed_sequence.shape[0]),
            reasoner_stop_gradient=reasoner_stop_gradient,
            gen_kv_gather_index=gen_kv_gather_index,
            gen_kv_offsets=gen_kv_offsets,
            gen_max_kv=gen_max_kv,
        )
        make_pack = sequence_pack_from_packed_sequence
    elif joint_attn_implementation == "three_way":
        if gen_visible_und_lens is not None:
            raise NotImplementedError(
                "reasoner_gen_prompt_only is only implemented for two_way attention."
            )
        attention_meta = SplitInfo(
            split_lens=split_lens,
            attn_modes=attn_modes,
            sample_lens=sample_lens,
            actual_len=int(packed_sequence.shape[0]),
            is_three_way=True,
            vision_token_shapes=vision_token_shapes,
            action_token_shapes=action_token_shapes,
            num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
            null_action_supertokens=null_action_supertokens,
            reasoner_stop_gradient=reasoner_stop_gradient,
        )
        make_pack = sequence_pack_from_packed_sequence
        # Some memory-driven attention paths implement temporal visibility in
        # their own attention kernels; skip NATTEN metadata for those paths.
        if not skip_natten_metadata:
            # Temporal causal: encode (T, S) supertoken layout; spatial NATTEN: encode (H, W) layout.
            if video_temporal_causal:
                natten_metadata_list = generate_temporal_causal_natten_metadata(
                    vision_token_shapes=vision_token_shapes,
                    num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
                    num_layers=num_layers,
                    head_dim=head_dim,
                    device=device,
                    dtype=packed_sequence.dtype,
                    requires_grad=packed_sequence.requires_grad,
                )
            else:
                natten_metadata_list = generate_natten_metadata(
                    token_shapes=token_shapes,
                    head_dim=head_dim,
                    num_layers=num_layers,
                    device=device,
                    dtype=packed_sequence.dtype,
                    requires_grad=packed_sequence.requires_grad,
                    natten_parameter_list=natten_parameter_list,
                )
    else:
        raise ValueError(
            f"Invalid joint_attn_implementation: {joint_attn_implementation}. Must be 'two_way' or 'three_way'."
        )

    input_pack = make_pack(
        packed_sequence=packed_sequence,
        attn_modes=attn_modes,
        split_lens=split_lens,
        sample_lens=sample_lens,
        packed_und_token_indexes=packed_und_token_indexes.to(device),
        packed_gen_token_indexes=packed_gen_token_indexes.to(device),
        is_image_batch=is_image_batch,
        cp_world_size=cp_world_size,
        pad_for_cuda_graphs=pad_for_cuda_graphs,
    )
    # Not needed anymore, can cause recompilations.
    input_pack.pop("split_lens", None)
    input_pack.pop("attn_modes", None)
    return input_pack, attention_meta, natten_metadata_list
