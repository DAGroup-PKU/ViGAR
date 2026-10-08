from __future__ import annotations

import gc
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedGateUpMLP(nn.Module):
    """Inference-only SwiGLU with gate/up projections combined into one GEMM."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        gate_proj = source.gate_proj
        up_proj = source.up_proj
        if not isinstance(gate_proj, nn.Linear) or not isinstance(up_proj, nn.Linear):
            raise TypeError("gate_proj and up_proj must be nn.Linear")
        if gate_proj.in_features != up_proj.in_features:
            raise ValueError("gate_proj and up_proj input sizes must match")
        if gate_proj.out_features != up_proj.out_features:
            raise ValueError("gate_proj and up_proj output sizes must match")

        self.intermediate_size = int(gate_proj.out_features)
        self.fused_weight = nn.Parameter(
            torch.cat((gate_proj.weight, up_proj.weight), dim=0),
            requires_grad=False,
        )
        if gate_proj.bias is None and up_proj.bias is None:
            self.fused_bias = None
        elif gate_proj.bias is not None and up_proj.bias is not None:
            self.fused_bias = nn.Parameter(
                torch.cat((gate_proj.bias, up_proj.bias), dim=0),
                requires_grad=False,
            )
        else:
            raise ValueError("gate_proj and up_proj must use the same bias setting")
        self.down_proj = source.down_proj
        self.act_fn = source.act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(x, self.fused_weight, self.fused_bias)
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class FusedQKVProjection(nn.Module):
    """Inference-only Q/K/V projections combined into one GEMM."""

    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear) -> None:
        super().__init__()
        if not (q_proj.in_features == k_proj.in_features == v_proj.in_features):
            raise ValueError("Q/K/V projection input sizes must match")
        self.output_sizes = (
            int(q_proj.out_features),
            int(k_proj.out_features),
            int(v_proj.out_features),
        )
        self.fused_weight = nn.Parameter(
            torch.cat((q_proj.weight, k_proj.weight, v_proj.weight), dim=0),
            requires_grad=False,
        )
        biases = (q_proj.bias, k_proj.bias, v_proj.bias)
        if all(bias is None for bias in biases):
            self.fused_bias = None
        elif all(bias is not None for bias in biases):
            self.fused_bias = nn.Parameter(
                torch.cat([bias for bias in biases if bias is not None], dim=0),
                requires_grad=False,
            )
        else:
            raise ValueError("Q/K/V projections must use the same bias setting")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = F.linear(x, self.fused_weight, self.fused_bias)
        return qkv.split(self.output_sizes, dim=-1)


def _unwrap_compiled(module: nn.Module) -> nn.Module:
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


def fuse_dense_swiglu(model: Any) -> int:
    """Replace every dense MoT SwiGLU with an inference-only fused gate/up MLP."""

    net = _unwrap_compiled(model.net)
    language_model = _unwrap_compiled(net.language_model)
    fused_count = 0
    for layer in language_model.model.layers:
        block = _unwrap_compiled(layer)
        for attr in ("mlp", "mlp_moe_gen"):
            source = getattr(block, attr)
            if isinstance(source, FusedGateUpMLP):
                continue
            if not all(
                hasattr(source, required)
                for required in ("gate_proj", "up_proj", "down_proj", "act_fn")
            ):
                continue
            setattr(block, attr, FusedGateUpMLP(source))
            fused_count += 1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fused_count


def fuse_attention_qkv(model: Any) -> int:
    """Attach fused QKV projections to both pathways of every MoT layer."""

    net = _unwrap_compiled(model.net)
    language_model = _unwrap_compiled(net.language_model)
    fused_count = 0
    for layer in language_model.model.layers:
        block = _unwrap_compiled(layer)
        attention = block.self_attn
        for output_attr, suffix in (
            ("qkv_proj_fused", ""),
            ("qkv_proj_moe_gen_fused", "_moe_gen"),
        ):
            if hasattr(attention, output_attr):
                continue
            projections = tuple(
                getattr(attention, f"{name}_proj{suffix}")
                for name in ("q", "k", "v")
            )
            if not all(isinstance(projection, nn.Linear) for projection in projections):
                continue
            q_proj, k_proj, v_proj = projections
            setattr(
                attention,
                output_attr,
                FusedQKVProjection(q_proj, k_proj, v_proj),
            )
            # The fused path is installed after checkpoint loading and is
            # inference-only. Drop the source projections so their 7B-scale
            # duplicate parameter storage can be reclaimed.
            for name in ("q", "k", "v"):
                setattr(attention, f"{name}_proj{suffix}", None)
            fused_count += 1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fused_count
