# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.callbacks.grad_clip import GradClip
from cosmos_framework.utils.vfm.optimizer import OptimizersContainer


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moe_gen = nn.Linear(4, 4)
        self.reasoner_lora_A = nn.Linear(4, 2, bias=False)
        self.reasoner_lora_B = nn.Linear(2, 4, bias=False)
        self.frozen_backbone = nn.Linear(4, 4)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = _TinyNet()


def _groups() -> list[dict]:
    return [
        {
            "name": "generator",
            "keys_to_select": ["moe_gen"],
            "optimizer_type": "AdamW",
            "lr": 2e-5,
            "betas": [0.9, 0.95],
            "clip_norm": 0.1,
        },
        {
            "name": "reasoner",
            "keys_to_select": ["lora_"],
            "optimizer_type": "AdamW",
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "clip_norm": 1.0,
        },
    ]


def test_named_groups_build_separate_optimizers() -> None:
    model = _TinyModel()
    container = OptimizersContainer(
        model,
        "AdamW",
        lr=2e-5,
        betas=[0.9, 0.95],
        eps=1e-8,
        fused=True,
        weight_decay=0.0,
        groups=_groups(),
    )

    assert container.optimizer_group_names == ["generator", "reasoner"]
    assert container.optimizer_clip_norms == [0.1, 1.0]
    assert container.optimizers[0].defaults["lr"] == 2e-5
    assert container.optimizers[0].defaults["betas"] == (0.9, 0.95)
    assert container.optimizers[1].defaults["lr"] == 1e-4
    assert container.optimizers[1].defaults["betas"] == (0.9, 0.999)

    generator_ids = {
        id(parameter)
        for param_group in container.optimizers[0].param_groups
        for parameter in param_group["params"]
    }
    reasoner_ids = {
        id(parameter)
        for param_group in container.optimizers[1].param_groups
        for parameter in param_group["params"]
    }
    assert generator_ids.isdisjoint(reasoner_ids)
    assert not model.net.frozen_backbone.weight.requires_grad


def test_named_groups_reject_overlapping_parameters() -> None:
    groups = _groups()
    groups[1]["keys_to_select"] = ["reasoner", "moe_gen"]

    with pytest.raises(ValueError, match="optimizer groups must be disjoint"):
        OptimizersContainer(
            _TinyModel(),
            "AdamW",
            lr=2e-5,
            fused=True,
            groups=groups,
        )


def test_named_groups_are_clipped_independently() -> None:
    model = _TinyModel()
    container = OptimizersContainer(
        model,
        "AdamW",
        lr=2e-5,
        fused=True,
        groups=_groups(),
    )
    for parameter in model.net.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)

    callback = GradClip(clip_norm=0.1, force_finite=False, track_per_modality=True)
    callback.config = SimpleNamespace(trainer=SimpleNamespace(logging_iter=100))
    callback.on_before_optimizer_step(model, container, None, None, iteration=1)

    generator_norm = torch.linalg.vector_norm(
        torch.cat([parameter.grad.flatten() for parameter in model.net.moe_gen.parameters()])
    )
    reasoner_norm = torch.linalg.vector_norm(
        torch.cat(
            [
                parameter.grad.flatten()
                for name, parameter in model.net.named_parameters()
                if "lora_" in name
            ]
        )
    )
    assert generator_norm.item() == pytest.approx(0.1, rel=1e-5)
    assert reasoner_norm.item() == pytest.approx(1.0, rel=1e-5)
