# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import torch

from cosmos_framework.model.vfm import omni_mot_model as model_module
from cosmos_framework.model.vfm.omni_mot_model import OmniMoTModel


def test_setup_parallelism_passes_hsdp_replicate_degree(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeParallelDims:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def build_meshes(self, *, device_type: str) -> None:
            captured["device_type"] = device_type

    monkeypatch.setattr(model_module, "ParallelDims", FakeParallelDims)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 16)

    model = OmniMoTModel.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        parallelism=SimpleNamespace(
            enable_inference_mode=False,
            data_parallel_shard_degree=8,
            data_parallel_replicate_degree=2,
            cfg_parallel_shard_degree=1,
            context_parallel_shard_degree=1,
        )
    )

    model.set_up_parallelism()

    assert captured == {
        "enable_inference_mode": False,
        "world_size": 16,
        "dp_shard": 8,
        "dp_replicate": 2,
        "cfgp": 1,
        "cp": 1,
        "device_type": model_module.DEVICE,
    }
