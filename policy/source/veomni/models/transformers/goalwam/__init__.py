"""Lazy GoalWAM registration; native CUDA dependencies remain optional."""

from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("goalwam")
def register_goalwam_config():
    from .configuration_goalwam import GoalWAMConfig

    return GoalWAMConfig


@MODELING_REGISTRY.register("goalwam")
def register_goalwam_modeling(architecture: str):
    from .modeling_goalwam import GoalWAMPolicy

    if architecture != "GoalWAMPolicy":
        raise ValueError(f"Unsupported GoalWAM architecture: {architecture}")
    return GoalWAMPolicy
