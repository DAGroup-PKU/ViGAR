"""Lazy ViGAR registration; native CUDA dependencies remain optional."""

from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("vigar")
def register_vigar_config():
    from .configuration_vigar import ViGARConfig

    return ViGARConfig


@MODELING_REGISTRY.register("vigar")
def register_vigar_modeling(architecture: str):
    from .modeling_vigar import ViGARPolicy

    if architecture != "ViGARPolicy":
        raise ValueError(f"Unsupported ViGAR architecture: {architecture}")
    return ViGARPolicy
