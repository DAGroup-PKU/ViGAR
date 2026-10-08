from transformers import AutoConfig, AutoModelForCausalLM

from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("myvln")
def register_myvln_model_config():
    from .configuration_myvln import MyVLNConfig

    return MyVLNConfig


@MODELING_REGISTRY.register("myvln")
def register_myvln_modeling(architecture: str):
    from .modeling_myvln import MyVLNForConditionalGeneration

    return MyVLNForConditionalGeneration


if not is_transformers_version_greater_or_equal_to("4.57.0"):
    from .configuration_myvln import MyVLNConfig
    from .modeling_myvln import MyVLNForConditionalGeneration

    AutoConfig.register("myvln", MyVLNConfig)
    AutoModelForCausalLM.register(MyVLNConfig, MyVLNForConditionalGeneration)


__all__ = [
    "register_myvln_model_config",
    "register_myvln_modeling",
]
