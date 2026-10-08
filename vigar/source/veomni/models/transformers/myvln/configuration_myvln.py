from typing import Optional

from transformers import PretrainedConfig

# TODO: add RopeParameters when transformers 4.57.0 is released
# from transformers.modeling_rope_utils import RopeParameters, rope_config_validation, standardize_rope_params
from transformers.modeling_rope_utils import rope_config_validation
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig, Qwen3VLTextConfig


class MyVLNMotionConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.

    """

    model_type = "myvln_motion"
    base_config_key = "motion_config"

    def __init__(
        self,
        motion_size: Optional[int] = 3,
        hidden_size: Optional[int] = 4096,
        intermediate_size: Optional[int] = 8192,
        num_hidden_layers: Optional[int] = 4,
        **kwargs,
    ):
        self.motion_size = motion_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        super().__init__(**kwargs)



class MyVLNConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        text_config (`Union[PretrainedConfig, dict]`, *optional*, defaults to `Qwen3VLTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PretrainedConfig, dict]`,  *optional*, defaults to `Qwen3VLVisionConfig`):
            The config object or dictionary of the vision backbone.
        action_config (`Union[PretrainedConfig, dict]`, *optional*, defaults to `ActionConfig`):
            The config object or dictionary of the action backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The start token index to encode the image prompt.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The end token index to encode the image prompt.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the word embeddings.

    ```python
    >>> from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig

    >>> # Initializing a Qwen3-VL style configuration
    >>> configuration = Qwen3VLConfig()

    >>> # Initializing a model from the Qwen3-VL-4B style configuration
    >>> model = Qwen3VLForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "myvln"
    sub_configs = {"vision_config": Qwen3VLVisionConfig, "text_config": Qwen3VLTextConfig, "motion_config": MyVLNMotionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        motion_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        motion_start_token_id=151669,
        motion_pad_history_token_id=151670,
        motion_pad_future_token_id=151671,
        motion_end_token_id=151672,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        if isinstance(motion_config, dict):
            self.motion_config = self.sub_configs["motion_config"](**motion_config)
        elif motion_config is None:
            self.motion_config = self.sub_configs["motion_config"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.motion_start_token_id = motion_start_token_id
        self.motion_pad_history_token_id = motion_pad_history_token_id
        self.motion_pad_future_token_id = motion_pad_future_token_id
        self.motion_end_token_id = motion_end_token_id
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


__all__ = ["MyVLNConfig", "MyVLNMotionConfig"]
