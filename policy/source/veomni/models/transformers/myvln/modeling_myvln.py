from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    replace_return_docstrings,
)

from ....distributed.parallel_state import get_parallel_state
from ....utils import helper
from veomni.models.transformers.myvln.configuration_myvln import MyVLNConfig, MyVLNMotionConfig
from veomni.models.transformers.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel, Qwen3VLCausalLMOutputWithPast, Qwen3VLForConditionalGeneration, apply_veomni_qwen3vl_patch
)

apply_veomni_qwen3vl_patch()

logger = helper.create_logger(__name__)

_CONFIG_FOR_DOC = "MyVLNConfig"


class MyVLNForConditionalGeneration(Qwen3VLForConditionalGeneration, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: MyVLNConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Motion adaptor: maps motion_size -> intermediate -> text_hidden_size
        # The output dimension must match text_config.hidden_size to be compatible with inputs_embeds
        self.motion_adaptor = torch.nn.Sequential(
            nn.Linear(config.motion_config.motion_size, config.motion_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.motion_config.hidden_size, config.text_config.hidden_size),
        )

        self.post_init()

    @replace_return_docstrings(output_type=Qwen3VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        motion_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        motion_history: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        motion_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask indicating positions of motion_pad_history_token_id in input_ids. Used for sequence parallel.
        motion_history (`torch.Tensor` of shape `(M, motion_size)`, *optional*):
            The motion history tensor where M is the number of motion tokens and motion_size is typically 3.

        Returns:

        """
        # Pre-compute motion_mask and zero out motion tokens BEFORE embedding
        # lookup to prevent out-of-bounds access (motion token IDs may exceed
        # the embedding table's vocab_size, similar to how image/video pad
        # tokens are zeroed out in the data processor).
        if motion_history is not None and motion_mask is None:
            if input_ids is None:
                raise ValueError(
                    "motion_mask must be provided when motion_history is not None and input_ids is None"
                )
            motion_mask = input_ids == self.config.motion_pad_history_token_id

        # Clamp any remaining out-of-bounds token IDs (e.g. motion_token_*
        # IDs that exceed the embedding table) to prevent CUDA device-side
        # asserts.  This is a safety net; the proper fix is to ensure that
        # resize_token_embeddings was called so the embedding table covers
        # the full vocabulary.
        if input_ids is not None:
            vocab_size = self.get_input_embeddings().num_embeddings
            oob_mask = input_ids >= vocab_size
            if oob_mask.any():
                logger.warning(
                    f"input_ids contains {oob_mask.sum().item()} token(s) >= vocab_size "
                    f"({vocab_size}). Clamping to 0. Consider resizing token embeddings."
                )
                input_ids[oob_mask] = 0

        # Get inputs_embeds first so we can scatter motion embeddings into it
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Process motion_history: M x motion_size -> M x hidden_size
        if motion_history is not None:
            # Validate and prepare motion_history
            expected_motion_size = self.config.motion_config.motion_size
            if motion_history.dim() != 2:
                raise ValueError(
                    f"motion_history must be 2D (M, motion_size), got shape {motion_history.shape}"
                )
            if motion_history.shape[-1] != expected_motion_size:
                raise ValueError(
                    f"motion_history last dim must be {expected_motion_size}, got {motion_history.shape[-1]}"
                )
            
            # Ensure motion_history is on the correct device and dtype
            # Use the first layer of motion_adaptor to get target device and dtype
            target_weight = self.motion_adaptor[0].weight
            motion_history = motion_history.to(device=target_weight.device, dtype=target_weight.dtype)
            
            motion_hidden_states = self.motion_adaptor(motion_history)

            # Validate that the number of motion tokens matches the motion features
            n_motion_tokens = motion_mask.sum().item()
            n_motion_features = motion_hidden_states.shape[0]
            if n_motion_tokens != n_motion_features:
                raise ValueError(
                    f"Motion features and motion tokens do not match: tokens: {n_motion_tokens}, features {n_motion_features}"
                )

            # Expand mask to match inputs_embeds shape and scatter motion embeddings
            motion_mask_expanded = motion_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            motion_hidden_states = motion_hidden_states.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(motion_mask_expanded, motion_hidden_states)
        elif get_parallel_state().fsdp_enabled:
            # Dummy forward of motion_adaptor to ensure all parameters participate in the forward pass
            # This is needed when some ranks have motion_history=None while others have valid motion_history
            dummy_motion = torch.zeros(
                (1, self.config.motion_config.motion_size),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device
            )
            dummy_motion_hidden = self.motion_adaptor(dummy_motion)
            dummy_motion_hidden = dummy_motion_hidden.mean() * 0.0
            inputs_embeds = inputs_embeds + dummy_motion_hidden

        # Set input_ids to None since we're passing inputs_embeds directly
        input_ids = None


        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            image_mask=image_mask,
            video_mask=video_mask,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        hidden_states = hidden_states[:, slice_indices, :]
        loss = None
        logits = None
        if labels is not None:
            loss, logits = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        image_mask=None,
        video_mask=None,
        motion_history=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation, handling motion_history and visual masks along with visual inputs.
        
        motion_history, image_mask, and video_mask are only used during the first forward pass (prefill stage).
        After that, we rely on the cached KV values and don't need to re-process them.
        """
        # Call parent's prepare_inputs_for_generation (from Qwen3VLForConditionalGeneration)
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # Handle prefill vs decode stages
        # During decode steps, visual embeddings and motion embeddings are already cached
        if cache_position is not None and cache_position[0] != 0:
            # This is a decode step (not prefill), don't pass these inputs
            model_inputs["motion_history"] = None
            model_inputs["image_mask"] = None
            model_inputs["video_mask"] = None
        else:
            # This is the prefill step, pass all inputs
            model_inputs["motion_history"] = motion_history
            model_inputs["image_mask"] = image_mask
            model_inputs["video_mask"] = video_mask

        return model_inputs


# Modification:
# Register the ModelClass which is used by veOmni to tell which class to match the config.json architecture
ModelClass = MyVLNForConditionalGeneration

__all__ = [
    "MyVLNForConditionalGeneration",
]
