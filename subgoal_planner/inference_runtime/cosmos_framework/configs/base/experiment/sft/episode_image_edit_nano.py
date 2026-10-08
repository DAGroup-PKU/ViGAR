# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Nano image-to-image SFT recipe for local episode datasets."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.vision_sft_nano import vision_sft_nano
from cosmos_framework.data.vfm.dataflow import (
    CosmosDataLoader,
    IdentityProcessor,
    MapDistributor,
    SequentialPackingBatcher,
    VFMListCollator,
)
from cosmos_framework.data.vfm.local_datasets.episode_image_edit_dataset import get_episode_image_edit_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L

cs = ConfigStore.instance()


episode_image_edit_nano = copy.deepcopy(vision_sft_nano)
episode_image_edit_nano.job.name = "episode_image_edit_nano"
episode_image_edit_nano.model.config.resolution = "256"
# Offline three-camera composites are 320x384 (AgiBot) or 384x320 (RoboTwin) and
# therefore belong to the framework's 480 resolution tier. Keep
# the legacy 256 tier for single-camera opt-out datasets and add the standard Nano 480
# shift so both paths can share this recipe.
episode_image_edit_nano.model.config.rectified_flow_training_config.shift = {"256": 3, "480": 5}
episode_image_edit_nano.model.config.vlm_config.tokenizer.pretrained_model_name = "${oc.env:QWEN_TOKENIZER_PATH}"
episode_image_edit_nano.model.config.vlm_config.tokenizer.config_variant = "hf"
# Episode image editing always conditions the frozen reasoner on the current/source frame plus the
# episode-level task. The generator consumes that causal K/V, but its flow loss cannot update the
# reasoner tower. Explicit reasoner co-training recipes may add their own CE/LoRA trainable surface.
episode_image_edit_nano.model.config.reasoner_stop_gradient = True
episode_image_edit_nano.dataloader_train = L(CosmosDataLoader)(
    distributor=L(MapDistributor)(
        dataset=L(get_episode_image_edit_dataset)(
            dataset_dir="${oc.env:EPISODE_IMAGE_EDIT_DATASET_PATH}",
            video_subdir="video",
            instruction_subdir="instructions",
            video_glob="episode*.mp4",
            instruction_keys=None,
            seed=42,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            max_caption_tokens=1024,
            cfg_dropout_rate=0.0,
            resize_resolution="256",
            include_final_as_input=True,
            dataset_name="episode_image_editing",
            metadata_num_workers=16,
            dataset_format="${oc.env:EPISODE_IMAGE_EDIT_DATASET_FORMAT,auto}",
            lerobot_video_key="${oc.env:EPISODE_IMAGE_EDIT_LEROBOT_VIDEO_KEY,observation.images.cam_high}",
            use_three_camera="${oc.env:EPISODE_IMAGE_EDIT_THREE_CAMERA,true}",
            target_mode="${oc.env:EPISODE_IMAGE_EDIT_TARGET_MODE,episode_final}",
            lerobot_segment_source="${oc.env:EPISODE_IMAGE_EDIT_SEGMENT_SOURCE,auto}",
            skip_mistake_segments=True,
            reasoner_subtask_target="${oc.env:EPISODE_IMAGE_EDIT_REASONER_TARGET,false}",
            next_subgoal_tail_fraction="${oc.env:EPISODE_IMAGE_EDIT_NEXT_SUBGOAL_TAIL_FRACTION,0.15}",
            goal_condition_mode="${oc.env:EPISODE_IMAGE_EDIT_GOAL_CONDITION_MODE,none}",
            goal_prompt="${oc.env:EPISODE_IMAGE_EDIT_GOAL_PROMPT,Refer to the goal frame to generate the subgoal image.}",
        ),
        shuffle=True,
        seed=42,
        name="episode_image_edit",
    ),
    processor=L(IdentityProcessor)(),
    batcher=L(SequentialPackingBatcher)(
        max_sequence_length=None,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_samples_per_batch=1,
        sound_latent_fps=0,
        audio_sample_rate=48000,
    ),
    collator=L(VFMListCollator)(),
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=True,
)
episode_image_edit_nano.dataloader_val = L(CosmosDataLoader)(
    distributor=L(MapDistributor)(
        dataset=L(get_episode_image_edit_dataset)(
            dataset_dir="${oc.env:EPISODE_IMAGE_EDIT_DATASET_PATH}",
            video_subdir="video",
            instruction_subdir="instructions",
            video_glob="episode*.mp4",
            instruction_keys=None,
            seed=4242,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            max_caption_tokens=1024,
            cfg_dropout_rate=0.0,
            resize_resolution="256",
            include_final_as_input=True,
            first_frame_only=True,
            dataset_name="episode_image_editing_val",
            metadata_num_workers=16,
            dataset_format="${oc.env:EPISODE_IMAGE_EDIT_DATASET_FORMAT,auto}",
            lerobot_video_key="${oc.env:EPISODE_IMAGE_EDIT_LEROBOT_VIDEO_KEY,observation.images.cam_high}",
            use_three_camera="${oc.env:EPISODE_IMAGE_EDIT_THREE_CAMERA,true}",
            target_mode="${oc.env:EPISODE_IMAGE_EDIT_TARGET_MODE,episode_final}",
            lerobot_segment_source="${oc.env:EPISODE_IMAGE_EDIT_SEGMENT_SOURCE,auto}",
            skip_mistake_segments=True,
            reasoner_subtask_target="${oc.env:EPISODE_IMAGE_EDIT_REASONER_TARGET,false}",
            next_subgoal_tail_fraction="${oc.env:EPISODE_IMAGE_EDIT_NEXT_SUBGOAL_TAIL_FRACTION,0.15}",
            goal_condition_mode="${oc.env:EPISODE_IMAGE_EDIT_GOAL_CONDITION_MODE,none}",
            goal_prompt="${oc.env:EPISODE_IMAGE_EDIT_GOAL_PROMPT,Refer to the goal frame to generate the subgoal image.}",
        ),
        shuffle=False,
        seed=4242,
        name="episode_image_edit_val",
    ),
    processor=L(IdentityProcessor)(),
    batcher=L(SequentialPackingBatcher)(
        max_sequence_length=None,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_samples_per_batch=1,
        sound_latent_fps=0,
        audio_sample_rate=48000,
    ),
    collator=L(VFMListCollator)(),
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=True,
)

cs.store(group="experiment", package="_global_", name="episode_image_edit_nano", node=episode_image_edit_nano)
