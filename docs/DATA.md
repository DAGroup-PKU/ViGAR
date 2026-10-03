# Data contracts

## Action-policy data

The selected policy uses 50 tasks, 2500 episodes and 3691 curated generated goals. The goal-manifest fingerprint is `146422bf3f2c091966f5625d684656219c6ec83a93d9d7127a3cae9a0de83019`. Point `GOALWAM_GENERATED_GOAL_CACHE` at its complete manifest and image directory.

`data/native49/restore_policy_metadata.py` verifies the original HDF against measured states. `build_dataset49.py` produces canonical 49D trajectories and masks: targets use the next measured state where available; the unknown final action is invalid. Set `VIGAR_DATASET_V7` and `VIGAR_DATASET49_WORK` to your source dataset and new conversion workspace. Original trajectories are not edited in place.

Normalization is fixed, with separate state/action statistics and `bounds_99_woclip`. The included normalizer SHA256 is `043abfe02703c23b944e23ee31c1d9465edd3bf8ddafa0518668f13d136c0617`. Its historical metadata is retained; do not silently regenerate it for another horizon or population.

Stage-v7 boundary builders and task contracts are under `data/stage_v7`; supply source/output roots explicitly. Their shared offline predicate/capture helpers are packaged under `data/stage_v7/policy`, independently of the online policy. Machine-specific runtime records, historical per-episode path maps and unrelated adapter tests have been removed.

## I2I data

The normal stage consumes the declared fixed-goal GT dataset. The first ROI stage has 16167 trajectories, 15167 with aligned geometry and 1000 explicit global-loss fallbacks. Random500 has 28669 trajectories, 26919 with aligned geometry and 1750 allowed fallbacks. The 50 tasks are balanced; clean/random sampling is 1:9.

Required files include:

- `meta/info.json`, `meta/episode_manifest.json`, `meta/annotations.json`, and referenced videos;
- `state_aligned_target_metadata.jsonl.gz` under the stage's ROI workspace;
- `roi_fallback_allowlist.json` for random500;
- `dataset_index/.cache/episode_image_edit_metadata.json` under `GOALWAM_I2I_METADATA_ROOT`.

The 2500 action trajectories do not replace this larger I2I GT pool. Inference checkpoints do not include datasets or ROI metadata.

## Generated-goal preparation

`data/goal_generation` contains generation, optional VLM curation and cache-export utilities. Set `VIGAR_GOAL_GENERATION_WORK`, `VIGAR_GOAL_ADMISSION_ROOT`, `VIGAR_DATASET_V7`, and `GOALWAM_GENERATED_GOAL_CACHE` for your prepared manifests and outputs. Run generation with the I2I inference runtime and `evaluation/` on `PYTHONPATH`.

VLM curation is offline preprocessing, requiring explicit `GOALWAM_VLM_BASE_URL` and `GOALWAM_VLM_API_KEY`. It is not automatically invoked by training or online control. Optional upstream Azure instruction-generation utilities require `AZURE_INFERENCE_ENDPOINT` and `AZURE_API_KEY`; existing templates need no API request.

Simulator assets, trajectory videos, generated goal images and credentials are not redistributed in the code repository.
