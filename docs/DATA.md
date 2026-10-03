# Data preparation

## One final task configuration

[`data/task_config.json`](../data/task_config.json) is the shared entry point for stage annotation and goal generation. It defines **19 multi-stage tasks + 31 final-goal tasks** (19 个多阶段任务 + 31 个最终目标任务), with 50 clean demonstrations per task: 2,500 episodes in total.

- For the 19 multi-stage tasks, expert replay finds the first recorded frame satisfying each intermediate stage predicate. The final stage uses official task success.
- For the 31 final-goal tasks, one segment spans the episode, the target is its last recorded frame, and the episode's own task instruction is preserved. No intermediate boundary extraction is needed.
- `place_bread_basket` has a variable number of stages: an episode containing one bread uses only its terminal stage. The task remains in the multi-stage group. There are 74 task/stage definitions; the selected 2,500 demonstrations contain 3,691 observed stage slots.

These are task annotation choices, not action dimensions. The separate `action_conversion/` tools produce the model's **49D state/action representation**, using measured joints, grippers and end-effector poses with validity masks. Both representations are used together during policy training.

```text
data/
  task_config.json       Final task modes, stage texts and predicate descriptions
  task_config.py         Shared validation and configuration fingerprint
  pipeline.py            One command entry point; each step runs explicitly
  stage_annotation/      Expert boundary extraction and direct dataset builder
  goal_generation/       Prepare, generate, review, export and verify goals
  action_conversion/     Convert the same trajectories to canonical 49D
```

The builder consumes the original LeRobot dataset, raw HDF5 trajectories and final-task boundaries directly. It does not require an older annotated dataset or a version-to-version conversion. Repeated historical manifests, migration scripts and checkpoint-specific selection records are no longer inputs. Configuration fingerprints bind the boundaries, annotations and generated goals together.

## 1. Extract boundaries and build annotations

Use your installed RoboTwin simulator environment for replay; it needs the original assets, `seed.txt`, HDF5 files and `_traj_data/episode*.pkl`. Use the data-processing environment with NumPy, h5py, PyArrow, Pillow, OpenCV and PyYAML for the CPU steps. Add SciPy for 49D conversion. Set `VIGAR_ROOT` to this checkout's absolute path and load your edited `.env` with `source scripts/env.sh` from a Bash shell.

```bash
python "$VIGAR_ROOT/data/pipeline.py" summary

# Run from the simulator root so RoboTwin can resolve its assets and envs package.
(cd "$ROBOTWIN_ROOT" && "$NATIVE_SIM_PYTHON" "$VIGAR_ROOT/data/pipeline.py" extract \
  --raw-root "$VIGAR_RAW_TRAJECTORIES" \
  --sim-config "$VIGAR_ROOT/data/stage_annotation/boundary_extraction/demo_clean_50.yml" \
  --output "$VIGAR_STAGE_BOUNDARIES")

python "$VIGAR_ROOT/data/pipeline.py" annotate \
  --base-root "$VIGAR_SOURCE_DATASET" \
  --raw-root "$VIGAR_RAW_TRAJECTORIES" \
  --stage-boundaries "$VIGAR_STAGE_BOUNDARIES" \
  --output-root "$VIGAR_ANNOTATED_DATASET"
```

Extraction covers **19 × 50 = 950** replay records. `--task` and `--episode-start/--episode-end` support sharding; use `pipeline.py merge --input shard.jsonl ... --output boundaries.jsonl --summary summary.json` before annotation. `--resume` validates the recorded configuration before appending. The explicit `--allow-recorded-terminal-on-replay-failure` option retains the existing documented replay-drift exception; it is off by default.

The builder writes `meta/annotations.json`, `family_mapping_v1.json`, `goal_graph.json`, `stage_contract.json`, `episode_manifest.json`, `source_action_stats.json`, a boundary audit and a completion receipt. Original Parquet/video payloads are linked rather than copied. The source statistics describe the original 14D trajectories; they do not replace the policy's fixed 49D normalizer.

The offline predicates select training targets. Public closed-loop evaluation continues to use goals generated from the current observation, with no oracle-stage switching.

## 2. Convert action trajectories

```bash
python "$VIGAR_ROOT/data/pipeline.py" recover-actions
python "$VIGAR_ROOT/data/pipeline.py" convert-actions
```

These steps read `VIGAR_ANNOTATED_DATASET` and write to the new `VIGAR_DATASET49_WORK` directory. Recovery verifies the original HDF against measured states. Conversion creates `dataset49/` and `dataset49_manifest.json`; targets use the next measured state, and the unknown final action is invalid. Original trajectories are not edited. Set `GOALWAM_DATASET_MANIFEST` to the resulting manifest.

Normalization remains fixed: separate state/action statistics, `bounds_99_woclip`, SHA256 `043abfe02703c23b944e23ee31c1d9465edd3bf8ddafa0518668f13d136c0617`. The normalizer and pretrained weights are unchanged by this preparation refactor.

## 3. Prepare and generate goals

```bash
python "$VIGAR_ROOT/data/pipeline.py" prepare-goals \
  --dataset "$VIGAR_ANNOTATED_DATASET" \
  --checkpoint "$GOALWAM_I2I_CHECKPOINT" \
  --output "$VIGAR_GOAL_GENERATION_WORK" --candidates 4

# Use the installed I2I inference environment on already allocated GPUs.
export PYTHONPATH="$VIGAR_ROOT/i2i/inference_runtime:$VIGAR_ROOT/evaluation${PYTHONPATH:+:$PYTHONPATH}"
export CURATION_GPU_LIST=0,1,2,3,4,5,6,7
"$PYTHON_BIN" "$VIGAR_ROOT/data/pipeline.py" generate
```

Preparation produces one case per observed stage from the same final annotations: input at stage start, expert reference at `end_frame - 1`, and deterministic candidate seeds. It uses the official RoboTwin encoded-RGB convention and a 320×384 three-camera canvas. Generation receives the input image and episode task prompt; expert reference images are used only for offline review. The generation contract records the checkpoint actually supplied and its DCP metadata hash. It does not assign a historical checkpoint label to new images.

## 4. Review, export and verify

```bash
# Local review works without a VLM service. Open the resulting review.html.
python "$VIGAR_ROOT/data/pipeline.py" review report --root "$VIGAR_GOAL_GENERATION_WORK"

# Optional: explicitly configure a compatible VLM endpoint, key and model first.
# GOALWAM_VLM_BASE_URL, GOALWAM_VLM_API_KEY, GOALWAM_VLM_MODEL
# python "$VIGAR_ROOT/data/pipeline.py" review score --root "$VIGAR_GOAL_GENERATION_WORK"

# In review.html, choose one generated candidate per stage and export human_decisions.json.
python "$VIGAR_ROOT/data/pipeline.py" export \
  --root "$VIGAR_GOAL_GENERATION_WORK" --decisions /path/to/human_decisions.json \
  --dataset "$VIGAR_ANNOTATED_DATASET" --output "$GOALWAM_GENERATED_GOAL_CACHE"
python "$VIGAR_ROOT/data/pipeline.py" verify \
  --cache "$GOALWAM_GENERATED_GOAL_CACHE" --dataset "$VIGAR_ANNOTATED_DATASET"
```

The export requires exactly one selected generated candidate for every annotated stage. It records actual selections and available scoring provenance, and rejects unreviewed stages, missing coverage, changed hashes and expert-image fallbacks. A cache becomes visible at the requested output path only after verification succeeds. `manifest.json`, `FROZEN.json` and `VERIFIED.json` record the result.

The released policy was trained on 3,691 curated generated goals, with frozen goal-manifest fingerprint `146422bf3f2c091966f5625d684656219c6ec83a93d9d7127a3cae9a0de83019`. Newly prepared metadata and regenerated images have their own hashes; this workflow does not recreate that immutable cache byte for byte. The training reader obtains stage indices and target frames from the goal cache alongside the 49D trajectory dataset.

## I2I training data

The 2,500 policy trajectories are distinct from the larger I2I training pool. The initial stage consumes the declared fixed-goal GT dataset. The first ROI stage has 16,167 trajectories, 15,167 with aligned geometry and 1,000 explicit global-loss fallbacks. Random500 has 28,669 trajectories, 26,919 with aligned geometry and 1,750 allowed fallbacks. The 50 tasks are balanced; clean/random sampling is 1:9.

Required files include `meta/info.json`, `meta/episode_manifest.json`, `meta/annotations.json` and referenced videos; `state_aligned_target_metadata.jsonl.gz` under the ROI workspace; `roi_fallback_allowlist.json` for random500; and `dataset_index/.cache/episode_image_edit_metadata.json` under `GOALWAM_I2I_METADATA_ROOT`.

Inference checkpoints do not contain these datasets or ROI metadata. Simulator assets, trajectory videos, generated goal images and credentials are not redistributed in the code repository.
