# Training

[← README](../README.md) · [Installation](installation.md) · [Download](download.md) · [Evaluation](evaluation.md) · **Training**

## Post-training RoboTwin policy

Uses the base checkpoint and training data from [Download](download.md). Also set in `.env`:

```bash
VIGAR_POLICY_WORKSPACE=/path/to/vigar-workspace/policy
```

```bash
source scripts/env.sh

# One machine, 8 GPUs x 32 samples
bash scripts/train_policy.sh

# Two machines, 16 GPUs x 16 samples (NODE_RANK=0 and 1)
NNODES=2 NODE_RANK=0 MASTER_ADDR=192.0.2.10 MASTER_PORT=29500 bash scripts/train_policy.sh --batch 16
```

Global batch 256, learning rate 1e-5, 50,000 updates, a checkpoint every 4,000 updates.
Outputs are written to `$VIGAR_POLICY_WORKSPACE/runs/robotwin_c2r`. Set `WANDB_MODE=online` to log to W&B.

### Rebuilding the training data

Optional; the downloaded data is ready to use. These steps rebuild it from raw RoboTwin
trajectories, using the `VIGAR_SOURCE_DATASET`, `VIGAR_RAW_TRAJECTORIES` and work-directory
paths in `.env`.

Stage annotations:

```bash
source scripts/env.sh
"$PYTHON_BIN" data/pipeline.py summary
(cd "$ROBOTWIN_ROOT" && "$NATIVE_SIM_PYTHON" "$VIGAR_ROOT/data/pipeline.py" extract \
  --raw-root "$VIGAR_RAW_TRAJECTORIES" \
  --sim-config "$VIGAR_ROOT/data/stage_annotation/boundary_extraction/demo_clean_50.yml" \
  --output "$VIGAR_STAGE_BOUNDARIES")
"$PYTHON_BIN" data/pipeline.py annotate --base-root "$VIGAR_SOURCE_DATASET" \
  --raw-root "$VIGAR_RAW_TRAJECTORIES" --stage-boundaries "$VIGAR_STAGE_BOUNDARIES" \
  --output-root "$VIGAR_ANNOTATED_DATASET"
```

49-D actions (writes `dataset49_manifest.json`, the new `VIGAR_DATASET_MANIFEST`):

```bash
"$PYTHON_BIN" data/pipeline.py recover-actions
"$PYTHON_BIN" data/pipeline.py convert-actions
```

Goal images (select one image per stage in `review.html`, then export the decisions):

```bash
"$PYTHON_BIN" data/pipeline.py prepare-goals --dataset "$VIGAR_ANNOTATED_DATASET" \
  --checkpoint "$SUBGOAL_PLANNER_CHECKPOINT" --output "$VIGAR_GOAL_GENERATION_WORK" --candidates 4
export PYTHONPATH="$VIGAR_ROOT/subgoal_planner/inference_runtime:$VIGAR_ROOT/evaluation"
export CURATION_GPU_LIST=0,1,2,3,4,5,6,7
"$PYTHON_BIN" data/pipeline.py generate
"$PYTHON_BIN" data/pipeline.py review report --root "$VIGAR_GOAL_GENERATION_WORK"
"$PYTHON_BIN" data/pipeline.py export --root "$VIGAR_GOAL_GENERATION_WORK" \
  --decisions /path/to/human_decisions.json --dataset "$VIGAR_ANNOTATED_DATASET" \
  --output "$VIGAR_GENERATED_GOAL_CACHE"
"$PYTHON_BIN" data/pipeline.py verify --cache "$VIGAR_GENERATED_GOAL_CACHE" \
  --dataset "$VIGAR_ANNOTATED_DATASET"
```

## Post-training Real-robot Policy

Uses the base checkpoint from [Download](download.md) and the `real_robot_900hr` data
(AgiBot A2). Also set in `.env`:

```bash
VIGAR_REAL_ROBOT_DATASET=/path/to/vigar/data/real_robot_900hr
VIGAR_POLICY_WORKSPACE=/path/to/vigar-workspace/policy
VIGAR_PARQUET_CACHE=/path/to/vigar-workspace/parquet-cache
```

```bash
source scripts/env.sh

# One machine, 8 GPUs x 128 samples
bash scripts/train_policy_real_robot.sh

# Two machines, 16 GPUs x 64 samples (NODE_RANK=0 and 1)
NNODES=2 NODE_RANK=0 MASTER_ADDR=192.0.2.10 MASTER_PORT=29500 bash scripts/train_policy_real_robot.sh --batch 64
```

34-D absolute actions (two 7-joint arms, two 10-joint hands) in 128-step chunks at 30 Hz,
with 17 video frames per chunk. The goal image is the last frame of the current subtask, or
of the next subtask during the final 15% of the current one. Episodes of the waist-turning
task 9734 are excluded. Global batch 1024, learning rate 1e-4, 100,000 updates, a checkpoint
every 10,000 updates. Outputs are written to `$VIGAR_POLICY_WORKSPACE/runs/real_robot_900hr`.

## Training Real-robot subgoal planner

Predicts the next subgoal image from the current frame and the task instruction, on the same
`real_robot_900hr` data and base checkpoint:

```bash
source scripts/env.sh

# One machine, 8 GPUs x 128 samples
bash scripts/train_planner_real_robot.sh normal

# Two machines, 16 GPUs x 64 samples (NODE_RANK=0 and 1)
NNODES=2 NODE_RANK=0 MASTER_ADDR=192.0.2.10 MASTER_PORT=29500 SAMPLES_PER_GPU=64 \
  bash scripts/train_planner_real_robot.sh normal
```

Every annotated frame is an input. Its target is the last frame of the current subtask, or of
the next subtask during the final 15% of the current one. Global batch 1024, learning rate
2e-5, up to 200,000 updates, a checkpoint every 10,000 updates. Outputs are written to
`$VIGAR_WORKSPACE/planner-real-robot-normal`.

## Training ICL subgoal planner

Also conditions the planner on a goal image: the last frame of the episode during training,
an image of the desired final state at inference. The instruction becomes
"Refer to the goal frame to generate the subgoal image.", so the goal image alone defines
the task.

```bash
source scripts/env.sh
bash scripts/train_planner_real_robot.sh icl
```

The goal image enters both the reasoner and the generator. Targets, schedule and batch
settings are the same as above. Outputs are written to `$VIGAR_WORKSPACE/planner-real-robot-icl`.
