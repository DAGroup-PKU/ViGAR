# Evaluation

[← README](../README.md) · [Installation](installation.md) · [Download](download.md) · **Evaluation** · [Training](training.md)

RoboTwin 2.0 Clean2Random: 50 tasks, 100 Clean and 100 Random episodes per task.
The default run uses one machine with 8 GPUs (4 policy/planner pairs).

```bash
source scripts/env.sh
"$PYTHON_BIN" scripts/prepare_eval.py --output "$VIGAR_WORKSPACE/eval"
export NATIVE_EVAL_ROOT="$VIGAR_WORKSPACE/eval"
bash scripts/evaluate.sh
```

| `prepare_eval.py` option | Effect |
| :--- | :--- |
| `--splits SPLIT` | Evaluate one split: `clean` or `random` |
| `--tasks TASK ...` | Evaluate selected tasks |
| `--episodes N` | Episodes per task and split (default 100) |
| `--gpus-per-node N` | Use 2 or 4 GPUs, selected with `CUDA_VISIBLE_DEVICES` |
| `--simulators-per-pair N` | Simulators per policy/planner pair (default 1) |
| `--paired-episodes FILE` | Reuse the seeds and instructions of an earlier run |
| `--fast-inference` | CUDA Graph compilation and kernel fusion (default off) |

`--fast-inference` enables accelerated evaluation that is numerically equivalent to the default up to BF16 rounding. Success rates may differ slightly from the default.

## Multiple machines

Prepare once with `--nodes 2` in a directory shared by both machines, then run:

```bash
EVAL_NODE_RANK=0 bash scripts/evaluate.sh    # EVAL_NODE_RANK=1 on the second machine
```

## Results

| File | Content |
| :--- | :--- |
| `<output>/robotwin_c2r/results/<split>/<task>/summary.json` | Per-task episodes and success |
| `<output>/robotwin_c2r/summary.json` | Clean and Random success rates |
| `<output>/robotwin_c2r/episodes.json` | Seeds and instructions used |

Rerun `bash scripts/evaluate.sh` with the same `NATIVE_EVAL_ROOT` to resume.
If a machine was lost, stop all workers and run `bash scripts/evaluate.sh --reset-locks` once on node 0.

<!-- As in the RoboTwin evaluator, seeds whose expert demonstration fails are skipped.
Policy: 10 sampling steps, guidance 1, shift 2, 32 of 48 predicted actions executed.
Planner: 35 sampling steps, guidance 2.5, shift 5. -->
