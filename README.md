# ViGAR

**Rethinking World-Action Model for Compositional and In-context Robotic Manipulation**

[Model weights](https://huggingface.co/DAGroup-PKU/ViGAR) · [Setup](docs/SETUP.md) · [Data preparation](docs/DATA.md) · [Validation](docs/VALIDATION.md)

This release contains the RoboTwin simulation pipeline and image-goal planner: data preparation, native action-policy training, I2I training, inference services and closed-loop evaluation. Ablation runs, RoboCasa, infrastructure management, credentials and private experiment records are excluded.

## Released configuration

| Component | Configuration |
|---|---|
| Action policy | `robotwin_c2r`: three-view native 49D, selected 50k EMA |
| I2I planner | Normal 30k initialization, end-effector 4:1 ROI training, random500 continuation to 70k; regular weights |
| Evaluation | Current-observation Strict Sync; 50 tasks × 2 splits × 50 episodes |
| Policy sampling | UniPC 10 steps, guidance 1, shift 2; predict 48 / execute 32 |
| I2I sampling | 35 steps, guidance 2.5, shift 5; RGB 384×320 |

The release combines the `robotwin_c2r` policy with the ROI70k planner. Results obtained with other planner checkpoints do not establish a score for this combination. See the [configuration](configs/selected_release.json) and [validation record](docs/VALIDATION.md).

## Getting started

```bash
git clone https://github.com/DAGroup-PKU/ViGAR.git
cd ViGAR
hf download DAGroup-PKU/ViGAR --local-dir /path/to/vigar-weights
cp .env.example .env
# Edit .env with your model, dataset, runtime and output paths.
```

The model and simulator use separate environments. Policy and I2I runtimes share a Python package name but contain different pinned implementations; launchers isolate their import paths. Legacy `goalwam` identifiers remain for checkpoint compatibility.

On already allocated resources:

```bash
# Action training: 2 nodes × 8 GPUs. Set NODE_RANK=0/1, MASTER_ADDR and MASTER_PORT.
bash scripts/train_policy.sh

# I2I lineage; the third stage resumes the complete ROI40k training state.
bash scripts/train_i2i_normal.sh
bash scripts/train_i2i_roi.sh roi40k
bash scripts/train_i2i_roi.sh random500

# Prepare a fresh evaluation directory, then launch on both nodes.
source scripts/env.sh
python scripts/prepare_eval.py --output "$GOALWAM_WORKSPACE/eval"
export NATIVE_EVAL_ROOT="$GOALWAM_WORKSPACE/eval"
# Set EVAL_NODE_RANK=0 or 1 on the corresponding node.
bash scripts/evaluate.sh
```

The released weights are inference exports. Reproducing the full training lineage also requires foundation weights, intermediate training states and datasets; they cannot be reconstructed from inference exports.

## Layout

```text
policy/       Native model, training, transforms, normalization and policy server
i2i/          Plain I2I and two ROI training stages; separate pinned runtimes
evaluation/   Strict Sync, RPC, checked CUDA graphs and resumable seed queue
simulation/   RoboTwin task environments, instruction templates and configuration
data/         Final task configuration, stage annotation, 49D action conversion and goal generation
configs/      Selected recipe and fixed evaluation seeds/instructions
scripts/      Training, preparation, evaluation and integrity entrypoints
tests/        Release contract tests
licenses/     Additional upstream license texts
```

Offline data preparation uses one [final task configuration](data/task_config.json): **19 multi-stage tasks + 31 final-goal tasks**. Stage annotation feeds goal generation directly; 49D action conversion is a separate representation step. Run `python data/pipeline.py summary` to inspect the configuration.

Simulator assets, trajectories and generated-goal caches must be installed separately. See [data preparation](docs/DATA.md).

## License

The root [MIT license](LICENSE) applies to original project contributions. Bundled third-party code and model materials retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). The code project's MIT license does not replace the OpenMDW terms of Cosmos-derived materials.
