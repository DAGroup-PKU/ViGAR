# Download

[← README](../README.md) · [Installation](installation.md) · **Download** · [Evaluation](evaluation.md) · [Training](training.md)

Weights and training data are in [DAGroup-PKU/ViGAR](https://huggingface.co/DAGroup-PKU/ViGAR) (about 96 GB):

```bash
.venv-model/bin/hf download DAGroup-PKU/ViGAR --local-dir /path/to/vigar
```

```text
vigar/
├── weights/
│   ├── policy/robotwin_c2r/                       # action policy, Wan2.2 VAE, Qwen3-VL tokenizer
│   ├── subgoal_planner/robotwin_subgoal_planner/  # subgoal planner
│   └── Cosmos3-Nano-dcp/                          # base checkpoint for training
└── data/
    ├── robotwin_segmented_2500/                   # RoboTwin policy training data
    └── real_robot_900hr/                          # real-robot training data (coming soon)
```

Set in `.env`:

```bash
VIGAR_POLICY_CHECKPOINT=/path/to/vigar/weights/policy/robotwin_c2r
SUBGOAL_PLANNER_CHECKPOINT=/path/to/vigar/weights/subgoal_planner/robotwin_subgoal_planner
WAN_VAE_PATH=/path/to/vigar/weights/policy/robotwin_c2r/vae/Wan2.2_VAE.pth
QWEN_TOKENIZER_PATH=/path/to/vigar/weights/policy/robotwin_c2r/text_tokenizer
BASE_CHECKPOINT_PATH=/path/to/vigar/weights/Cosmos3-Nano-dcp
VIGAR_DATASET_MANIFEST=/path/to/vigar/data/robotwin_segmented_2500/dataset.json
VIGAR_GENERATED_GOAL_CACHE=/path/to/vigar/data/robotwin_segmented_2500/goal_cache
VIGAR_REAL_ROBOT_DATASET=/path/to/vigar/data/real_robot_900hr
```

| Item | Content |
| :--- | :--- |
| Policy | Model checkpoint of the world-action policy in ViGAR |
| Subgoal planner | Model checkpoint of the subgoal planner in ViGAR |
| Base checkpoint | Pretrained [nvidia/Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) in DCP format |
| RoboTwin data | 50 tasks × 50 Clean demonstrations, 3,691 generated goal images |
| Real-robot data | About 900 hours on AgiBot A2: 85,043 episodes with subtask annotations (coming soon) |
