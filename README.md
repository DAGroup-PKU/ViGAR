<div align="center">

# Rethinking World-Action Model for Compositional and In-Context Robotic Manipulation

ViGAR: **Vi**sual **G**oal-conditioned **A**ction **R**easoning

Shukai Gong<sup>1*</sup> · Xuanran Zhai<sup>2*</sup> · Yintianrun Zhang<sup>1*</sup> · Ruopeng Cui<sup>2</sup> · Ye Huang<sup>1</sup> · Yiyang Fu<sup>1</sup> · Dexuan Lyu<sup>2</sup><br>
Chaojie Li<sup>2</sup> · Xinyi Song<sup>2</sup> · Peiwen Lin<sup>2</sup> · Chuang Wang<sup>2</sup> · Mingyuan Jia<sup>3</sup> · Yufan Deng<sup>1</sup><br>
Jiaxin Fang<sup>3</sup> · Bo Liang<sup>1</sup> · Jiaxin Li<sup>1</sup> · Yuxiang Gao<sup>3†</sup> · Hao Liu<sup>2†</sup> · Daquan Zhou<sup>1†</sup>

<sup>1</sup> Peking University &nbsp; <sup>2</sup> AgiBot &nbsp; <sup>3</sup> CocoMatrix

<sup>*</sup> Equal contribution &nbsp; <sup>†</sup> Corresponding author

[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f?logo=googlechrome&logoColor=white)](https://dagroup-pku.github.io/ViGAR/)
[![arXiv](https://img.shields.io/badge/arXiv-2610.02368-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.02368)
[![HF Daily Paper](https://img.shields.io/badge/🤗%20Hugging%20Face-Daily%20Paper-yellow)](https://huggingface.co/papers/2610.02368)
[![Hugging Face Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models%20%26%20Data-yellow)](https://huggingface.co/DAGroup-PKU/ViGAR)
[![Python](https://img.shields.io/badge/Python-3.13%20%7C%203.10-3776AB?logo=python&logoColor=white)](docs/installation.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)](docs/installation.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

![ViGAR teaser: task-diverse and long-horizon data train a subgoal planner and a world-action policy, enabling long-horizon manipulation by subtask decomposition and in-context learning on unseen tasks](assets/teaser.jpg)

## 📑 Todo List

- [x] Evaluation code and checkpoints on **RoboTwin**.
- [x] Training code and data on **RoboTwin**.
- [ ] 900-hour real-robot dataset with subtask annotations.

## 🎬 Demo

https://github.com/user-attachments/assets/deeeb726-079d-49eb-a710-f549fe473520

## ⚙️ Getting Started

```bash
git clone https://github.com/DAGroup-PKU/ViGAR.git
cd ViGAR
```

| Step | Guide |
| :--- | :--- |
| Set up the environments | [Installation](docs/installation.md) |
| Download weights and data | [Download](docs/download.md) |
| Evaluate on Simulation Benchmark | [Evaluation](docs/evaluation.md) |
| Train the policies and subgoal planners in ViGAR | [Training](docs/training.md) |

## 💪 Repository

```text
vigar/            Policy model, training and serving
subgoal_planner/  Planner runtime and training implementation
simulation/       RoboTwin tasks and simulator setup
data/             Annotation, goal generation and action conversion
evaluation/       Policy/planner services and closed-loop evaluation
scripts/          Installation and launch commands
configs/          Task and inference settings
docs/             Installation, download, evaluation and training guides
```

The policy uses VeOmni with a bundled Cosmos runtime. Planner processes select
their own Cosmos runtime through `PYTHONPATH`; they run separately from policy
and simulator processes.

## 🙏 Acknowledgements

Built on [NVIDIA Cosmos](https://github.com/NVIDIA/cosmos),
[VeOmni](https://github.com/ByteDance-Seed/VeOmni),
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin),
[Wan2.2](https://github.com/Wan-Video/Wan2.2) and
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL).

## ✏️ Citation

```bibtex
@article{gong2026rethinking,
  title={Rethinking World-Action Model for Compositional and In-Context Robotic Manipulation},
  author={Gong, Shukai and Zhai, Xuanran and Zhang, Yintianrun and Cui, Ruopeng and Huang, Ye and Fu, Yiyang and Lyu, Dexuan and Li, Chaojie and Song, Xinyi and Lin, Peiwen and others},
  journal={arXiv preprint arXiv:2610.02368},
  year={2026}
}
```

## License

Original project code uses the [MIT license](LICENSE). Bundled components retain their upstream licenses. Cosmos-derived model materials use OpenMDW-1.1.