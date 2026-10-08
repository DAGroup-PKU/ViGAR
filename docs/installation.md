# Installation

[← README](../README.md) · **Installation** · [Download](download.md) · [Evaluation](evaluation.md) · [Training](training.md)

## Requirements

- Ubuntu 22.04 / 24.04 (x86_64), NVIDIA driver supporting CUDA 12.8
- CUDA 12.8 toolkit, GCC 11 or later, Git, FFmpeg, Vulkan loader
- [uv](https://docs.astral.sh/uv/)

```bash
sudo apt-get install -y git build-essential ffmpeg libvulkan1 vulkan-tools
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Model environment

Python 3.13 and PyTorch 2.10, installed to `.venv-model`:

```bash
bash scripts/setup_models.sh
```

## Simulator environment

Python 3.10 and RoboTwin with its assets, installed to a separate workspace.
Use `--cuda-archs 9.0` for H100/H200 and `10.0` for B200.

```bash
bash scripts/setup_simulator.sh --workspace /path/to/robotwin-workspace \
  --cuda-home /usr/local/cuda-12.8 --cuda-archs 10.0
```

Add `--stage checkout|environment|assets|check` to run one stage at a time.

## Check

```bash
.venv-model/bin/python scripts/check_environment.py --component models
/path/to/robotwin-workspace/.venv/bin/python scripts/check_environment.py --component simulator
```

## Configure

```bash
cp .env.example .env
```

Set these paths in `.env`, then add the [download paths](download.md):

```bash
PYTHON_BIN=/path/to/ViGAR/.venv-model/bin/python
NATIVE_SIM_PYTHON=/path/to/robotwin-workspace/.venv/bin/python
ROBOTWIN_ROOT=/path/to/robotwin-workspace/RoboTwin
VIGAR_WORKSPACE=/path/to/vigar-workspace
```

Load it before running evaluation or training:

```bash
source scripts/env.sh
```
