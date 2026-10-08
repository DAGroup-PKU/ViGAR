import argparse
import importlib
import importlib.metadata
import importlib.util
import os
import subprocess
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_models(cpu_only):
    required = {'torch': '2.10.0', 'torchvision': '0.25.0', 'torchcodec': '0.10.0', 'transformers': '4.57.6'}
    for name, expected in required.items():
        actual = importlib.metadata.version(name)
        if actual.split('+')[0] != expected:
            raise RuntimeError(f'{name}: expected {expected}, found {actual}')
    import torch
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected CUDA 12.8, found {torch.version.cuda}")
    import torchcodec
    from PIL import Image
    import av
    import pyarrow
    for path in ['vigar/source', 'vigar/source/third_party/cosmos_runtime', 'subgoal_planner/inference_runtime']:
        target = ROOT / path
        module = ('recipes.ViGAR.data.dataset' if path == 'vigar/source' else
                  'cosmos_framework.utils.flags' if cpu_only else
                  'cosmos_framework.configs.toml_config.sft_config' if path.startswith('subgoal_planner') else
                  'cosmos_framework.inference.common.init')
        code = 'import sys;sys.path.insert(0,sys.argv[1]);__import__(sys.argv[2]);print(sys.argv[2])'
        subprocess.run([sys.executable, '-c', code, str(target), module], check=True)
    if not cpu_only:
        import transformer_engine.pytorch
        import flash_attn
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; check the NVIDIA driver and GPU allocation')
        from flash_attn import flash_attn_func
        query = torch.randn(1, 16, 2, 64, device='cuda', dtype=torch.bfloat16)
        result = flash_attn_func(query, query, query)
        if not torch.isfinite(result).all():
            raise RuntimeError('FlashAttention returned nonfinite values')
        torch.cuda.synchronize()
        print(torch.cuda.get_device_name())
    print('Model imports ready' if cpu_only else 'Model environment ready')


def check_simulator(cpu_only):
    for name in ['sapien', 'mplib', 'torch', 'pytorch3d', 'cv2', 'h5py', 'imageio']:
        importlib.import_module(name)
    if not cpu_only:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        import sapien
        setup_path = ROOT / 'vigar/source/recipes/simulation/robotwin/common/setup.py'
        spec = importlib.util.spec_from_file_location('simulator_setup', setup_path)
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        os.environ.update(setup.runtime_environment(Path(sys.prefix).parent))
        engine = sapien.Engine()
        engine.set_renderer(sapien.SapienRenderer())
        scene = engine.create_scene()
        camera = scene.add_camera('check', 32, 32, 1.0, 0.1, 10)
        scene.step()
        scene.update_render()
        camera.take_picture()
        import numpy as np
        if not np.isfinite(camera.get_picture("Color")).all():
            raise RuntimeError("Renderer returned nonfinite pixels")
    print('Simulator imports ready' if cpu_only else 'Simulator environment ready')


if __name__ == '__main__':
    libraries = [str(path) for folder in site.getsitepackages()
                 for path in (Path(folder) / 'nvidia').glob('*/lib') if path.is_dir()]
    existing = os.environ.get('LD_LIBRARY_PATH', '').split(':')
    if any(path not in existing for path in libraries):
        os.environ['LD_LIBRARY_PATH'] = ':'.join(libraries + [path for path in existing if path])
        os.execv(sys.executable, [sys.executable, *sys.argv])
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', choices=['models', 'simulator'], required=True)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    (check_models if args.component == 'models' else check_simulator)(args.cpu_only)
