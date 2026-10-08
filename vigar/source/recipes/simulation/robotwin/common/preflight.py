"""Check simulator imports, assets, CUDA and an actual SAPIEN RGB render."""

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(passed=False, python=sys.version, root=str(args.root), packages={})
    try:
        for name in ("torch", "torchvision", "sapien", "mplib", "open3d", "pytorch3d", "curobo", "toppra"):
            importlib.import_module(name)
            result["packages"][name] = importlib.metadata.version("nvidia-curobo" if name == "curobo" else name)
        import numpy as np
        import sapien.core as sapien
        import torch

        assert torch.cuda.is_available(), "CUDA is unavailable"
        result["gpu"] = torch.cuda.get_device_name()
        for name, minimum in [("background_texture", 10000), ("embodiments", 200), ("objects", 9000)]:
            count = sum(p.is_file() for p in (args.root / "assets" / name).rglob("*"))
            assert count >= minimum, f"Incomplete {name}: {count} files"
        # Match RoboTwin's ray-tracing/OIDN renderer, including camera readback.
        engine = sapien.Engine()
        renderer = sapien.SapienRenderer()
        engine.set_renderer(renderer)
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(4)
        sapien.render.set_ray_tracing_path_depth(4)
        sapien.render.set_ray_tracing_denoiser("oidn")
        scene = engine.create_scene()
        scene.add_ground(0)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [1, 1, 1])
        camera = scene.add_camera("test", 64, 48, 1.0, 0.1, 10)
        camera.set_pose(sapien.Pose([-1, 0, 1], [0.9238795, 0, 0.3826834, 0]))
        scene.update_render()
        camera.take_picture()
        rgba = camera.get_picture("Color")
        assert rgba.shape == (48, 64, 4) and np.isfinite(rgba).all()
        result.update(passed=True, rendered_shape=list(rgba.shape))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
