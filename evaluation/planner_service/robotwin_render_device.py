"""Pin SAPIEN 3's render system to the simulator's visible CUDA device.

SapienRenderer(**kwargs) in the installed 3.0.0b1 compatibility wrapper ignores
its kwargs; selecting the device there would silently leave all shards on GPU0.
"""

import os


def configure():
    import sapien
    import sapien.core as core

    def create_scene(config=None):
        if config is None:
            config = core.SceneConfig()
        sapien.physx.set_scene_config(config)
        render = sapien.render.RenderSystem(os.environ.get("VIGAR_RENDER_DEVICE", "cuda:0"))
        return sapien.Scene([sapien.physx.PhysxCpuSystem(), render])

    core.Engine.create_scene = staticmethod(create_scene)
