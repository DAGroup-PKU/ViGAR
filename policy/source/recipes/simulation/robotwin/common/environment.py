"""Shared RoboTwin task setup and fresh/retained scene lifecycle."""

import copy
import importlib
import random

import yaml

from .task_state import initialize_reset_hooks, initialize_task_state


class UnstableScene(RuntimeError):
    """Expected seed rejection reported by RoboTwin's stability check."""


def task_arguments(root, task, task_config):
    args = yaml.safe_load((root / "task_config" / f"{task_config}.yml").read_text())
    if args["embodiment"] != ["aloha-agilex"]:
        raise ValueError("The current 49D simulator bridge is for aloha-agilex")
    robots = yaml.safe_load((root / "task_config/_embodiment_config.yml").read_text())
    robot_path = (root / robots["aloha-agilex"]["file_path"]).resolve()
    robot = yaml.safe_load((robot_path / "config.yml").read_text())
    camera_configs = yaml.safe_load((root / "task_config/_camera_config.yml").read_text())
    head = camera_configs[args["camera"]["head_camera_type"]]
    args.update(
        task_name=task,
        task_config=task_config,
        ckpt_setting="goalwam",
        left_robot_file=str(robot_path),
        right_robot_file=str(robot_path),
        dual_arm_embodied=True,
        left_embodiment_config=robot,
        right_embodiment_config=copy.deepcopy(robot),
        head_camera_h=head["h"],
        head_camera_w=head["w"],
        eval_mode=True,
        eval_video_log=False,
        eval_video_save_dir=None,
        render_freq=0,
        save_data=False,
        collect_data=False,
    )
    args["camera"].update(collect_head_camera=True, collect_wrist_camera=True)
    args["data_type"].update(rgb=True, endpose=True, qpos=True, pointcloud=False, depth=False)
    return args


def make_environment(task, args, seed):
    from envs.utils.create_actor import UnStableError

    # Upstream initializes NumPy/Torch; explicitly seed Python as well so the
    # expert and policy resets reproduce randomized scene construction.
    random.seed(seed)
    env = getattr(importlib.import_module(f"envs.{task}"), task)()
    initialize_reset_hooks(env, task)
    try:
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **copy.deepcopy(args))
        initialize_task_state(env, task)
    except BaseException as error:
        try:
            env.close_env(clear_cache=True)
        except Exception:
            pass
        if isinstance(error, UnStableError):
            raise UnstableScene(str(error)) from error
        raise
    return env


class RetainedEnvironmentFactory:
    """Reuse upstream's robot/planner objects across scene resets, as its evaluator does."""

    def __init__(self):
        self.environment = None

    def make(self, task, args, seed):
        from envs.utils.create_actor import UnStableError

        if self.environment is None:
            self.environment = getattr(importlib.import_module(f"envs.{task}"), task)()
            initialize_reset_hooks(self.environment, task)
        random.seed(seed)
        try:
            self.environment.setup_demo(now_ep_num=0, seed=seed, is_test=True, **copy.deepcopy(args))
            initialize_task_state(self.environment, task)
        except BaseException as error:
            self.environment.close_env(clear_cache=True)
            if isinstance(error, UnStableError):
                raise UnstableScene(str(error)) from error
            raise
        return self.environment
