"""Preserve scene resets and initialize upstream success-check bookkeeping."""

import importlib


def initialize_reset_hooks(environment, task):
    if task not in {"dump_bin_bigbin", "put_bottles_dustbin"}:
        return
    original = environment.get_cluttered_table

    def get_cluttered_table(cluttered_numbers=10, xlim=None, ylim=None, zlim=None):
        # Upstream adds table_xy_bias in place to mutable default lists. Fresh
        # bounds preserve the intended first-reset scene across expert retries
        # and policy resets, without accumulating the table offset.
        return original(
            cluttered_numbers=cluttered_numbers,
            xlim=list((-0.59, 0.59) if xlim is None else xlim),
            ylim=list((-0.34, 0.34) if ylim is None else ylim),
            zlim=list((0.741,) if zlim is None else zlim),
        )

    environment.get_cluttered_table = get_cluttered_table


def initialize_task_state(environment, task):
    # The upstream evaluator reuses the expert environment object across resets.
    # Fresh policy environments need these same fields, derived before any
    # motion from the initial scene. They are never sent to the policy model.
    if task not in {"open_laptop", "place_object_scale", "put_object_cabinet"}:
        return
    module = importlib.import_module(f"envs.{task}")
    if task == "open_laptop":
        face = module.get_face_prod(environment.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        environment.arm_tag = module.ArmTag("left" if face > 0 else "right")
    else:
        pose = environment.object.get_pose()
        environment.arm_tag = module.ArmTag("right" if pose.p[0] > 0 else "left")
        if task == "put_object_cabinet":
            environment.origin_z = pose.p[2]
