from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from recipes.simulation.robotwin.common.control import smooth_joint_commands, validate_joint_smoothing
from recipes.simulation.robotwin.common.environment import UnstableScene, make_environment, task_arguments
from native_expert_guard import run_expert_rollout
from recipes.simulation.robotwin.common.geometry import controller_actions, encode_observation, extract_images
from recipes.simulation.robotwin.common.rendering import head_video_frame
from recipes.simulation.robotwin.common.setup import REVISION, git_head
from recipes.simulation.robotwin.common.transport import PolicyClient
from recipes.simulation.robotwin.vigar.generated import load_generated_goal, load_generated_references, validate_generated_config
from recipes.simulation.robotwin.vigar.oracle import load_oracle_references, validate_oracle_config
from recipes.simulation.robotwin.vigar.sampling import resolve_sampling


def load_config(path, output=None):
    config = yaml.safe_load(Path(path).read_text())
    if not config.get("async_planner") or not config.get("planner_port"):
        raise ValueError("Explicit async_planner and planner_port required")
    if config.get("policy_obs_order") != "bgr":
        raise ValueError("The policy expects BGR observations")
    if config.get("oracle_reference_roots") or config.get("generated_goal_root"):
        raise ValueError("Async evaluation cannot use expert/cached goal references")
    if output:
        config["output"] = str(Path(output).resolve())
    for key in ("episodes", "max_seed_attempts", "request_timeout"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if not 1 <= config["replan_steps"] <= 48:
        raise ValueError("replan_steps must be in [1,48]")
    if config["action_type"] not in {"ee", "qpos"}:
        raise ValueError("action_type must be ee or qpos")
    validate_joint_smoothing(config.get("joint_smoothing_window", 1), config["action_type"])
    resolve_sampling(config.get("sampling"))
    validate_oracle_config(config)
    validate_generated_config(config)
    if config.get("generated_goal_root") is not None:
        config["generated_goal_root"] = str(Path(config["generated_goal_root"]).resolve())
    if config.get("oracle_reference_roots") is not None:
        config["oracle_reference_roots"] = [str(Path(p).resolve()) for p in config["oracle_reference_roots"]]
    if config["max_policy_steps"] is not None and config["max_policy_steps"] < 1:
        raise ValueError("max_policy_steps must be positive or null")
    if not config["tasks"] or len(set(config["tasks"])) != len(config["tasks"]):
        raise ValueError("Select a nonempty, unique task list")
    for task in config["tasks"]:
        if not isinstance(task, str) or not task.isidentifier():
            raise ValueError(f"Invalid task name {task!r}")
    root = Path(config["robotwin_root"]).resolve()
    if git_head(root) != REVISION:
        raise ValueError(f"RoboTwin must be pinned to {REVISION}")
    missing = [task for task in config["tasks"] if not (root / "envs" / f"{task}.py").is_file()]
    if missing:
        raise ValueError(f"Requested tasks absent from pinned simulator: {missing}")
    return config


def write_summary(output, config, records, errors):
    tasks = {}
    for task in config["tasks"]:
        rows = [r for r in records if r["task"] == task]
        finished = [r for r in rows if r["status"] == "evaluated"]
        success = sum(r["success"] for r in finished)
        tasks[task] = dict(
            evaluated=len(finished),
            successes=success,
            success_rate=success / len(finished) if finished else None,
            rejected_expert_seeds=sum(r["status"] == "expert_rejected" for r in rows),
        )
    total = sum(t["evaluated"] for t in tasks.values())
    successful = sum(t["successes"] for t in tasks.values())
    rates = [t["success_rate"] for t in tasks.values() if t["success_rate"] is not None]
    summary = dict(
        protocol=("strict sync current-obs subgoal planner / no expert goal / no oracle switch" if config.get("goal_refresh_mode") == "sync_current" else "realtime async subgoal planner / no expert goal / no oracle switch" if config.get("async_planner") else
            "same-seed generated goal with robot reference / current-only / 48-step chunks"
            if config.get("generated_goal_root")
            else "same-seed expert terminal RGB / current-only / 48-step chunks"
        ),
        tasks=tasks,
        evaluated=total,
        successes=successful,
        micro_success_rate=successful / total if total else None,
        macro_success_rate=float(np.mean(rates)) if rates else None,
        errors=errors,
        records=records,
        complete=not errors and all(t["evaluated"] == config["episodes"] for t in tasks.values()),
    )
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "summary.json")
    return summary


def evaluate(config, client):
    root, output = Path(config["robotwin_root"]).resolve(), Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "eval_contract.json"
    contract = {key:value for key,value in dict(config, revision=REVISION).items()
                if key not in {"server_url", "planner_port"}}
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Evaluation settings changed; choose a new output directory")
    temporary_contract = contract_path.with_suffix(".tmp")
    temporary_contract.write_text(json.dumps(contract, indent=2) + "\n")
    temporary_contract.replace(contract_path)
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "description/utils"))
    from generate_episode_instructions import generate_episode_descriptions
    from sim_acceleration import install, AsyncWriter
    # Optional simulator acceleration; the defaults run the unmodified simulator.
    acceleration=config.get('sim_acceleration') or {}
    install(defer_render=acceleration.get('defer_render',False),cache_every=acceleration.get('cache_every',1))
    writer=AsyncWriter() if acceleration.get('async_io',False) else None

    previous = output / "summary.json"
    records, errors = [], []
    if previous.exists():
        old = json.loads(previous.read_text())
        (output / ("summary.before_resume." + str(time.time_ns()) + ".json")).write_text(previous.read_text())
        records = [x for x in old["records"] if x["status"] in ("evaluated", "expert_rejected")]
    try:
        for task in config["tasks"]:
            args = task_arguments(root, task, config["task_config"])
            async_enabled = bool(config["async_planner"])
            generated = bool(config.get("generated_goal_root"))
            references = None if async_enabled else (load_generated_references(config, task) if generated else load_oracle_references(config, task))
            completed = sum(x["task"] == task and x["status"] == "evaluated" for x in records)
            if completed == config["episodes"]:
                continue
            seen = {x["seed"] for x in records if x["task"] == task}
            paired = None
            if config.get("paired_episode_file"):
                paired = json.loads(Path(config["paired_episode_file"]).read_text())[config["paired_split"] + "/" + task]
                assert len(paired) == config["episodes"]
            for attempt in range(len(paired) if paired is not None else (len(references) if references is not None else config["max_seed_attempts"])):
                reference = references[attempt] if references is not None else None
                seed = paired[attempt]["seed"] if paired is not None else (reference["seed"] if reference else config["start_seed"] + attempt)
                if seed in seen:
                    continue
                row = dict(task=task, seed=seed, status="preparing")
                records.append(row)
                provenance, goal_canvas = {}, None
                if generated:
                    initial, instruction = reference["initial"], reference["instruction"]
                    goal, goal_canvas, provenance = load_generated_goal(
                        reference, config.get("generated_goal_timeout", 7200)
                    )
                    row.update(provenance)
                elif reference:
                    initial, goal, instruction = reference["initial"], reference["goal"], reference["instruction"]
                    row.update(
                        oracle_reference=reference["source"],
                        oracle_goal_sha256=reference["goal_sha256"],
                    )
                else:
                    try:
                        expert = make_environment(task, args, seed)
                    except UnstableScene as error:
                        row.update(status="expert_rejected", reason=str(error))
                        write_summary(output, config, records, errors)
                        continue
                    try:
                        initial = copy.deepcopy(encode_observation(expert.get_obs()))
                        episode_info = run_expert_rollout(expert, row)
                        accepted = bool(row["expert_plan_success"] and row["expert_goal_reached"])
                        goal = {} if async_enabled else (copy.deepcopy(extract_images(expert.get_obs())) if accepted else None)
                    finally:
                        expert.close_env(clear_cache=True)
                        del expert
                        gc.collect()
                    if not accepted and paired is not None:
                        raise RuntimeError("Requested expert seed failed: " + str(seed))
                    if not accepted:
                        row["status"] = "expert_rejected"
                        row["reason"] = "planner_failure" if not row["expert_plan_success"] else "goal_not_reached"
                        write_summary(output, config, records, errors)
                        continue
                    random.seed(seed + 7919)
                    np.random.seed(seed + 7919)
                    descriptions = generate_episode_descriptions(task, [episode_info["info"]], config.get("instruction_count", config["episodes"]))
                    instruction = random.choice(descriptions[0][config["instruction_type"]])
                    if paired is not None: instruction = paired[attempt]["instruction"]
                if async_enabled:
                    client.reset(task, seed)
                environment = make_environment(task, args, seed)
                video = None
                try:
                    current = encode_observation(environment.get_obs())
                    np.testing.assert_allclose(current["state"], initial["state"], rtol=0, atol=1e-5)
                    np.testing.assert_allclose(current["cam2world_gl"], initial["cam2world_gl"], rtol=0, atol=1e-6)
                    for camera in initial["images"]:
                        np.testing.assert_array_equal(current["images"][camera], initial["images"][camera])
                    environment.set_instruction(instruction=instruction)
                    directory = output / task / f"seed_{seed:09d}"
                    if directory.exists():
                        directory.rename(directory.with_name(directory.name + ".failed." + str(time.time_ns())))
                    directory.mkdir(parents=True)
                    import imageio.v2 as imageio

                    if generated:
                        imageio.imwrite(directory / "goal_canvas.png", goal_canvas)
                        (directory / "goal_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
                    for camera in goal:
                        imageio.imwrite(directory / f"goal_{camera}.png", goal[camera])
                        imageio.imwrite(directory / f"initial_{camera}.png", current["images"][camera])
                    np.savez_compressed(
                        directory / "initial_state.npz", state=current["state"], cam2world_gl=current["cam2world_gl"]
                    )
                    if config["save_video"]:
                        video = imageio.get_writer(directory / "rollout.mp4", fps=10, macro_block_size=1)
                    limit = environment.step_lim
                    if config["max_policy_steps"] is not None:
                        limit = min(limit, config["max_policy_steps"])
                    queries, latency = 0, []
                    commands = []
                    while environment.take_action_cnt < limit and not environment.eval_success:
                        request = encode_observation(environment.get_obs())
                        request.update(
                            instruction=instruction,
                            **({} if async_enabled else dict(goal_images=goal)),
                            generation_seed=config["generation_seed"] + queries,
                            artifact_context=dict(
                                **provenance,
                                task=task,
                                episode_seed=seed,
                                query_index=queries,
                                control_step=environment.take_action_cnt,
                            ),
                        )
                        if generated:
                            request["goal_canvas_sha256"] = provenance["goal_canvas_sha256"]
                        if "sampling" in config:
                            request["sampling"] = config["sampling"]
                        start = time.monotonic()
                        result = client.call("/infer", request)
                        latency.append(time.monotonic() - start)
                        # Preserve the inference-time camera transform for the
                        # WHOLE chunk; do not re-anchor deltas after each step.
                        chunk = controller_actions(
                            result["absolute"], result["valid"], request["cam2world_gl"], config["action_type"]
                        )
                        if config.get("joint_smoothing_window", 1) > 1:
                            chunk = smooth_joint_commands(chunk, config["joint_smoothing_window"])
                        (writer.savez if writer else np.savez_compressed)(
                            directory / f"chunk_{queries:04d}.npz",
                            **result,
                            controller_commands=chunk,
                            cam2world_gl=request["cam2world_gl"],
                        )
                        queries += 1
                        for command in chunk[: config["replan_steps"]]:
                            if environment.take_action_cnt >= limit or environment.eval_success:
                                break
                            if video:
                                video.append_data(head_video_frame(environment))
                            else:
                                # Keep the same crazy-light RNG updates whether
                                # video recording is enabled or disabled.
                                environment._update_render()
                            environment.take_action(command, action_type=config["action_type"])
                            commands.append(command)
                    final = encode_observation(environment.get_obs())
                    if video:
                        video.append_data(final["images"]["head"])
                    for camera, rgb in final["images"].items():
                        imageio.imwrite(directory / f"final_{camera}.png", rgb)
                    np.savez_compressed(
                        directory / "final_state.npz", state=final["state"], cam2world_gl=final["cam2world_gl"]
                    )
                    np.save(directory / "executed_commands.npy", np.asarray(commands))
                    row.update(
                        status="evaluated",
                        success=bool(environment.eval_success),
                        instruction=instruction,
                        control_steps=environment.take_action_cnt,
                        queries=queries,
                        inference_seconds=latency,
                        action_type=config["action_type"],
                        step_limit=limit,
                    )
                finally:
                    if video:
                        video.close()
                    environment.close_env(clear_cache=True)
                    del environment
                    gc.collect()
                if writer:
                    writer.close();writer=AsyncWriter()
                completed += 1
                write_summary(output, config, records, errors)
                if completed == config["episodes"]:
                    break
            if completed != config["episodes"]:
                raise RuntimeError(f"{task}: exhausted expert seed attempts ({completed}/{config['episodes']})")
    except BaseException as error:
        errors.append(f"{type(error).__name__}: {error}")
        if records and records[-1]["status"] == "preparing":
            records[-1]["status"] = "runtime_error"
        raise
    finally:
        if writer:writer.close()
        write_summary(output, config, records, errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/vigar.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.output)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    from native_async_goal import NativeAsyncGoalClient
    from async_subgoal import AsyncSubgoalProducer
    from robotwin_wire import Client as PlannerClient
    from robotwin_async_gateway import concat_views
    from robotwin_render_device import configure
    configure()
    planner = PlannerClient(config["planner_port"])
    from sync_subgoal import SyncSubgoalProducer
    refresh = config.get("goal_refresh_mode", "async_latest")
    if refresh not in ("async_latest", "sync_current"): raise ValueError("Invalid refresh mode")
    producer = (SyncSubgoalProducer(planner) if refresh == "sync_current" else AsyncSubgoalProducer(planner))
    client = NativeAsyncGoalClient(PolicyClient(config["server_url"], config["request_timeout"]),
        producer, concat_views, Path(config["output"] + ".async_goals.jsonl"),
        policy_obs_order=config["policy_obs_order"], goal_refresh_mode=refresh)
    health = client.call("/health")
    if (health.get("action_horizon"), health.get("action_dim")) != (48, 49):
        raise ValueError("Server does not implement 48-step / 49-D actions")
    if config["action_type"] not in health.get("control_modes", []):
        raise ValueError("Selected control mode is not supervised by this checkpoint")
    try:
        evaluate(config, client)
    finally:
        producer.stop()
        planner.close()


if __name__ == "__main__":
    main()
