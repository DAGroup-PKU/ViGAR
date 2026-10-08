"""Evaluate ViGAR tasks, optionally starting and managing a local policy server."""

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from recipes.simulation.robotwin.common.transport import PolicyClient
from recipes.simulation.robotwin.vigar.rollout import load_config


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def terminate(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def simulator_launch(config, gpu, path):
    """Preserve the simulator interpreter, graphics descriptors and environment."""
    workspace = Path(config["workspace"])
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        ROBOTWIN_WORKSPACE=str(workspace),
        PYTHONPATH=str(REPO),
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        PATH=f"{workspace}/.venv/bin:{os.environ.get('PATH', '')}",
    )
    for key, name in (("VK_ICD_FILENAMES", "nvidia_icd.json"), ("__EGL_VENDOR_LIBRARY_FILENAMES", "nvidia_egl.json")):
        if not environment.get(key) and (workspace / name).is_file():
            environment[key] = str(workspace / name)
    command = [
        environment.get("ROBOTWIN_PYTHON", "") or str(workspace / ".venv/bin/python"),
        "-m",
        "recipes.simulation.robotwin.vigar.rollout",
        "--config",
        str(path),
    ]
    return command, environment


def validate_health(health, config):
    if health.get("model") != "ViGAR" or (health.get("action_horizon"), health.get("action_dim")) != (48, 49):
        raise ValueError("Expected a ViGAR server with 48-step / 49-D actions")
    if config["action_type"] not in health.get("control_modes", []):
        raise ValueError("Checkpoint does not support the requested control mode")


def wait_for_server(server, config, output, timeout):
    deadline = time.monotonic() + timeout
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"Policy server exited; see {output / 'policy.log'}")
        try:
            with urllib.request.urlopen(config["server_url"] + "/health", timeout=2) as response:
                health = json.load(response)
            validate_health(health, config)
            if health.get("output") != str(output / "policy"):
                raise ValueError("A pre-existing server is listening on this port; choose an unused --server-url")
            return health
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Policy server did not become ready; see {output / 'policy.log'}") from None
            time.sleep(0.5)


def run_tasks(config, output, gpus, server=None):
    """Keep static task assignment and isolate each task in a simulator process."""
    tasks, results = config["tasks"], []
    completed = queue.Queue()
    stopped, lock, children = threading.Event(), threading.Lock(), set()
    started = time.monotonic()

    def write_summary():
        total = sum(row["evaluated"] for row in results)
        success = sum(row["successes"] for row in results)
        report = dict(
            tasks_requested=tasks,
            tasks_finished=len(results),
            evaluated=total,
            successes=success,
            success_rate=success / total if total else None,
            complete=len(results) == len(tasks) and all(row["complete"] for row in results),
            seconds=time.monotonic() - started,
            results=sorted(results, key=lambda row: tasks.index(row["task"])),
        )
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(output / "summary.json")
        return report

    def worker(gpu, selected):
        for task in selected:
            if stopped.is_set():
                return
            values = dict(config, tasks=[task], output=str(output / task))
            path = output / f"{task}.yaml"
            path.write_text(yaml.safe_dump(values))
            command, environment = simulator_launch(config, gpu, path)
            with (output / f"{task}.log").open("w") as log:
                with lock:
                    if stopped.is_set():
                        return
                    child = subprocess.Popen(
                        command,
                        cwd=REPO,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    children.add(child)
                try:
                    code = child.wait()
                finally:
                    with lock:
                        children.discard(child)
            summary = output / task / "summary.json"
            row = dict(task=task, exit_code=code)
            if summary.exists():
                row.update(json.loads(summary.read_text()))
                row["complete"] = row["complete"] and code == 0
            else:
                row.update(evaluated=0, successes=0, complete=False, errors=["No summary; inspect task log"])
            completed.put(row)

    pool = ThreadPoolExecutor(max_workers=len(gpus))
    try:
        futures = [pool.submit(worker, gpu, tasks[i :: len(gpus)]) for i, gpu in enumerate(gpus)]
        while any(not future.done() for future in futures) or not completed.empty():
            if server is not None and server.poll() is not None:
                raise RuntimeError(f"Policy server exited during evaluation; see {output / 'policy.log'}")
            for future in futures:
                if future.done():
                    future.result()
            try:
                row = completed.get(timeout=1)
            except queue.Empty:
                continue
            results.append(row)
            report = write_summary()
            print(
                json.dumps(dict(task=row["task"], evaluated=report["evaluated"], successes=report["successes"])),
                flush=True,
            )
        for future in futures:
            future.result()
        return write_summary()
    finally:
        with lock:
            stopped.set()
            active = list(children)
        for child in active:
            terminate(child)
        pool.shutdown(wait=True, cancel_futures=True)
        write_summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs/vigar.yaml")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for the aggregate and per-task results"
    )
    parser.add_argument("--checkpoint", type=Path, help="Start a local server; omit to use the server in --config")
    parser.add_argument("--server-url", help="Override the policy endpoint in --config")
    parser.add_argument(
        "--simulator-gpus", default="2", help="Comma-separated simulator slots; repeat GPU IDs to share"
    )
    local = parser.add_argument_group("Local server options (require --checkpoint)")
    local.add_argument("--recipe", type=Path, help="Optional training recipe")
    local.add_argument("--policy-gpus", help="GPUs sharding one server (default: 0,1)")
    local.add_argument("--weights", choices=["regular", "ema"], help="Checkpoint weight tree (default: ema)")
    local.add_argument(
        "--decode-video", action="store_true", help="Decode predicted video on a locally started server"
    )
    local.add_argument("--ready-timeout", type=int, help="Server startup timeout in seconds (default: 600)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and print commands without starting jobs"
    )
    args = parser.parse_args()
    if not args.checkpoint and (
        args.decode_video
        or any(value is not None for value in (args.recipe, args.policy_gpus, args.weights, args.ready_timeout))
    ):
        parser.error("Local server options require --checkpoint; configure an existing server when starting it")
    args.policy_gpus = "0,1" if args.policy_gpus is None else args.policy_gpus
    args.weights = args.weights or "ema"
    args.ready_timeout = 600 if args.ready_timeout is None else args.ready_timeout
    if args.ready_timeout <= 0:
        parser.error("--ready-timeout must be positive")
    output = args.output.resolve()
    config = load_config(args.config, output)
    if args.server_url:
        config["server_url"] = args.server_url.rstrip("/")
    gpus = [gpu.strip() for gpu in args.simulator_gpus.split(",")]
    policy_gpus = [gpu.strip() for gpu in args.policy_gpus.split(",")]
    if not all(gpus) or not all(policy_gpus):
        parser.error("GPU lists must contain nonempty IDs")
    server_command = None
    if args.checkpoint:
        endpoint = urllib.parse.urlparse(config["server_url"])
        if endpoint.hostname not in {"localhost", "127.0.0.1"} or endpoint.scheme != "http" or not endpoint.port:
            parser.error("Starting a local server requires server_url=http://127.0.0.1:PORT")
        server_command = [
            "bash",
            str(HERE / "serve.sh"),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--weights",
            args.weights,
            "--host",
            endpoint.hostname,
            "--port",
            str(endpoint.port),
            "--output",
            str(output / "policy"),
        ]
        if args.recipe:
            server_command += ["--recipe", str(args.recipe.resolve())]
        if args.decode_video:
            server_command += ["--decode-video"]
    plan = dict(
        policy_command=server_command,
        policy_gpus=policy_gpus if server_command else None,
        simulator_slots=[dict(gpu=gpu, tasks=config["tasks"][i :: len(gpus)]) for i, gpu in enumerate(gpus)],
        simulator_command=simulator_launch(config, gpus[0], output / f"{config['tasks'][0]}.yaml")[0],
        config=config,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    server = None

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        if server_command:
            environment = dict(
                os.environ, CUDA_VISIBLE_DEVICES=",".join(policy_gpus), NPROC_PER_NODE=str(len(policy_gpus))
            )
            with (output / "policy.log").open("w") as log:
                server = subprocess.Popen(
                    server_command,
                    cwd=REPO,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            health = wait_for_server(server, config, output, args.ready_timeout)
        else:
            health = PolicyClient(config["server_url"], config["request_timeout"]).call("/health")
            validate_health(health, config)
        (output / "policy.json").write_text(json.dumps(health, indent=2))
        report = run_tasks(config, output, gpus, server)
        if not report["complete"]:
            raise RuntimeError(f"Evaluation incomplete; see {output / 'summary.json'} and task logs")
        print(f"Evaluation complete: {output / 'summary.json'}")
    finally:
        terminate(server)
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
