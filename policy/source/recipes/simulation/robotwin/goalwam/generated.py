"""Recorded initial scenes with generated goals; never load expert terminal RGB."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..common.setup import REVISION


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_generated_config(config):
    root = config.get("generated_goal_root")
    if root is None:
        return
    if not isinstance(root, str) or not root:
        raise ValueError("generated_goal_root must be a nonempty directory path")
    if config.get("oracle_reference_roots") is not None:
        raise ValueError("Generated goals and oracle_reference_roots are mutually exclusive")
    timeout = config.get("generated_goal_timeout", 7200)
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("generated_goal_timeout must be finite and positive")


def split_goal_canvas(canvas, img_size, boxes):
    from recipes.GoalWAM.data.images import camera_layout

    shape, expected = camera_layout(img_size, ["head", "left", "right"])
    if boxes != expected or canvas.shape != (*shape, 3) or canvas.dtype != np.uint8:
        raise ValueError("Generated goal camera layout does not match the policy canvas")
    restored = np.zeros_like(canvas)
    views = {}
    for name, (y, x, h, w) in boxes.items():
        views[name] = canvas[y : y + h, x : x + w].copy()
        restored[y : y + h, x : x + w] = views[name]
    if not np.array_equal(canvas, restored):
        raise ValueError("Generated goal must have black outer canvas padding")
    return views


def load_generated_references(config, task):
    validate_generated_config(config)
    root = Path(config["generated_goal_root"]).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest["condition"] != "initial_plus_robot_appearance":
        raise ValueError("Expected generation with robot reference")
    cases = [case for case in manifest["cases"] if case["task"] == task]
    if len(cases) != config["episodes"] or len({c["seed"] for c in cases}) != len(cases):
        raise ValueError(f"Generated manifest must contain exactly {config['episodes']} unique seeds for {task}")
    references = []
    for case in cases:
        source = Path(case["source_summary"])
        contract = read_json(source.parent / "eval_contract.json")
        if contract.get("revision") != REVISION:
            raise ValueError("Generated reference uses a different simulator revision")
        for key in ("task_config", "instruction_type", "start_seed", "episodes"):
            if contract.get(key) != config[key]:
                raise ValueError(f"Generated reference disagrees with {key}")
        if Path(contract["robotwin_root"]).resolve() != Path(config["robotwin_root"]).resolve():
            raise ValueError("Generated reference uses a different simulator installation")
        rows = [
            r
            for r in read_json(source)["records"]
            if r["status"] == "evaluated" and r["seed"] == case["seed"] and r["task"] == task
        ]
        if len(rows) != 1 or rows[0]["instruction"] != case["instruction"]:
            raise ValueError("Generated reference task/seed/instruction mismatch")
        episode = source.parent / task / f"seed_{case['seed']:09d}"
        if episode.resolve() != Path(case["reference_episode"]).resolve():
            raise ValueError("Generated reference initial episode mismatch")
        with np.load(episode / "initial_state.npz", allow_pickle=False) as values:
            initial = {key: values[key].copy() for key in ("state", "cam2world_gl")}
        initial["images"] = {}
        for camera in ("head", "left", "right"):
            with Image.open(episode / f"initial_{camera}.png") as image:
                initial["images"][camera] = np.array(image.convert("RGB"))
        directory = (root / case["directory"]).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Generated case directory escapes experiment")
        references.append(
            dict(case, initial=initial, directory=str(directory), img_size=manifest["config"]["img_size"])
        )
    return references


def load_generated_goal(reference, timeout):
    """Wait for an atomic cached API result, failing on errors without GT fallback."""
    directory = Path(reference["directory"])
    deadline = time.monotonic() + timeout
    while True:
        result_path = directory / "result.json"
        result = read_json(result_path) if result_path.exists() else {}
        if result.get("status") == "complete":
            break
        if result.get("status") == "error":
            raise RuntimeError(f"Goal generation failed: {directory}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Generated goal not ready: {directory}")
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    request = read_json(directory / "request.json")
    if (
        digest(json.dumps(request, sort_keys=True).encode() + (directory / "valid_pixels.png").read_bytes())
        != result["request_sha256"]
    ):
        raise ValueError("Generated goal request fingerprint mismatch")
    if digest((directory / "initial.png").read_bytes()) != request["initial_sha256"]:
        raise ValueError("Generated goal initial image changed")
    refs = request.get("references", [])
    if len(refs) != 1 or refs[0]["file"] != "robot_reference.png":
        raise ValueError("Generated goal must use the robot appearance reference")
    if digest((directory / "robot_reference.png").read_bytes()) != refs[0]["sha256"]:
        raise ValueError("Generated goal robot reference changed")
    path = directory / "generated_goal.png"
    if digest(path.read_bytes()) != result["output_sha256"][path.name]:
        raise ValueError("Generated goal PNG checksum mismatch")
    with Image.open(path) as image:
        canvas = np.array(image.convert("RGB"))
    goal = split_goal_canvas(canvas, reference["img_size"], reference["camera_boxes"])
    if request["initial_sha256"] != reference["initial_sha256"]:
        raise ValueError("Generated API initial image does not match the prepared case")
    for camera, expected in reference["raw_initial_sha256"].items():
        if digest((Path(reference["reference_episode"]) / f"initial_{camera}.png").read_bytes()) != expected:
            raise ValueError("Reference raw initial image changed after preparation")
    provenance = dict(
        goal_source="initial_plus_robot_appearance",
        generated_goal_reference=str(path),
        generated_goal_png_sha256=result["output_sha256"][path.name],
        goal_canvas_sha256=digest(canvas.tobytes()),
        request_sha256=result["request_sha256"],
        image_generation_request_id=result.get("request_id"),
    )
    return goal, canvas, provenance
