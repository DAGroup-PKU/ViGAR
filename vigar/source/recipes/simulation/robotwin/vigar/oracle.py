"""Reuse completed expert goals for comparisons with identical conditioning."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..common.setup import REVISION


def validate_oracle_config(config):
    roots = config.get("oracle_reference_roots")
    if roots is not None:
        if not isinstance(roots, list) or not roots or any(not isinstance(p, str) or not p for p in roots):
            raise ValueError("oracle_reference_roots must be a nonempty list of output directories")
        if len(set(roots)) != len(roots):
            raise ValueError("oracle_reference_roots must be distinct")
    timeout = config.get("oracle_reference_timeout", 7200)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("oracle_reference_timeout must be finite and positive")


def load_oracle_references(config, task):
    validate_oracle_config(config)
    roots = config.get("oracle_reference_roots")
    if roots is None:
        return None
    started = time.monotonic()
    while True:
        found = []
        for root in map(Path, roots):
            path = root / task / "summary.json"
            if not path.exists():
                path = root / "summary.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text())
            if "records" not in summary or task not in summary.get("tasks", {}):
                continue
            if not summary["complete"]:
                if summary["errors"]:
                    raise RuntimeError(f"Oracle reference failed: {path}: {summary['errors']}")
                continue
            found.append((path, summary))
        if len(found) > 1:
            raise ValueError(f"Multiple oracle references for {task}")
        if found:
            break
        remaining = config.get("oracle_reference_timeout", 7200) - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(f"No completed oracle reference for {task} in {roots}")
        time.sleep(min(1, remaining))

    import imageio.v2 as imageio

    path, summary = found[0]
    contract = json.loads((path.parent / "eval_contract.json").read_text())
    if contract.get("revision") != REVISION:
        raise ValueError("Oracle reference uses a different simulator revision")
    for key in ("task_config", "instruction_type", "start_seed", "episodes"):
        if contract.get(key) != config[key]:
            raise ValueError(f"Oracle reference disagrees with {key}")
    if Path(contract["robotwin_root"]).resolve() != Path(config["robotwin_root"]).resolve():
        raise ValueError("Oracle reference uses a different simulator installation")
    rows = [row for row in summary["records"] if row["task"] == task and row["status"] == "evaluated"]
    if len(rows) != config["episodes"]:
        raise ValueError("Oracle reference has an incompatible episode count")
    references = []
    for row in rows:
        directory = path.parent / task / f"seed_{row['seed']:09d}"
        with np.load(directory / "initial_state.npz", allow_pickle=False) as values:
            initial = {key: values[key].copy() for key in ("state", "cam2world_gl")}
        initial["images"] = {
            camera: imageio.imread(directory / f"initial_{camera}.png") for camera in ("head", "left", "right")
        }
        goal = {camera: imageio.imread(directory / f"goal_{camera}.png") for camera in initial["images"]}
        references.append(
            dict(
                seed=row["seed"],
                instruction=row["instruction"],
                initial=initial,
                goal=goal,
                source=str(path.resolve()),
                goal_sha256={camera: hashlib.sha256(rgb.tobytes()).hexdigest() for camera, rgb in goal.items()},
            )
        )
    return references
