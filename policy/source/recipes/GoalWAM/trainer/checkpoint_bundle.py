"""Portable checkpoint assets and completion manifests; no model arithmetic."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path


BUNDLE_VERSION = 1
BUNDLE_ASSETS = {
    "vae": "vae/Wan2.2_VAE.pth",
    "tokenizer": "text_tokenizer",
    "backbone_config": "architecture/Qwen3-VL-8B-Instruct.json",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def local_asset(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Checkpoint asset must stay inside its directory: {relative}")
    return path


def read_bundle(root):
    """Validate completeness and local dependencies without hashing model shards on load."""
    root = Path(root).resolve()
    if not (root / "complete.json").is_file():
        raise ValueError(f"Incomplete GoalWAM bundle: {root}")
    complete = json.loads((root / "complete.json").read_text())
    if complete.get("manifest_sha256") != sha256_file(root / "bundle_manifest.json"):
        raise ValueError("Checkpoint completion marker does not match its bundle manifest")
    manifest = json.loads((root / "bundle_manifest.json").read_text())
    if manifest.get("version") != BUNDLE_VERSION or manifest.get("format") != "goalwam-dcp":
        raise ValueError("Unsupported GoalWAM checkpoint bundle format")
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "goalwam" or config.get("bundle_assets") != manifest["assets"]:
        raise ValueError("Checkpoint configuration and bundled assets disagree")
    for relative, record in manifest["files"].items():
        path = local_asset(root, relative)
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError(f"Missing or truncated checkpoint file: {relative}")
        if relative in {"config.json", manifest["assets"]["backbone_config"]}:
            if sha256_file(path) != record.get("sha256"):
                raise ValueError(f"Checkpoint configuration checksum mismatch: {relative}")
    for key, relative in manifest["assets"].items():
        path = local_asset(root, relative)
        if not path.exists():
            raise ValueError(f"Missing bundled {key}: {relative}")
    if not (root / "model/.metadata").is_file():
        raise ValueError("Checkpoint has no native DCP model metadata")
    return manifest


def copy_assets(destination, *, vae, tokenizer, backbone_config):
    """Materialize independent files; moving the checkpoint never needs source symlinks."""
    destination = Path(destination)
    for key, source in (("vae", vae), ("tokenizer", tokenizer), ("backbone_config", backbone_config)):
        target = destination / BUNDLE_ASSETS[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        if key == "tokenizer":
            shutil.copytree(source, target, ignore=shutil.ignore_patterns(".cache", "__pycache__"))
        else:
            shutil.copy2(source, target)


def save_portable_config(config, destination):
    """Replace runtime machine paths with bundle-relative references in the exported config."""
    config = copy.deepcopy(config)
    config.bundle_assets = dict(BUNDLE_ASSETS)
    native = config.cosmos
    model = native["model"]["config"]
    model["tokenizer"]["vae_path"] = BUNDLE_ASSETS["vae"]
    model["vlm_config"]["tokenizer"]["pretrained_model_name"] = BUNDLE_ASSETS["tokenizer"]
    model["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"] = BUNDLE_ASSETS["backbone_config"]
    model["vlm_config"]["pretrained_weights"]["backbone_path"] = "."
    if model["vlm_config"]["pretrained_weights"]["enabled"]:
        raise ValueError("A complete GoalWAM checkpoint must not reload an external reasoner")
    native["checkpoint"]["load_path"] = "."
    config._name_or_path = ""
    config.save_pretrained(destination)


def finish_bundle(root, *, kind, iteration, world_size, provenance=None):
    """Publish the completion marker only after weights, config and all assets exist.

    DCP payloads carry sizes here; the independent audit streams their tensors.
    Small metadata and immutable auxiliary assets also receive SHA256 hashes.
    """
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "goalwam" or config.get("bundle_assets") != BUNDLE_ASSETS:
        raise ValueError("Cannot complete bundle without its portable GoalWAM configuration")
    for relative in ["model/.metadata", *BUNDLE_ASSETS.values()]:
        if not local_asset(root, relative).exists():
            raise ValueError(f"Cannot complete bundle with missing asset: {relative}")
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"bundle_manifest.json", "complete.json"}:
            continue
        relative = str(path.relative_to(root))
        record = {"size": path.stat().st_size}
        if path.suffix != ".distcp":
            record["sha256"] = sha256_file(path)
        files[relative] = record
    manifest = dict(
        version=BUNDLE_VERSION,
        format="goalwam-dcp",
        kind=kind,
        iteration=iteration,
        world_size=world_size,
        assets=BUNDLE_ASSETS,
        files=files,
        provenance=provenance or {},
    )
    write_json(root / "bundle_manifest.json", manifest)
    write_json(
        root / ".complete.json.tmp",
        dict(
            iteration=iteration,
            world_size=world_size,
            bundle_version=BUNDLE_VERSION,
            manifest_sha256=sha256_file(root / "bundle_manifest.json"),
        ),
    )
    (root / ".complete.json.tmp").replace(root / "complete.json")
    read_bundle(root)
    return manifest
