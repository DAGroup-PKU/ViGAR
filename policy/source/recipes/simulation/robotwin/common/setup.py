"""Install the pinned, model-independent RoboTwin simulator with uv/Python 3.10."""

from __future__ import annotations

import argparse
import ctypes.util
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
UPSTREAM = "https://github.com/RoboTwin-Platform/RoboTwin.git"
REVISION = "c3ddfa8b97d5519efa828b075999bd0006778e5e"
CUROBO_REVISION = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
ASSET_REPO = "TianxingChen/RoboTwin2.0"
ASSET_REVISION = "9dc9299c163db059931898a9f0852098a61155a1"
OIDN_VERSION = "2.3.3"
OIDN_SHA256 = "3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f"


def run(command, **kwargs):
    # Parent training pyproject overrides must never alter simulator resolution.
    if command[0] == "uv":
        command = ["uv", "--no-config", *command[1:]]
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def git_head(root):
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def checkout(root, url, revision):
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", "--no-checkout", url, root])
        run(["git", "-C", root, "checkout", "--detach", revision])
    actual_root = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True).strip()
    if Path(actual_root).resolve() != root.resolve() or git_head(root) != revision:
        raise ValueError(f"Expected standalone checkout {root} at {revision}; existing files were preserved")


def patch_simulator_packages(python):
    # These are the compatibility edits in the pinned upstream installation
    # guide. They are confined to the dedicated simulator environment.
    run(
        [
            python,
            "-c",
            """
from pathlib import Path
import sapien, mplib
p = Path(sapien.__file__).parent / "wrapper/urdf_loader.py"
t = p.read_text()
t = t.replace('with open(urdf_file, "r") as f:', 'with open(urdf_file, "r", encoding="utf-8") as f:')
t = t.replace('srdf_file = urdf_file[:-4] + "srdf"', 'srdf_file = urdf_file[:-4] + ".srdf"')
t = t.replace('with open(srdf_file, "r") as f:', 'with open(srdf_file, "r", encoding="utf-8") as f:')
p.write_text(t)
p = Path(mplib.__file__).parent / "planner.py"
p.write_text(p.read_text().replace(
    "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
    "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"))
""",
        ]
    )


def install_oidn(python, cache):
    archive = cache / f"oidn-{OIDN_VERSION}.x86_64.linux.tar.gz"
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        url = f"https://github.com/RenderKit/oidn/releases/download/v{OIDN_VERSION}/{archive.name}"
        run(["curl", "-L", "--fail", "--retry", "3", url, "-o", temporary])
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != OIDN_SHA256:
        raise ValueError(f"OIDN checksum mismatch: {archive}; repair/remove this archive before retrying")
    site = Path(
        subprocess.check_output(
            [str(python), "-c", "import sapien; from pathlib import Path; print(Path(sapien.__file__).parent)"],
            text=True,
        )
        .strip()
        .splitlines()[-1]
    )
    with tarfile.open(archive) as tar:
        for name in ("libOpenImageDenoise", "libOpenImageDenoise_core", "libOpenImageDenoise_device_cuda"):
            filename = f"{name}.so.{OIDN_VERSION}"
            member = f"oidn-{OIDN_VERSION}.x86_64.linux/lib/{filename}"
            with tar.extractfile(member) as source, (site / "oidn_library" / filename).open("wb") as target:
                shutil.copyfileobj(source, target)
    patch = site / "_oidn_tricks.py"
    patch.write_text(patch.read_text().replace("2.0.1", OIDN_VERSION))


def install_environment(args):
    if not shutil.which("uv"):
        raise ValueError("Install uv first; setup never modifies the active model environment")
    env = args.workspace / ".venv"
    if not env.exists():
        run(["uv", "venv", "--python", "3.10", "--seed", env])
    python = env / "bin/python"
    run([python, "-c", "import sys; assert sys.version_info[:2] == (3, 10)"])
    marker = env / "robotwin_environment.json"
    fingerprint = dict(
        requirements_sha256=hashlib.sha256((HERE / "requirements.txt").read_bytes()).hexdigest(),
        curobo_revision=CUROBO_REVISION,
        cuda_home=str(args.cuda_home),
        cuda_archs=args.cuda_archs,
        oidn=OIDN_VERSION,
    )
    if marker.exists() and json.loads(marker.read_text()) == fingerprint:
        print("Simulator environment already installed; run --stage check to verify runtime.")
        return
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "setuptools==69.5.1",
            "wheel==0.45.1",
            "ninja==1.11.1.3",
            "Cython==3.0.12",
            "numpy==1.26.4",
        ]
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--index-strategy",
            "unsafe-best-match",
            "--no-build-isolation",
            "-r",
            HERE / "requirements.txt",
        ]
    )
    run(
        [
            python,
            "-c",
            "import torch, torchvision; assert torch.__version__ == '2.7.1+cu128'; "
            "assert torchvision.__version__ == '0.22.1+cu128'; assert torch.version.cuda == '12.8'",
        ]
    )
    patch_simulator_packages(python)
    curobo = args.root / "envs/curobo"
    checkout(curobo, "https://github.com/NVlabs/curobo.git", CUROBO_REVISION)
    if not (args.cuda_home / "bin/nvcc").is_file():
        raise ValueError(f"CUDA toolkit missing: {args.cuda_home}")
    environment = dict(os.environ)
    environment.update(
        CUDA_HOME=str(args.cuda_home),
        CUDA_PATH=str(args.cuda_home),
        PATH=f"{args.cuda_home}/bin:{env}/bin:{environment['PATH']}",
        TORCH_CUDA_ARCH_LIST=args.cuda_archs,
        MAX_JOBS=str(args.jobs),
    )
    run(["uv", "pip", "install", "--python", python, "--no-build-isolation", "-e", curobo], env=environment)
    install_oidn(python, args.workspace / "cache")
    run(["uv", "pip", "freeze", "--python", python], stdout=(args.workspace / "environment.lock.txt").open("w"))
    marker.write_text(json.dumps(fingerprint, indent=2) + "\n")


def install_assets(args):
    python = args.workspace / ".venv/bin/python"
    # Extraction into a staging directory leaves prior assets intact on failure.
    # Pin archives to a content revision; HF's cache verifies downloaded objects.
    code = """
import json, os, shutil, sys, tempfile, time, zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download
root, cache, repo, revision = sys.argv[1:]
root, cache = Path(root), Path(cache)
for name, minimum in [("background_texture",10000),("embodiments",200),("objects",9000)]:
    target = root / "assets" / name
    marker = root / "assets" / ("." + name + ".revision")
    if target.exists():
        if not marker.exists() or marker.read_text().strip() != revision:
            raise ValueError(f"Unmanaged assets at {target}; use a fresh checkout or verify them separately")
        if sum(p.is_file() for p in target.rglob("*")) < minimum:
            raise ValueError(f"Incomplete assets at {target}")
        continue
    for attempt in range(6):
        try:
            archive = hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision,
                                      filename=name+".zip", cache_dir=cache)
            break
        except Exception as error:
            cause, retryable = error, False
            while cause is not None:
                status = getattr(getattr(cause, "response", None), "status_code", None)
                retryable |= status in (429, 500, 502, 503, 504)
                retryable |= isinstance(cause, (TimeoutError, ConnectionError))
                cause = cause.__cause__
            if not retryable or attempt == 5:
                raise
            delay = min(5 * 2**attempt, 60)
            print(f"Transient Hub error for {name}; retrying in {delay}s", flush=True)
            time.sleep(delay)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=root/"assets") as staging:
        staging = Path(staging)
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                path = (staging/item.filename).resolve()
                if not path.is_relative_to(staging.resolve()):
                    raise ValueError("Archive path traversal")
            z.extractall(staging)
        extracted = staging / name
        if sum(p.is_file() for p in extracted.rglob("*")) < minimum:
            raise ValueError(f"Incomplete downloaded {name}")
        extracted.rename(target)
    marker.write_text(revision+"\\n")
"""
    run([python, "-c", code, args.root, args.workspace / "cache/huggingface", ASSET_REPO, ASSET_REVISION])
    run([python, "script/update_embodiment_config_path.py"], cwd=args.root)


def runtime_environment(workspace):
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PATH"] = f"{workspace}/.venv/bin:{environment['PATH']}"
    # Some containers mount NVIDIA graphics libraries without an ICD JSON.
    # Create only a workspace-local descriptor; do not touch /etc or drivers.
    if not environment.get("VK_ICD_FILENAMES") and ctypes.util.find_library("GLX_nvidia"):
        icd = workspace / "nvidia_icd.json"
        icd.write_text(
            json.dumps(
                {"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"}}
            )
            + "\n"
        )
        environment["VK_ICD_FILENAMES"] = str(icd)
    if not environment.get("__EGL_VENDOR_LIBRARY_FILENAMES") and ctypes.util.find_library("EGL_nvidia"):
        egl = workspace / "nvidia_egl.json"
        egl.write_text(
            json.dumps({"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}) + "\n"
        )
        environment["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl)
    return environment


def check(args):
    python = args.workspace / ".venv/bin/python"
    environment = runtime_environment(args.workspace)
    run(
        [python, HERE / "preflight.py", "--root", args.root, "--output", args.workspace / "preflight.json"],
        env=environment,
        cwd=args.root,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("ROBOTWIN_WORKSPACE", "/path/to/vigar/assets")),
    )
    parser.add_argument("--root", type=Path, help="Standalone RoboTwin checkout; default WORKSPACE/RoboTwin")
    parser.add_argument("--stage", choices=["all", "checkout", "environment", "assets", "check"], default="all")
    parser.add_argument("--cuda-home", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8")))
    parser.add_argument("--cuda-archs", default=os.environ.get("TORCH_CUDA_ARCH_LIST", "8.9;9.0"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    args.root = (args.root or args.workspace / "RoboTwin").resolve()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    print(
        json.dumps(
            dict(
                root=str(args.root),
                workspace=str(args.workspace),
                revision=REVISION,
                asset_revision=ASSET_REVISION,
                stage=args.stage,
                python="3.10",
                model_packages=False,
            ),
            indent=2,
        )
    )
    if args.dry_run:
        return
    (args.workspace / "cache").mkdir(parents=True, exist_ok=True)
    os.environ.update(runtime_environment(args.workspace))
    os.environ.setdefault("UV_HTTP_TIMEOUT", "300")
    os.environ.setdefault("UV_LINK_MODE", "copy")
    checkout(args.root, UPSTREAM, REVISION)
    for stage, function in [("environment", install_environment), ("assets", install_assets), ("check", check)]:
        if args.stage in {"all", stage}:
            function(args)
    runtime = dict(
        root=str(args.root),
        python=str(args.workspace / ".venv/bin/python"),
        revision=REVISION,
        asset_revision=ASSET_REVISION,
        workspace=str(args.workspace),
    )
    (args.workspace / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")


if __name__ == "__main__":
    main()
