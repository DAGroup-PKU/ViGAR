import os,sys,json
from pathlib import Path
_PROFILE_PATH=Path(__file__).resolve().parents[2]/'configs/i2i_backend.json'
_PROFILE=json.loads(_PROFILE_PATH.read_text())
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'i2i/inference_runtime'))
os.environ["GOALWAM_BATCH_CFG"]="0"
"""Serve the identical frozen planner used to build the training goal cache."""

import argparse
import threading
import time

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.utils.context_managers import distributed_init, model_init
from cosmos_framework.utils.lazy_config import instantiate
from evaluate_episode_image_edit_nano import _keep_eval_callbacks

from robotwin_i2i_goal import generate_goal
from robotwin_wire import serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sft-toml", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ema", action="store_true", help="Only original30k uses EMA; continuation checkpoints store regular weights")
    args = parser.parse_args()

    # Same EMA-to-single-net path as generate_i2i_cache.py; inference owns one GPU.
    from cosmos_framework.utils import distributed
    import torch.distributed as dist
    with distributed_init():
        distributed.init()
    overrides = [
        "job.project=vigar", "job.group=i2i_async_eval",
        f"job.name=planner_port{args.port}", "job.wandb_mode=disabled",
        f"checkpoint.load_path={args.checkpoint}", "checkpoint.load_training_state=false",
        "checkpoint.only_load_scheduler_state=false", "checkpoint.keys_to_skip_loading=[]",
        "checkpoint.dcp_async_mode_enabled=false", "checkpoint.load_ema_to_reg=false",
        f"checkpoint.load_ema_to_reg_single_net={str(args.ema).lower()}", "model.config.ema.enabled=false",
        "trainer.run_validation=false", "trainer.run_validation_on_start=false",
        "trainer.max_iter=1", "model.config.compile.enabled=false",
        "model.config.parallelism.data_parallel_shard_degree=1",
        "model.config.parallelism.data_parallel_replicate_degree=1",
    ]
    config = load_experiment_from_toml(args.sft_toml, extra_overrides=overrides)
    config.validate()
    config.freeze()
    trainer = config.trainer.type(config)
    _keep_eval_callbacks(trainer)
    with model_init():
        model = instantiate(config.model)
    model = model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    trainer.checkpointer.load(model)
    trainer.callbacks.on_train_start(model, iteration=30000)
    model.eval()
    from inference_backend import CheckedBackend
    backend=CheckedBackend(model,generate_goal,_PROFILE)
    print("I2I_BACKEND pending same-instance exact-pixel validation", flush=True)
    lock = threading.Lock()

    def handle(request):
        if request.get("cmd") == "ping":
            return {"ok": True, "checkpoint": args.checkpoint, "weight_variant": "ema" if args.ema else "regular",
                    "canvas_hw": [384, 320]}
        if request.get("cmd") != "infer":
            raise ValueError("I2I accepts ping/infer only")
        if set(request) - {"cmd", "image", "prompt", "seed"}:
            raise ValueError("Planner accepts current image, global instruction and seed only")
        seed = int(request.get("seed", 0))
        started = time.monotonic()
        with lock:
            goal = backend.generate(request["image"], request["prompt"], seed=seed)
        return {"goal_image": goal, "seed": seed,
                "server_timing": {"i2i_seconds": time.monotonic() - started}}

    print(f"I2I_ASYNC_MODEL_READY checkpoint={args.checkpoint} EMA={int(args.ema)} shard=1", flush=True)
    try:
        serve(handle, args.port)
    finally:
        trainer.checkpointer.finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
