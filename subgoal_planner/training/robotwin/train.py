import argparse
import os
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "training_runtime"
OUTPUT = Path(os.environ["SUBGOAL_PLANNER_OUTPUT"])
CHECKPOINT = Path(os.environ["BASE_CHECKPOINT_PATH"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(OUTPUT)
    wandb_staging = OUTPUT / "wandb"
    wandb_staging.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_staging))
    from cosmos_framework.configs.toml_config.sft_config import (
        load_experiment_from_toml,
    )
    from cosmos_framework.utils.context_managers import (
        distributed_init,
        model_init,
        data_loader_init,
    )
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.serialization import to_yaml

    with distributed_init():
        distributed.init()
    world = int(os.environ["WORLD_SIZE"])
    if world != 8:
        raise ValueError("Expected 8 GPUs and global batch 32")
    overrides = [
        "job.project=vigar",
        "job.group=robotwin",
        "job.name=subgoal_planner",
        f"job.wandb_mode={os.environ.get('WANDB_MODE', 'offline')}",
        f"checkpoint.load_path={args.resume or CHECKPOINT}",
        f"checkpoint.load_training_state={str(bool(args.resume)).lower()}",
        "checkpoint.only_load_scheduler_state=false",
        "checkpoint.keys_to_skip_loading=[]",
        "checkpoint.load_ema_to_reg=false",
        "checkpoint.load_ema_to_reg_single_net=false",
        "checkpoint.dcp_async_mode_enabled=false",
        "checkpoint.save_iter=10000",
        "model.config.ema.enabled=false",
        "model.config.compile.enabled=false",
        f"model.config.parallelism.data_parallel_shard_degree={world}",
        "model.config.parallelism.data_parallel_replicate_degree=1",
        "trainer.grad_accum_iter=1",
        "trainer.max_iter=100000",
        "trainer.logging_iter=10",
        "trainer.seed=42",
        "trainer.run_validation=false",
        "trainer.run_validation_on_start=false",
        "optimizer.lr=2e-5",
        "scheduler.cycle_lengths=[100000]",
        "scheduler.warm_up_steps=[50]",
        "scheduler.f_min=[1.0]",
        "scheduler.f_start=[0.1]",
        "dataloader_train.batcher.max_samples_per_batch=4",
        "dataloader_train.num_workers=2",
        "dataloader_train.prefetch_factor=2",
        "dataloader_train.distributor.dataset.metadata_num_workers=4",
        "dataloader_train.distributor.dataset._target_=training_dataset.get_dataset",
        "dataloader_train.distributor.dataset.next_subgoal_tail_fraction=0.15",
        "dataloader_train.collator._target_=roi_loss.ROICollator",
        "model.config.activation_checkpointing.mode=full",
    ]
    config = load_experiment_from_toml(
        str(SOURCE / "episode_image_edit_nano.toml"), extra_overrides=overrides
    )
    from omegaconf import open_dict

    with open_dict(config.trainer.callbacks):
        for name in ("every_n_sample_reg", "every_n_sample_ema"):
            config.trainer.callbacks.pop(name, None)
    config.validate()
    config.freeze()
    trainer = config.trainer.type(config)
    if distributed.get_rank() == 0:
        Path(config.job.path_local).mkdir(parents=True, exist_ok=True)
        to_yaml(config, str(Path(config.job.path_local) / "config.yaml"))
    with model_init():
        model = instantiate(config.model)
    with data_loader_init():
        train_data = instantiate(config.dataloader_train)
    trainer.train(model, train_data, None)


if __name__ == "__main__":
    main()
