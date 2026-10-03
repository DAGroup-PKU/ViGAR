"""Use the pinned upstream trainer; no alternate loss or optimizer implementation."""
import argparse
import os
from pathlib import Path

from protocol import CHECKPOINT, MODES, OUTPUT, SOURCE, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--save-every', type=int, required=True)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    os.environ['QUALITY_MODE'] = args.mode
    os.environ['IMAGINAIRE_OUTPUT_ROOT'] = str(OUTPUT / 'training')
    # Some containers have no writable home directory. W&B tables stage artifacts
    # separately from WANDB_DIR; configure that location before importing W&B.
    wandb_staging = Path('/tmp/goalwam-wandb-artifacts') / args.mode
    wandb_staging.mkdir(parents=True, exist_ok=True)
    os.environ['WANDB_DATA_DIR'] = str(wandb_staging)
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils.context_managers import distributed_init, model_init, data_loader_init
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.serialization import to_yaml
    from training_proof import TrainingProof

    with distributed_init():
        distributed.init()
    world = int(os.environ['WORLD_SIZE'])
    overrides = [f'job.project=i2i_quality', f'job.group={args.mode}', 'job.name=seed42',
        'job.wandb_mode=online', f'checkpoint.load_path={args.resume or CHECKPOINT}',
        f'checkpoint.load_training_state={str(bool(args.resume)).lower()}', 'checkpoint.only_load_scheduler_state=false',
        'checkpoint.keys_to_skip_loading=[]', 'checkpoint.load_ema_to_reg=false',
        f'checkpoint.load_ema_to_reg_single_net={str(not bool(args.resume)).lower()}', 'checkpoint.dcp_async_mode_enabled=false',
        f'checkpoint.save_iter={args.save_every}', 'model.config.ema.enabled=false',
        'model.config.compile.enabled=false', f'model.config.parallelism.data_parallel_shard_degree={world}',
        'model.config.parallelism.data_parallel_replicate_degree=1', 'trainer.grad_accum_iter=1',
        f'trainer.max_iter={args.steps}', 'trainer.logging_iter=10', 'trainer.seed=42',
        'trainer.run_validation=false', 'trainer.run_validation_on_start=false',
        'optimizer.lr=2e-5', f'scheduler.cycle_lengths=[{args.steps}]', 'scheduler.warm_up_steps=[50]',
        'scheduler.f_min=[1.0]', 'scheduler.f_start=[0.1]',
        f'dataloader_train.batcher.max_samples_per_batch={args.batch}',
        'dataloader_train.num_workers=2', 'dataloader_train.prefetch_factor=2',
        'dataloader_train.distributor.dataset.metadata_num_workers=4',
        'dataloader_train.distributor.dataset._target_=training_dataset.get_dataset',
        'dataloader_train.collator._target_=roi_loss.ROICollator',
        'model.config.activation_checkpointing.mode=full']
    config = load_experiment_from_toml(str(SOURCE / 'episode_image_edit_nano.toml'), extra_overrides=overrides)
    # These optional visualization callbacks unconditionally write BatchInfo to
    # s3://rundir. The matched offline image evaluation replaces their role.
    from omegaconf import open_dict
    with open_dict(config.trainer.callbacks):
        removed = [name for name in ('every_n_sample_reg', 'every_n_sample_ema')
                   if config.trainer.callbacks.pop(name, None) is not None]
    print('S3_DRAW_CALLBACKS_DISABLED', removed, flush=True)
    config.validate()
    config.freeze()
    trainer = config.trainer.type(config)
    proof = TrainingProof(OUTPUT / 'training_proofs' / args.mode, args.batch, world)
    proof.config, proof.trainer = config, trainer
    trainer.callbacks._callbacks.append(proof)
    if distributed.get_rank() == 0:
        Path(config.job.path_local).mkdir(parents=True, exist_ok=True)
        to_yaml(config, str(Path(config.job.path_local) / 'config.yaml'))
        write_json(OUTPUT / 'audit' / f'launch_{args.mode}.json', dict(mode=args.mode, steps=args.steps,
                   per_gpu_batch=args.batch, world_size=world, global_batch=args.batch * world,
                   initialization='Resume full training state' if args.resume else 'Original RoboTwin I2I 30k EMA; fresh optimizer/schedule',
                   source_checkpoint=str(args.resume or CHECKPOINT), disabled_callbacks=removed,
                   code_root=str(SOURCE), overrides=overrides))
    with model_init():
        model = instantiate(config.model)
    with data_loader_init():
        train_data = instantiate(config.dataloader_train)
    trainer.train(model, train_data, None)
    # train() destroys its process group on success; each rank writes a distinct receipt.
    write_json(OUTPUT / 'training_proofs' / args.mode / f'completed.rank{os.environ["RANK"]}.json',
               dict(complete=True, optimizer_steps=args.steps))


if __name__ == '__main__':
    main()
