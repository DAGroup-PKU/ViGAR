"""Native Cosmos training lifecycle for the real_robot_900hr policy.

Same network, loss and update loop as train_native.py. The AgiBot A2 data
contract (34-D actions, 128-step chunks, 17-frame video) and the real-robot
optimizer schedule replace the RoboTwin settings.
"""
import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('COSMOS_TRAINING', '1')


def verify_inference(core, raw, tokenizer_config, out, rank):
    """Generate one action chunk from an observed state and goal."""
    import torch
    from cosmos_framework.utils import misc
    from cosmos_framework.model.vfm.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig
    from recipes.ViGAR.data.data_loader import collate_samples
    from recipes.ViGAR.trainer.evaluator import rng_state, restore_rng, write_json
    from recipes.simulation.robotwin.vigar.policy import reshard_network
    from real_robot_dataset import ACTION_DIM, RealRobotSFTDataset

    saved = rng_state()
    core.eval()
    try:
        with torch.no_grad():
            x = raw[0]
            x['action'][1:] = 0
            sample = RealRobotSFTDataset(raw, tokenizer_config=tokenizer_config, cfg_dropout_rate=0.)._transform(
                x, '320x384')
            sampler = UniPCSampler(cfg=UniPCSamplerConfig(), tensor_kwargs=core.tensor_kwargs)
            generated = core.generate_samples_from_batch(misc.to(collate_samples([sample]), device='cuda'),
                                                         sampler=sampler, seed=[9000], num_steps=10, guidance=1.,
                                                         shift=2.)
            normalized = generated['action'][0][1:, :ACTION_DIM].float().cpu()
            assert normalized.shape == (raw.chunk, ACTION_DIM) and torch.isfinite(normalized).all()
            commands = normalized * raw.action_stats['std'] + raw.action_stats['mean']
            assert torch.isfinite(commands).all()
            write_json(out / f'inference.rank{rank}.json', dict(complete=True, command_shape=list(commands.shape),
                       finite=True, solver='UniPC', steps=10, guidance=1., shift=2.))
    finally:
        reshard_network(core.net)
        core.train()
        restore_rng(saved)


def main():
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    import wandb
    from omegaconf import OmegaConf
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions
    from cosmos_framework.model.vfm.omni_mot_model import OmniMoTModel
    from cosmos_framework.utils import misc
    from cosmos_framework.utils.callback import CallBackGroup
    from cosmos_framework.inference.common.init import _init_log_console
    from recipes.ViGAR.data.data_loader import StatefulWindowLoader
    from recipes.ViGAR.trainer.evaluator import save_checkpoint, seed_all, write_json
    from recipes.ViGAR.trainer.callbacks import UpdateAudit
    from recipes.ViGAR.trainer.checkpoint_bundle import copy_assets
    from veomni.models.transformers.vigar.configuration_vigar import ViGARConfig
    from checkpoint_directory import reserve_checkpoint_directory
    from real_robot_dataset import RESOLUTION, ROOT, SOURCE, RealRobotDataset, RealRobotSFTDataset

    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--save-every', type=int, default=10000)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--resume', type=Path)
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', device_id=torch.device('cuda', int(os.environ['LOCAL_RANK'])))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert (a.batch == 1 and a.steps == 2) if a.preflight else (a.batch * world == 1024)
    run_name = os.environ.get('VIGAR_POLICY_RUN_NAME', 'real_robot_900hr')
    out = ROOT / 'runs' / (('preflight_' if a.preflight else '') + run_name)
    out.mkdir(parents=True, exist_ok=True)
    _init_log_console()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed_all(42)
    raw = RealRobotDataset(os.environ['VIGAR_REAL_ROBOT_DATASET'], parquet_cache_dir=os.environ.get('VIGAR_PARQUET_CACHE'))
    model_config = ViGARConfig.from_json_file(SOURCE / 'recipes/ViGAR/configs/migration/initialization_config.json')
    base = Path(os.environ['BASE_CHECKPOINT_PATH'])
    vae = Path(os.environ['WAN_VAE_PATH'])
    tokenizer = Path(os.environ['QWEN_TOKENIZER_PATH'])
    cfg = model_config.runtime_config(base_checkpoint=base, vae_path=vae, tokenizer_path=tokenizer, shard_degree=world)
    model = cfg.model.config
    model.resolution = RESOLUTION
    model.rectified_flow_training_config.shift[RESOLUTION] = 5
    model.tokenizer.encode_chunk_frames[RESOLUTION] = 24
    model.tokenizer.encode_exact_durations = [raw.chunk // raw.video_stride + 1]
    model.vlm_config.model_instance.config.freeze_und = True
    model.compile.enabled = False
    model.activation_checkpointing.mode = 'full'
    model.log_enc_time_every_n = 0
    cfg.optimizer.lr = 1e-4
    cfg.optimizer.disable_weight_decay_for_1d_params = True
    cfg.optimizer.lr_multipliers = {'action2llm': 5., 'llm2action': 5., 'action_modality_embed': 5.,
                                    'goal_vision_embed': 5.}
    cfg.scheduler.f_max = [.4]
    cfg.scheduler.f_min = [0.]
    cfg.scheduler.f_start = [0.]
    cfg.scheduler.warm_up_steps = [0]
    cfg.scheduler.cycle_lengths = [500000]
    cfg.trainer.max_iter = a.steps
    cfg.job.project = 'vigar_real_robot'
    cfg.job.group = 'vigar_real_robot'
    cfg.job.name = os.environ.get('WANDB_NAME', run_name) if not a.preflight else 'vigar_real_robot_preflight'
    cfg.job.wandb_mode = os.environ.get('WANDB_MODE', 'offline')
    cfg.job.path_local = str(out)
    signal = torch.tensor(float(rank), device='cuda')
    dist.all_reduce(signal)
    assert signal.item() == world*(world-1)/2
    write_json(out / f'nccl.rank{rank}.json', dict(world=world, sum=signal.item(), host=os.uname().nodename))

    core = OmniMoTModel(cfg.model.config)
    core.on_train_start(torch.preserve_format)
    optim, scheduler = core.init_optimizer_scheduler(cfg.optimizer, cfg.scheduler)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    dataset = RealRobotSFTDataset(raw, tokenizer_config=cfg.model.config.vlm_config.tokenizer, cfg_dropout_rate=.1)
    loader = StatefulWindowLoader(dataset, rank=rank, world_size=world, seed=42,
                                  batch_size=a.batch, num_workers=a.num_workers, prefetch_factor=2)
    state = get_model_state_dict(core)
    fresh = ('action2llm', 'llm2action', 'action_modality_embed', 'action_pos_embed', 'goal_vision_embed')
    required = {k: v for k, v in state.items() if k.startswith('net.') and not any(s in k for s in fresh)}
    metadata = dcp.FileSystemReader(base / 'model').read_metadata().state_dict_metadata
    missing = [k for k, v in required.items() if k not in metadata or metadata[k].size != v.shape]
    if missing:
        raise ValueError(f'Base weights missing/shape mismatch: {missing[:10]}')
    dcp.load(required, checkpoint_id=base / 'model')
    state.update(required)
    incompatible = set_model_state_dict(core, state, options=StateDictOptions(strict=False))
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    core.net_ema_worker.copy_to(src_model=core.net, tgt_model=core.net_ema)
    seed_all(42000 + rank)
    start = 0
    if a.resume:
        from recipes.ViGAR.trainer.evaluator import load_checkpoint
        start = load_checkpoint(core, optim, scheduler, loader, a.resume, resume=True)
    recipe = dict(data=dict(raw.selection_record(), resolution=RESOLUTION, goal_layout='concat_320x384',
                            cfg_dropout_rate=.1),
                  train=dict(micro_batch_size=a.batch, global_batch_size=a.batch*world, max_steps=a.steps,
                             checkpoint_every=a.save_every, lr=cfg.optimizer.lr, seed=42))
    run = None
    if rank == 0:
        run = wandb.init(project=cfg.job.project, entity=os.environ.get('WANDB_ENTITY'), name=cfg.job.name,
                         dir=str(out), mode=cfg.job.wandb_mode, config=recipe)
        write_json(out / 'wandb.json', dict(id=run.id, url=run.url, mode=cfg.job.wandb_mode))
        write_json(out / 'recipe_config.json', recipe)
        write_json(out / 'data_contract.json', raw.selection_record())
        write_json(out / 'base_mapping.json', dict(source=str(base), loaded=len(required),
                   initialized_fresh=[k for k in state if k.startswith('net.') and k not in required]))
        OmegaConf.save(cfg, out / 'native_config.yaml')
    context = SimpleNamespace(config=cfg, grad_scaler=scaler)
    callbacks = CallBackGroup(cfg, context)
    callbacks._callbacks.insert(0, UpdateAudit(record=False))
    callbacks.on_train_start(core, iteration=start)
    iterator = iter(loader)
    optim.zero_grad(set_to_none=True)
    trainable = [(n, p) for n, p in core.net.named_parameters() if p.requires_grad]
    probe_name, probe = next((n, p) for n, p in trainable if 'goal_vision_embed' in n)
    to_local = lambda t: t.to_local() if hasattr(t, 'to_local') else t
    metrics_file = out / f'progress.rank{rank}.jsonl'
    def save(step):
        path = out / 'checkpoints' / f'iter_{step:09d}'
        reserve_checkpoint_directory(path, dist)
        save_checkpoint(core, optim, scheduler, loader, step, path, mark_complete=False)
        if rank == 0:
            copy_assets(path, vae=vae, tokenizer=tokenizer,
                        backbone_config=cfg.model.config.vlm_config.model_instance.config.base_config.json_file)
            OmegaConf.save(cfg, path / 'native_config.yaml')
            write_json(path / 'recipe_config.json', recipe)
            write_json(path / 'data_contract.json', raw.selection_record())
            write_json(path / 'complete.json', dict(iteration=step, world_size=world))
            print('CHECKPOINT_COMPLETE', path, flush=True)
        dist.barrier()
    window_started = time.monotonic()
    window_steps = 0
    window_data_max = 0.
    window_step_max = 0.
    for it in range(start, a.steps):
        began = time.monotonic()
        batch = misc.to(next(iterator), device='cuda')
        loaded = time.monotonic()
        callbacks.on_training_step_start(core, batch, iteration=it)
        lr_used = [float(group['lr']) for opt in optim.optimizers for group in opt.param_groups]
        callbacks.on_before_forward(iteration=it)
        output, loss = core.training_step(batch, it)
        callbacks.on_after_forward(iteration=it)
        callbacks.on_before_backward(core, loss, iteration=it)
        scaler.scale(loss).backward()
        core.on_after_backward()
        callbacks.on_after_backward(core, iteration=it)
        g = to_local(probe.grad).detach().float().abs().sum().item()
        before = to_local(probe).detach().float().clone()
        callbacks.on_before_optimizer_step(core, optim, scheduler, scaler, iteration=it)
        scaler.step(optim)
        scaler.update()
        scheduler.step()
        callbacks.on_before_zero_grad(core, optim, scheduler, iteration=it)
        core.on_before_zero_grad(optim, scheduler, iteration=it)
        delta = (to_local(probe).detach().float()-before).abs().sum().item()
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-began
        values = torch.tensor([loss.detach().item(), g, delta, elapsed, loaded-began,
                               output['flow_matching_loss_action'].detach().item(),
                               output['flow_matching_loss_vision'].detach().item()], device='cuda')
        dist.all_reduce(values)
        values = values.cpu().tolist()
        maxima = torch.tensor([elapsed, loaded-began], device='cuda')
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        step_max, data_max = maxima.cpu().tolist()
        window_steps += 1
        window_data_max += data_max
        window_step_max += step_max
        record = dict(iteration=it+1, loss=values[0]/world, goal_gradient_l1=values[1], goal_update_l1=values[2],
                      mean_step_seconds=values[3]/world, mean_data_wait_seconds=values[4]/world,
                      action_fm=values[5]/world, video_fm=values[6]/world,
                      lr_min=min(lr_used), lr_max=max(lr_used),
                      allocated_gib=torch.cuda.max_memory_allocated()/2**30, world_size=world)
        if it < 5 or (it+1) % 10 == 0:
            reported = time.monotonic()
            record.update(wall_timestamp=time.time(), window_steps=window_steps,
                          window_wall_seconds=reported-window_started,
                          effective_seconds_per_step=(reported-window_started)/window_steps,
                          window_mean_slowest_rank_step=window_step_max/window_steps,
                          window_mean_slowest_rank_data_wait=window_data_max/window_steps)
            with metrics_file.open('a') as f:
                f.write(json.dumps(record)+'\n')
            if rank == 0:
                print('TRAIN_UPDATE', json.dumps(record), flush=True)
                wandb.log({f'train/{k}': v for k, v in record.items() if k!='iteration'}, step=it+1)
            window_started, window_steps = reported, 0
            window_data_max = window_step_max = 0.
        if it == start+1:
            verify_inference(core, raw, cfg.model.config.vlm_config.tokenizer, out, rank)
        if os.environ.get('VIGAR_POLICY_CHECKPOINT_PROBE') == str(it+1):
            save(it+1)
            write_json(out / f'checkpoint_probe.rank{rank}.json', dict(iteration=it+1, complete=True))
        if not a.preflight and ((it+1) % a.save_every == 0 or it+1 == a.steps):
            save(it+1)
    write_json(out / f'completed.rank{rank}.json', dict(steps=a.steps, complete=True))
    if run:
        run.finish()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
