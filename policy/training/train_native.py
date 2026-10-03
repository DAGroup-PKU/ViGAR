"""Native Cosmos training lifecycle for the matched 49-D generated-goal experiment.

No VeOmni trainer or alternate network/loss/optimizer. Recipe-owned dataset,
checkpoint and inference helpers retain their tested contract.
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('COSMOS_TRAINING', '1')

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
    from recipes.GoalWAM.data.dataset import LeRobot0824SFTDataset
    from recipes.GoalWAM.data.data_loader import StatefulWindowLoader
    from recipes.GoalWAM.trainer.evaluator import save_checkpoint, seed_all, write_json
    from recipes.GoalWAM.trainer.callbacks import UpdateAudit
    from recipes.GoalWAM.trainer.checkpoint_bundle import copy_assets, finish_bundle, save_portable_config
    from veomni.models.transformers.goalwam.configuration_goalwam import GoalWAMConfig
    from paired_dataset import PairedDataset, ROOT, SOURCE, NORMALIZER

    p = argparse.ArgumentParser()
    p.add_argument('--goal-mode', choices=['multi_view'], default='multi_view')
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--save-every', type=int, default=4000)
    p.add_argument('--resume', type=Path)
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', device_id=torch.device('cuda', int(os.environ['LOCAL_RANK'])))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert (world == 8 and a.batch == 1 and a.steps == 2) if a.preflight else (world == 16 and a.batch * world == 256)
    run_suffix = os.environ.get('POLICY8_RUN_SUFFIX', '')
    run_name = os.environ.get('POLICY8_RUN_NAME', 'robotwin_c2r')
    out = ROOT / 'runs' / (('preflight_' if a.preflight else '') + run_name + run_suffix)
    out.mkdir(parents=True, exist_ok=True)
    _init_log_console()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed_all(42)
    model_config = GoalWAMConfig.from_json_file(SOURCE / 'recipes/GoalWAM/configs/migration/initialization_config.json')
    base = Path(os.environ['BASE_CHECKPOINT_PATH'])
    vae = Path(os.environ['WAN_VAE_PATH'])
    tokenizer = Path(os.environ['QWEN_TOKENIZER_PATH'])
    cfg = model_config.runtime_config(base_checkpoint=base, vae_path=vae, tokenizer_path=tokenizer, shard_degree=world)
    cfg.model.config.tokenizer.encode_exact_durations = [1, 5, 9, 13]
    cfg.model.config.compile.enabled = False
    cfg.model.config.activation_checkpointing.mode = 'full'
    cfg.model.config.log_enc_time_every_n = 0
    cfg.optimizer.lr = 1e-5
    cfg.optimizer.disable_weight_decay_for_1d_params = True
    cfg.optimizer.lr_multipliers = {'action2llm': 10., 'llm2action': 10., 'action_modality_embed': 10., 'goal_vision_embed': 5.}
    cfg.scheduler.f_max = [1.]
    cfg.scheduler.f_min = [0.]
    cfg.scheduler.f_start = [0.]
    cfg.scheduler.warm_up_steps = [0]
    cfg.scheduler.cycle_lengths = [100000]
    cfg.trainer.max_iter = a.steps
    cfg.job.project = 'vigar_robotwin'
    cfg.job.group = 'vigar_robotwin'
    cfg.job.name = os.environ.get('WANDB_NAME', 'robotwin_c2r') if not a.preflight else 'goalwam_policy_preflight'
    cfg.job.wandb_mode = 'online'
    cfg.job.path_local = str(out)
    signal = torch.tensor(float(rank), device='cuda')
    dist.all_reduce(signal)
    assert signal.item() == world*(world-1)/2
    write_json(out / f'nccl.rank{rank}.json', dict(world=world, sum=signal.item(), host=os.uname().nodename))

    core = OmniMoTModel(cfg.model.config)
    core.on_train_start(torch.preserve_format)
    optim, scheduler = core.init_optimizer_scheduler(cfg.optimizer, cfg.scheduler)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    raw = PairedDataset(a.goal_mode)
    dataset = LeRobot0824SFTDataset(raw, tokenizer_config=cfg.model.config.vlm_config.tokenizer, cfg_dropout_rate=.1)
    loader = StatefulWindowLoader(dataset, rank=rank, world_size=world, seed=42,
                                  batch_size=a.batch, num_workers=4, prefetch_factor=2)
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
        from recipes.GoalWAM.trainer.evaluator import load_checkpoint
        start = load_checkpoint(core, optim, scheduler, loader, a.resume, resume=True)
    recipe = dict(data=dict(norm_type='bounds_99_woclip', img_size=[224, 288], img_size_buckets=[],
                           enable_cameras=['head', 'left', 'right'], goal_image_composition=a.goal_mode,
                           supervise_head_eef=False, supervise_arm_head_torso=True,
                           state_arm_eef_coordinate='head_camera', include_robot_type_text_context=False,
                           norm_stat_files={'robotwin_aloha_agilex': str(NORMALIZER)},
                           include_tail_windows=True, text_conditioning='episode',
                           caption_source='generated_manifest.phase_text_or_prompt',lookahead_ratio=.15),
                  train=dict(gen_num_steps=10, gen_guidance=1., gen_shift=2., gen_seed=9000,
                             micro_batch_size=a.batch, global_batch_size=a.batch*world, max_steps=a.steps,
                             checkpoint_every=a.save_every, seed=42), image_preprocessing_version=2)
    from color_jitter import CONFIG as JITTER_CONFIG
    recipe['data']['photometric_augmentation'] = JITTER_CONFIG
    run = None
    if rank == 0:
        run = wandb.init(project=cfg.job.project, entity=os.environ.get('WANDB_ENTITY'), name=cfg.job.name,
                         dir=str(out), mode='online', config=recipe)
        assert run is not None and run.settings.mode == 'online'
        write_json(out / 'wandb.json', dict(id=run.id, url=run.url, mode='online'))
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
        from checkpoint_directory import reserve_checkpoint_directory
        path = out / 'checkpoints' / f'iter_{step:09d}'
        reserve_checkpoint_directory(path, dist)
        save_checkpoint(core, optim, scheduler, loader, step, path, mark_complete=False)
        if rank == 0:
            copy_assets(path, vae=vae, tokenizer=tokenizer,
                        backbone_config=cfg.model.config.vlm_config.model_instance.config.base_config.json_file)
            model_config.cosmos = OmegaConf.to_container(cfg, resolve=True)
            save_portable_config(model_config, path)
            write_json(path / 'recipe_config.json', recipe)
            write_json(path / 'data_contract.json', raw.selection_record())
            finish_bundle(path, kind='training', iteration=step, world_size=world,
                          provenance=dict(base=str(base), dataset='generated2500', native_trainer=True))
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
            from inference_check import verify_inference
            verify_inference(core, cfg, raw, recipe['data'], out, rank)
        if os.environ.get('POLICY8_CHECKPOINT_PROBE') == str(it+1):
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
