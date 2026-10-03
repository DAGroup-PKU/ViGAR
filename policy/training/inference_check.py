"""One actual generated 48-action chunk through the matching serving processor."""
import numpy as np
import torch
from recipes.GoalWAM.data.images import CAMERA_KEYS
from recipes.GoalWAM.data.dataset import ROBOT_DOMAINS
from recipes.GoalWAM.trainer.evaluator import rng_state, restore_rng, write_json
from recipes.simulation.robotwin.goalwam.policy import ObservationProcessor, reshard_network
from recipes.simulation.robotwin.common.geometry import ACTION_MASK, STATE_MASK, controller_actions
from recipes.GoalWAM.data.relative_action import to_global_action
from paired_dataset import CACHE, goal_views

@torch.no_grad()
def verify_inference(core, cfg, raw, data_config, out, rank):
    from cosmos_framework.utils import misc
    from cosmos_framework.model.vfm.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig
    from recipes.GoalWAM.data.data_loader import collate_samples
    saved = rng_state()
    core.eval()
    x = raw.physical_sample(0)
    c = raw.case(0)
    views = goal_views(CACHE / c['goal_path'], c['goal_sha256'])
    images = {name: raw._camera_frames(x['episode'], key, [x['start']])[0].permute(1,2,0).numpy()
              for name, key in CAMERA_KEYS.items() if name in ('head','left','right')}
    request = dict(state=x['anchor_state'].numpy(), state_valid_mask=STATE_MASK.copy(),
                   action_valid_mask=ACTION_MASK.copy(), images=images,
                   goal_images={k:v[0].permute(1,2,0).numpy() for k,v in views.items()},
                   instruction=c.get('phase_text') or c['prompt'], robot_type='robotwin_aloha_agilex')
    processor = ObservationProcessor(data_config, x['normalizer'], cfg.model.config.vlm_config.tokenizer)
    sample, anchor, mask = processor.prepare(request)
    assert not sample['action'][1:].any()
    sampler = UniPCSampler(cfg=UniPCSamplerConfig(), tensor_kwargs=core.tensor_kwargs)
    try:
        generated = core.generate_samples_from_batch(misc.to(collate_samples([sample]), device='cuda'),
                                                     sampler=sampler, seed=[9000], num_steps=10, guidance=1., shift=2.)
        normalized = generated['action'][0][1:].float().cpu()
        assert normalized.shape == (48,49) and torch.isfinite(normalized).all()
        relative = processor.normalizer.denormalize_action(normalized).masked_fill(~mask,0)
        absolute = to_global_action(anchor, relative, processor.layout).masked_fill(~mask,0)
        commands = controller_actions(absolute.numpy(), mask.numpy(), np.eye(4), 'qpos')
        assert commands.shape == (48,14) and np.isfinite(commands).all()
        write_json(out / f'inference.rank{rank}.json', dict(complete=True, normalized_shape=[48,49],
                   command_shape=[48,14], finite=True, expert_goal_input=False, oracle_stage_switch=False,
                   solver='UniPC', steps=10, guidance=1., shift=2., execution_prefix=32))
    finally:
        reshard_network(core.net)
        core.train()
        restore_rng(saved)
