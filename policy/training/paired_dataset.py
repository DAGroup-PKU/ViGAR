"""Selected three-view policy: 15% next-stage redirect and shared ColorJitter."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from color_jitter import augment_sample, CONFIG as JITTER_CONFIG
import torch.nn.functional as F
from PIL import Image

from recipes.GoalWAM.data.dataset import LeRobot0824Dataset
from recipes.GoalWAM.data.images import compose_goal_image
from lookahead_contract import policy_case

SOURCE = Path(__file__).resolve().parents[1] / 'source'
ROOT = Path(os.environ['GOALWAM_POLICY_WORKSPACE'])
CACHE = Path(os.environ['GOALWAM_GENERATED_GOAL_CACHE'])
NORMALIZER = SOURCE / 'recipes/GoalWAM/assets/norm_stats_merged/robotwin_aloha_agilex.json'

def goal_views(path, expected_hash):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f'Goal checksum mismatch: {path.name}')
    image = np.asarray(Image.open(path).convert('RGB')).copy()
    if image.shape != (384, 320, 3):
        raise ValueError(f'Unexpected source goal layout {image.shape}')
    t = torch.from_numpy(image).permute(2, 0, 1)[None]
    tiles = {'head': t[:, :, :256, :], 'left': t[:, :, 256:, :160], 'right': t[:, :, 256:, 160:]}
    # The old canvas stretched 240x320 cameras into 256x320 / 128x160 cells.
    return {k: F.interpolate(v.float(), size=(240, 320), mode='bilinear', align_corners=False)
            .round().clamp(0, 255).byte() for k, v in tiles.items()}

class PairedDataset(LeRobot0824Dataset):
    def __init__(self, goal_mode):
        if goal_mode != 'multi_view':
            raise ValueError(goal_mode)
        self.paired_goal_mode = goal_mode
        manifest = json.loads((CACHE / 'manifest.json').read_text())
        assert manifest['complete'] and manifest['episodes'] == 2500 and manifest['slots'] == 3691
        self.cases = {}
        for c in manifest['cases']:
            self.cases.setdefault(int(c['episode_index']), []).append(c)
        assert len(self.cases) == 2500
        for cases in self.cases.values():
            cases.sort(key=lambda c: c['subtask_index'])
        self.goal_manifest_hash = hashlib.sha256((CACHE / 'manifest.json').read_bytes()).hexdigest()
        super().__init__(os.environ['GOALWAM_DATASET_MANIFEST'],
                         {'robotwin_aloha_agilex': str(NORMALIZER)}, split='all', training=True,
                         include_tail_windows=True, img_size=[224, 288],
                         enable_cameras=['head', 'left', 'right'], img_size_buckets=[],
                         norm_type='bounds_99_woclip', goal_image_composition=goal_mode,
                         supervise_head_eef=False,
                         parquet_cache_dir=os.environ.get('POLICY8_PARQUET_CACHE'))

    def case(self, index):
        position = index[0] if isinstance(index, tuple) else index
        ep, start = self.locate(position)
        cases = self.cases[int(ep['metadata']['episode_index'])]
        return policy_case(cases,start,ratio=.15)

    def physical_sample(self, index):
        x = super().physical_sample(index)
        goal = int(self.case(index)['target_frame'])
        # Stored command k targets measured row k+1. No commands beyond the stage goal.
        temporal = x['action_indices'] < goal
        x['action_time_valid'] &= temporal
        x['action_valid_mask'][1:] &= temporal[:, None]
        x['relative_valid'] &= temporal[:, None]
        x['action'][1:].masked_fill_(~x['action_valid_mask'][1:], 0)
        x['relative_actions'].masked_fill_(~x['relative_valid'], 0)
        x['absolute_actions'].masked_fill_(~x['relative_valid'], 0)
        x['video_indices'] = x['start'] + torch.arange(1 + 4 * min(3, max(0, goal-x['start']) // 16)) * 4
        x['goal_index'] = goal
        return x

    def __getitem__(self, index):
        x = super().__getitem__(index)
        c = self.case(index)
        goal, mask, boxes = compose_goal_image(
            goal_views(CACHE / c['goal_path'], c['goal_sha256']),
            [224, 288], self.enable_cameras, self.paired_goal_mode)
        x.update(goal_frame=goal.permute(1, 0, 2, 3).contiguous(),
                 goal_pixel_mask=mask, goal_camera_boxes=boxes,
                 ai_caption=c.get('phase_text') or c['prompt'], goal_cache_key=c['key'])
        if not hasattr(self, '_jitter_rng'):
            self._jitter_rng = torch.Generator().manual_seed(torch.initial_seed() ^ 0x4a4954544552)
        return augment_sample(x, self._jitter_rng)

    def selection_record(self):
        return dict(super().selection_record(), generated_goal_sha256=self.goal_manifest_hash,
                    goal_ablation=self.paired_goal_mode, original_images_unchanged=True,
                    photometric_augmentation=JITTER_CONFIG, training_pixels_augmented=True, next_state_conversion=True, unknown_final_command_masked=True,lookahead_ratio=.15)
