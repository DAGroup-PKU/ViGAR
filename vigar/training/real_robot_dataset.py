"""Dataset integration for the real_robot_900hr policy (AgiBot A2).

34-D absolute actions (two 7-joint arms, two 10-joint hands), a precomputed
320x384 three-camera canvas, and the recorded end frame of the current subtask
as the visual goal. Windows start on every annotated frame and may cross
subtask boundaries; a window starting in the final 15% of a subtask targets
the next subtask's end frame.
"""
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.action_normalization import load_action_stats, normalize_action
from recipes.ViGAR.data.dataset import FrameReader, ROBOT_DOMAINS
from recipes.ViGAR.data.parquet_cache import ParquetEpisodeCache

SOURCE = Path(__file__).resolve().parents[1] / 'source'
ROOT = Path(os.environ['VIGAR_POLICY_WORKSPACE'])
CANVAS = 'observation.images.concat_view_320x384'
RESOLUTION = '320x384'
ACTION_DIM = 34
# Appended to the caption after the viewpoint text, as in the original recipe.
VIEW_DESCRIPTION = ('The top row is from the head camera looking at the dual-arm robot and workspace. '
                    'The bottom row contains two horizontally concatenated wrist-camera views, '
                    'left-arm wrist on the left and right-arm wrist on the right.')
# Stamp-card handover turns the waist, which the 34-D layout cannot command.
# Task id -> expected episode count.
EXCLUDED_TASKS = {9734: 589}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RealRobotDataset(Dataset):
    def __init__(self, root, *, chunk=128, video_stride=8, lookahead_ratio=.15,
                 excluded_tasks=EXCLUDED_TASKS, parquet_cache_dir=None):
        self.root = Path(root)
        info = json.loads((self.root / 'meta/info.json').read_text())
        if info['codebase_version'] != 'v3.0':
            raise ValueError(f'{self.root}: only LeRobot v3.0 is supported')
        for key in ('observation.state', 'action'):
            if info['features'][key]['shape'] != [ACTION_DIM]:
                raise ValueError(f'{self.root}: {key} must have {ACTION_DIM} dimensions')
        if info['features'].get(CANVAS, {}).get('shape', [])[:2] != [320, 384]:
            raise ValueError(f'{self.root}: missing 320x384 camera canvas {CANVAS}')
        if chunk % video_stride or not 0 <= lookahead_ratio < 1:
            raise ValueError('chunk must be divisible by video_stride; lookahead_ratio must be in [0, 1)')
        if not parquet_cache_dir:
            raise ValueError('Set VIGAR_PARQUET_CACHE: random episode reads need the Arrow row cache')
        self.info, self.fps = info, float(info['fps'])
        self.chunk, self.video_stride, self.lookahead_ratio = chunk, video_stride, lookahead_ratio
        self.stats_path = self.root / 'action_stats.json'
        self.action_stats, self.state_stats = (
            {k: torch.from_numpy(v).float() for k, v in load_action_stats(str(self.stats_path), stats_key=key).items()}
            for key in ('absolute_action', 'robot_state'))
        for stats in (self.action_stats, self.state_stats):
            if any(stats[k].shape != (ACTION_DIM,) for k in ('mean', 'std')):
                raise ValueError(f'{self.stats_path}: expected {ACTION_DIM}-D mean/std')

        keep = ('episode_index', 'length', 'tasks', 'data/chunk_index', 'data/file_index',
                f'videos/{CANVAS}/chunk_index', f'videos/{CANVAS}/file_index', f'videos/{CANVAS}/from_timestamp')
        episodes = {}
        for path in sorted((self.root / 'meta/episodes').glob('chunk-*/*.parquet')):
            for row in pq.read_table(path, columns=list(keep)).to_pylist():
                episodes[int(row['episode_index'])] = row
        annotations_path = self.root / 'meta/annotations.json'
        self.annotations_sha256 = digest(annotations_path)
        raw = json.loads(annotations_path.read_text())
        annotations = {int(a['episode_index']): a for a in (raw.values() if isinstance(raw, dict) else raw)}
        if set(annotations) != set(episodes):
            raise ValueError('meta/annotations.json must cover exactly the dataset episodes')

        self.episodes, self.captions = [], []
        excluded = dict.fromkeys(excluded_tasks, 0)
        segments = []
        for eid in sorted(episodes):
            ep, annotation = episodes[eid], annotations[eid]
            task_id = int(annotation['task_id'])
            if task_id in excluded:
                excluded[task_id] += 1
                continue
            caption = str(annotation['task_name']).strip()
            if not caption or caption not in ep['tasks']:
                raise ValueError(f'episode {eid}: annotation task_name disagrees with meta/episodes')
            length, steps = int(ep['length']), annotation['action_steps']
            if int(annotation['frame_count']) != length or not steps:
                raise ValueError(f'episode {eid}: frame_count/action_steps disagree with the episode')
            slot, previous = len(self.episodes), None
            for i, step in enumerate(steps):
                start, end = int(step['start_frame']), int(step['end_frame'])
                if not 0 <= start < end <= length or (previous is not None and start != previous):
                    raise ValueError(f'episode {eid}: subtask {i} must continue at frame {previous}')
                segments.append((slot, i, start, end))
                previous = end
            if previous != length:
                raise ValueError(f'episode {eid}: subtasks must end at the episode boundary')
            self.episodes.append(ep)
            self.captions.append(caption)
        for task_id, count in excluded.items():
            if count != excluded_tasks[task_id]:
                raise ValueError(f'task {task_id}: expected {excluded_tasks[task_id]} episodes, found {count}')
        self.excluded = excluded
        self.segments = np.asarray(segments, dtype=np.int64)
        self.stops = np.cumsum(self.segments[:, 3] - self.segments[:, 2])
        self._rows, self._readers = OrderedDict(), OrderedDict()
        self._parquet_cache = ParquetEpisodeCache(parquet_cache_dir)

    def __len__(self):
        return int(self.stops[-1])

    def window(self, index):
        """Episode slot, window start and goal frame of a flat window index."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        s = int(np.searchsorted(self.stops, index, side='right'))
        slot, subtask, start, end = self.segments[s].tolist()
        frame = start + index - (int(self.stops[s - 1]) if s else 0)
        goal = s
        tail = int(self.lookahead_ratio * (end - start))
        if tail > 0 and frame >= end - tail and s + 1 < len(self.segments):
            following = self.segments[s + 1]
            if following[0] == slot and following[1] == subtask + 1:
                goal = s + 1
        return slot, frame, int(self.segments[goal, 3]) - 1

    def _episode_rows(self, slot):
        if slot not in self._rows:
            ep = self.episodes[slot]
            table = self._parquet_cache.read_episode(self.root, self.info, ep).sort_by('frame_index')
            if table['frame_index'].to_pylist() != list(range(ep['length'])):
                raise ValueError(f"episode {ep['episode_index']}: noncontiguous rows")
            self._rows[slot] = {
                key: pc.list_flatten(table[key].combine_chunks()).to_numpy(zero_copy_only=False)
                .reshape(-1, ACTION_DIM).astype(np.float32) for key in ('observation.state', 'action')}
            while len(self._rows) > 2:
                self._rows.popitem(last=False)
        self._rows.move_to_end(slot)
        return self._rows[slot]

    def _frames(self, slot, indices):
        ep = self.episodes[slot]
        path = self.root / self.info['video_path'].format(
            video_key=CANVAS, chunk_index=ep[f'videos/{CANVAS}/chunk_index'],
            file_index=ep[f'videos/{CANVAS}/file_index'])
        key = str(path)
        if key not in self._readers:
            self._readers[key] = FrameReader(path, self.fps)
            while len(self._readers) > 6:
                self._readers.popitem(last=False)[1].close()
        self._readers.move_to_end(key)
        offset = ep[f'videos/{CANVAS}/from_timestamp']
        frames = self._readers[key].read([offset + i / self.fps for i in indices])
        if frames.shape[1:] != (320, 384, 3):
            raise ValueError(f"episode {ep['episode_index']}: decoded canvas is not 320x384")
        return torch.from_numpy(frames).permute(3, 0, 1, 2)

    def __getitem__(self, index):
        slot, start, goal = self.window(index)
        length = int(self.episodes[slot]['length'])
        # Row 0 is the observed state; rows 1..chunk are commands, repeating the
        # final row past the episode end.
        indices = [min(start + i, length - 1) for i in range(self.chunk + 1)]
        rows = self._episode_rows(slot)
        state = torch.from_numpy(rows['observation.state'][start])
        actions = torch.from_numpy(rows['action'][indices[1:]])
        action = torch.cat([normalize_action(state[None], 'meanstd', self.state_stats),
                            normalize_action(actions, 'meanstd', self.action_stats)])
        frames = self._frames(slot, indices[::self.video_stride] + [goal])
        return dict(ai_caption=self.captions[slot], video=frames[:, :-1], goal_frame=frames[:, -1:],
                    action=action, mode='policy', viewpoint='concat_view', additional_view_description=VIEW_DESCRIPTION,
                    conditioning_fps=torch.tensor(self.fps / self.video_stride, dtype=torch.float32),
                    action_fps=torch.tensor(self.fps, dtype=torch.float32),
                    domain_id=torch.tensor(ROBOT_DOMAINS['agibot'], dtype=torch.long))

    def selection_record(self):
        return dict(dataset=str(self.root), annotations_sha256=self.annotations_sha256,
                    stats_sha256=digest(self.stats_path), episodes=len(self.episodes),
                    subtasks=len(self.segments), windows=len(self), excluded_tasks=self.excluded,
                    chunk=self.chunk, video_stride=self.video_stride, lookahead_ratio=self.lookahead_ratio,
                    action_dim=ACTION_DIM, action_normalization='meanstd', state_normalization='meanstd',
                    canvas=CANVAS, caption='annotation task_name')

    def close(self):
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        self._parquet_cache.close()


class RealRobotSFTDataset(Dataset):
    def __init__(self, dataset, *, tokenizer_config, cfg_dropout_rate=.1, max_action_dim=64):
        from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

        self._dataset, self.cfg_dropout_rate = dataset, cfg_dropout_rate
        self._transform = ActionTransformPipeline(
            tokenizer_config=tokenizer_config, cfg_dropout_rate=cfg_dropout_rate, max_action_dim=max_action_dim,
            action_video_downsample_factor=dataset.video_stride, goal_frame_key='goal_frame',
            goal_frame_injection='generator', goal_layout='concat_320x384')

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._transform(self._dataset[index], RESOLUTION)
