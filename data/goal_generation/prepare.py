"""Prepare goal-generation cases directly from the final annotated dataset."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image

from common import CONFIG, CONFIG_SHA256, require_config, sha, write


def canvas(handle, frame):
    views = []
    for camera in ('head_camera', 'left_camera', 'right_camera'):
        raw = np.asarray(handle[f'observation/{camera}/rgb'][frame])
        if raw.ndim != 1 and raw.ndim != 0:
            raise ValueError('Expected official RoboTwin encoded RGB; unencoded input needs an explicit channel convention')
        rgb = cv2.imdecode(np.frombuffer(bytes(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f'Cannot decode {camera} frame {frame}')
        # Official RoboTwin passes RGB directly to imencode: do not flip channels.
        views.append(rgb)
    head, left, right = views
    height, width = head.shape[:2]
    bottom = np.concatenate([cv2.resize(x, (width // 2, height // 2)) for x in (left, right)], axis=1)
    return cv2.resize(np.concatenate([head, bottom]), (320, 384))


def stage_cases(annotations, candidates=1):
    if candidates < 1:
        raise ValueError('At least one candidate is required')
    cases = []
    for eid, row in sorted(annotations.items(), key=lambda item: int(item[0])):
        require_config(row)
        family = row['family']
        spec = CONFIG['tasks'][family]
        steps = row['action_steps']
        count = len(steps)
        expected = len(spec['stage_texts']) if spec['mode'] == 'subgoal' else 1
        if count != expected and not (spec['dynamic_stage_count'] and count == 1):
            raise ValueError(f'Unexpected stage count for {family}')
        previous = 0
        for index, step in enumerate(steps):
            start, end = int(step['start_frame']), int(step['end_frame'])
            if start != previous or not start < end <= row['frame_count']:
                raise ValueError(f'Invalid stage interval for episode {eid}')
            previous = end
            key = f'episode_{int(eid):06d}_stage_{index:02d}'
            seeds = [int.from_bytes(hashlib.sha256(f'{key}:{i}'.encode()).digest()[:4], 'big') & 0x7fffffff
                     for i in range(candidates)]
            cases.append(dict(key=key, task=family, episode_index=int(eid), subtask_index=index,
                              phase=index, goal_policy=spec['mode'], prompt=row['task_name'],
                              phase_text=step['action_text'], split='clean', input_frame=start,
                              target_frame=end - 1, end_frame_exclusive=end,
                              candidates=[dict(id=f'C{i+1}', variant='generated', path=f'C{i+1}.png', seed=seed)
                                          for i, seed in enumerate(seeds)],
                              scope='offline training goal review', task_config_sha256=CONFIG_SHA256))
        if previous != row['frame_count']:
            raise ValueError(f'Incomplete episode annotation: {eid}')
    return cases


def prepare(dataset, output, checkpoint, candidates=1):
    runtime = json.loads((dataset / 'runtime.meta.json').read_text())
    require_config(runtime)
    annotations = json.loads((dataset / 'meta/annotations.json').read_text())
    cases = stage_cases(annotations, candidates)
    if len(annotations) != CONFIG['dataset']['episodes']:
        raise ValueError('Incomplete annotated dataset')
    metadata = checkpoint / 'model/.metadata'
    metadata_hash = sha(metadata)
    output.mkdir(parents=True, exist_ok=False)
    for row in annotations.values():
        with h5py.File(row['source_hdf5'], 'r') as handle:
            if len(handle['joint_action/vector']) != row['frame_count']:
                raise ValueError('Raw HDF length differs from the annotation')
            episode_cases = [case for case in cases if case['episode_index'] == row['episode_index']]
            for case in episode_cases:
                folder = output / case['key']
                folder.mkdir()
                for filename, frame in [('input_rgb.png', case['input_frame']), ('gt_rgb.png', case['target_frame'])]:
                    Image.fromarray(canvas(handle, frame)).save(folder / filename)
                    case[filename + '_sha256'] = sha(folder / filename)
    write(output / 'cases.json', cases)
    write(output / 'contract.json', dict(schema='vigar-goal-generation/v1', task_config_sha256=CONFIG_SHA256,
          dataset=str(dataset.resolve()), annotations_sha256=sha(dataset / 'meta/annotations.json'),
          checkpoint=str(checkpoint.resolve()), checkpoint_metadata_sha256=metadata_hash,
          weight_variant='regular', split='clean', episodes=len(annotations), slots=len(cases),
          input_frame_policy='stage start', prompt_policy='episode task prompt', gt_sent_to_i2i=False))
    write(output / 'PREPARED.json', dict(contract_sha256=sha(output / 'contract.json'), cases_sha256=sha(output / 'cases.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--candidates', type=int, default=1)
    args = parser.parse_args()
    prepare(args.dataset, args.output, args.checkpoint, args.candidates)
