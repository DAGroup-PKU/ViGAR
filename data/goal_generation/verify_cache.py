"""Check generated-goal coverage, annotation binding, RGB shape and hashes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from PIL import Image

from common import CONFIG, CONFIG_SHA256, inside, require_config, sha, write


def verify(cache, dataset):
    manifest = json.loads((cache / 'manifest.json').read_text())
    require_config(manifest)
    frozen = json.loads((cache / 'FROZEN.json').read_text())
    if sha(cache / 'manifest.json') != frozen['manifest_sha256'] or not manifest['complete'] or manifest['provisional']:
        raise ValueError('Cache manifest is incomplete or changed')
    if sha(dataset / 'meta/annotations.json') != manifest['annotations_sha256']:
        raise ValueError('Cache was prepared for different annotations')
    annotations = json.loads((dataset / 'meta/annotations.json').read_text())
    expected = {(int(eid), index): (row, step) for eid, row in annotations.items()
                for index, step in enumerate(row['action_steps'])}
    seen, keys, tasks = set(), set(), defaultdict(set)
    for case in manifest['cases']:
        require_config(case)
        pair = (case['episode_index'], case['subtask_index'])
        if pair in seen or case['key'] in keys or pair not in expected:
            raise ValueError('Duplicate or unknown goal stage')
        seen.add(pair); keys.add(case['key'])
        row, step = expected[pair]
        require_config(row)
        if (case['task'] != row['family'] or case['target_frame'] != step['end_frame'] - 1
                or case['end_frame_exclusive'] != step['end_frame']):
            raise ValueError('Goal does not match the annotated stage')
        goal = inside(cache, case['goal_path'])
        if sha(goal) != case['goal_sha256']:
            raise ValueError('Goal image checksum mismatch')
        with Image.open(goal) as image:
            if image.format != 'PNG' or image.mode != 'RGB' or image.size != (320, 384):
                raise ValueError('Expected a 320x384 RGB PNG goal')
            image.load()
        if case.get('human_approved') is not True:
            raise ValueError('Explicit selection approval is missing')
        tasks[case['task']].add(case['episode_index'])
    if seen != set(expected) or manifest['slots'] != len(seen) or frozen['slots'] != len(seen):
        raise ValueError('Goal cache does not cover every annotated stage')
    if set(tasks) != set(CONFIG['tasks']) or any(len(episodes) != CONFIG['dataset']['episodes_per_task'] for episodes in tasks.values()):
        raise ValueError('Goal cache must cover all 50 tasks and their 50 episodes')
    if manifest['episodes'] != CONFIG['dataset']['episodes'] or manifest['tasks'] != CONFIG['dataset']['tasks']:
        raise ValueError('Cache dataset count mismatch')
    receipt = dict(passed=True, manifest_sha256=sha(cache / 'manifest.json'),
                   task_config_sha256=CONFIG_SHA256, episodes=manifest['episodes'], slots=len(seen),
                   all_goal_sha_verified=True, policy_success_measured=False)
    write(cache / 'VERIFIED.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.cache, args.dataset), indent=2))
