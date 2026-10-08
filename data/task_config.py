"""Shared, versioned task selection for offline annotation and goal generation."""
import hashlib
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).with_suffix('.json')
CONFIG = json.loads(CONFIG_PATH.read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate(config):
    tasks = config['tasks']
    counts = config['dataset']
    if len(tasks) != counts['tasks'] or counts['episodes'] != len(tasks) * counts['episodes_per_task']:
        raise ValueError('Task/episode counts do not match the dataset contract')
    multi = {name for name, row in tasks.items() if row['mode'] == 'subgoal'}
    final = {name for name, row in tasks.items() if row['mode'] == 'episode_goal'}
    if len(multi) != counts['multi_stage_tasks'] or len(final) != counts['final_goal_tasks']:
        raise ValueError('Task modes do not match the declared counts')
    for name, row in tasks.items():
        texts, criteria = row['stage_texts'], row['stage_criteria']
        if name in multi:
            if len(texts) < 2 or len(texts) != len(criteria) or criteria[-1]['name'] != 'official_success':
                raise ValueError(f'Invalid multi-stage task: {name}')
        elif texts or criteria:
            raise ValueError(f'Final-goal task must use its episode prompt: {name}')
    units = sum(len(tasks[name]['stage_texts']) for name in multi) + len(final)
    if units != counts['semantic_stage_units']:
        raise ValueError('Semantic stage count mismatch')


validate(CONFIG)
CONFIG_SHA256 = fingerprint(CONFIG)
SUBGOAL_TASKS = frozenset(name for name, row in CONFIG['tasks'].items() if row['mode'] == 'subgoal')


def require_config(value):
    if value.get('task_config_sha256') != CONFIG_SHA256:
        raise ValueError('Artifact belongs to a different task configuration; rebuild its inputs')


if __name__ == '__main__':
    print(json.dumps(dict(CONFIG['dataset'], version=CONFIG['version'], task_config_sha256=CONFIG_SHA256), indent=2))
