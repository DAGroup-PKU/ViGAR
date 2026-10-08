"""Run one preparation step using data/task_config.json as the shared task entry point."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from task_config import CONFIG, CONFIG_SHA256

ROOT = Path(__file__).resolve().parent
STEPS = {
    'extract': 'stage_annotation/boundary_extraction/extract_stage_boundaries.py',
    'merge': 'stage_annotation/boundary_extraction/merge_stage_boundaries.py',
    'annotate': 'stage_annotation/build_dataset.py',
    'prepare-goals': 'goal_generation/prepare.py',
    'generate': 'goal_generation/generate.py',
    'review': 'goal_generation/curate.py',
    'export': 'goal_generation/export_cache.py',
    'verify': 'goal_generation/verify_cache.py',
    'recover-actions': 'action_conversion/restore_policy_metadata.py',
    'convert-actions': 'action_conversion/build_dataset49.py',
}


def main():
    if len(sys.argv) == 1 or sys.argv[1] == 'summary':
        print(json.dumps(dict(CONFIG['dataset'], task_config=str(ROOT / 'task_config.json'),
                              task_config_sha256=CONFIG_SHA256), indent=2))
        return 0
    # Forward step-specific --help to the real tool without importing its runtime.
    if sys.argv[1] in STEPS:
        return subprocess.call([sys.executable, str(ROOT / STEPS[sys.argv[1]]), *sys.argv[2:]])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['summary', *STEPS])
    parser.parse_args()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
