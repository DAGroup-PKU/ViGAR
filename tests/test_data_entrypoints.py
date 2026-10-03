import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_data_tools_expose_help_without_assets_or_private_checkouts():
    for name in ['pipeline.py', 'stage_annotation/build_dataset.py',
                 'stage_annotation/boundary_extraction/extract_stage_boundaries.py',
                 'stage_annotation/boundary_extraction/merge_stage_boundaries.py',
                 'goal_generation/prepare.py', 'goal_generation/generate.py',
                 'goal_generation/curate.py', 'goal_generation/export_cache.py',
                 'goal_generation/verify_cache.py']:
        result = subprocess.run([sys.executable, str(ROOT / 'data' / name), '--help'], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert 'usage:' in result.stdout.lower()


def test_offline_capture_helpers_have_a_closed_import_chain():
    root = ROOT / 'data/stage_annotation/policy'
    code = 'import sys;sys.path.insert(0,sys.argv[1]);from cosmos_policy.subtask_oracle import ExpertSubtaskCapture;from cosmos_policy.stage_success import SUBGOAL_TASKS;print(len(SUBGOAL_TASKS))'
    result = subprocess.run([sys.executable, '-c', code, str(root)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '19'
