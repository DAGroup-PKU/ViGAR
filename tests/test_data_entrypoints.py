import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_stage_builders_and_merger_import_without_private_checkouts():
    for name in ['01_builders/build_robotwin_official50_manualgoal.py',
                 '01_builders/build_robotwin_stage_success_v7.py',
                 '04_boundary_extraction/merge_stage_boundaries.py']:
        result=subprocess.run([sys.executable,str(ROOT/'data/stage_v7'/name),'--help'],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert 'usage:' in result.stdout

def test_offline_capture_helpers_have_a_closed_import_chain():
    root=ROOT/'data/stage_v7/policy'
    code='import sys;sys.path.insert(0,sys.argv[1]);from cosmos_policy.subtask_oracle import ExpertSubtaskCapture;from cosmos_policy.stage_success import STAGE_CONTRACT_VERSION;print(STAGE_CONTRACT_VERSION)'
    result=subprocess.run([sys.executable,'-c',code,str(root)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()
