import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_public_plan_has_no_deployment_paths():
    plan=json.loads((ROOT/'configs/eval_plan.json').read_text())
    assert not {'source','simroot','checkpoints','planner','planner_release','paired_episode_file'} & set(plan)
    selection=json.loads((ROOT/'configs/selected_release.json').read_text())
    assert selection['weights_repository']=='DAGroup-PKU/ViGAR'
    assert 'source_checkpoint' not in selection['policy']
    assert 'source_checkpoint' not in selection['i2i']
    assert not (ROOT/'provenance').exists()

def test_weight_verifier_rejects_path_traversal(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('verify_weights',ROOT/'scripts/verify_weights.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    (tmp_path/'SHA256SUMS.json').write_text(json.dumps([{'path':'../outside','size':0,'sha256':'0'*64}]))
    monkeypatch.setattr('sys.argv',['verify_weights',str(tmp_path)])
    import pytest
    with pytest.raises(ValueError,match='leaves the download directory'):mod.main()
