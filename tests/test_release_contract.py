"""Check the selected release binding and the prepared evaluation contract."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_user_selected_combination_is_explicit():
    s=json.loads((ROOT/'configs/selected_release.json').read_text())
    assert s['policy']['step']==50000 and s['policy']['lookahead_ratio']==.15
    assert s['i2i']['step']==70000 and s['i2i']['weights']=='regular'
    assert s['i2i']['next_subgoal_tail_fraction']==0
    assert s['evaluation']['combination_score'] is None

def test_plan_replaces_historical_100k_planner_without_writing(tmp_path):
    spec=importlib.util.spec_from_file_location('prepare_eval',ROOT/'scripts/prepare_eval.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    output=tmp_path/'not_created'
    policy=tmp_path/'policy/iter_000050000';planner=tmp_path/'i2i/iter_000070000'
    p=module.build_plan(output,policy,planner,tmp_path/'RoboTwin')
    assert p['planner']==str(planner) and p['checkpoints']==[str(policy)]
    assert p['goal_refresh_mode']=='sync_current'
    assert p['no_expert_goal'] and p['no_oracle_switch']
    assert p['seed_batch_size']==5 and p['simulators_per_model_pair']==2
    assert len(p['tasks'])==50 and p['episodes_per_checkpoint']==5000
    assert p['planner_release']==str(ROOT/'i2i/inference_runtime')
    assert not output.exists()

def test_frozen_normalizer_and_episode_instructions():
    s=json.loads((ROOT/'configs/selected_release.json').read_text())
    norm=ROOT/'policy/source/recipes/GoalWAM/assets/norm_stats_merged/robotwin_aloha_agilex.json'
    assert hashlib.sha256(norm.read_bytes()).hexdigest()==s['policy']['normalizer_sha256']
    plan=json.loads((ROOT/'configs/eval_plan.json').read_text())
    pairs=ROOT/'configs/paired_episodes.json'
    assert hashlib.sha256(pairs.read_bytes()).hexdigest()==plan['paired_manifest_sha256']
    rows=json.loads(pairs.read_text())
    assert len(rows)==100 and sum(map(len,rows.values()))==5000
    assert all(len(v)==50 for v in rows.values())
