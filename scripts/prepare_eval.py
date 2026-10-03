"""Create a new Strict Sync evaluation plan; never submit jobs or run a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def build_plan(output, policy, planner, robotwin):
    plan = json.loads((REPO/'configs/eval_plan.json').read_text())
    for key in ['migration_from', 'backend_cutover', 'followup_priority', 'i2i_vendor_by_checkpoint']:
        plan.pop(key, None)
    plan.update(source=str(REPO/'policy/source'), simroot=str(robotwin),
                checkpoints=[str(policy)], planner=str(planner),
                planner_release=str(REPO/'i2i/inference_runtime'),
                i2i_vendor_dir=str(REPO/'evaluation/vendor_speedup_lru'),
                paired_episode_file=str(REPO/'configs/paired_episodes.json'),
                name='goalwam-policy-la015-cj50-i2i-roi70',
                combination_evaluation_status='not_run_for_this_release',
                seed_batch_size=5, simulators_per_model_pair=2,
                goal_refresh_mode='sync_current', no_expert_goal=True, no_oracle_switch=True)
    return plan

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy', type=Path, default=os.environ.get('GOALWAM_POLICY_CHECKPOINT'))
    p.add_argument('--planner', type=Path, default=os.environ.get('GOALWAM_I2I_CHECKPOINT'))
    p.add_argument('--robotwin', type=Path, default=os.environ.get('ROBOTWIN_ROOT'))
    p.add_argument('--dry-run', action='store_true')
    a=p.parse_args()
    if not all([a.policy,a.planner,a.robotwin]): p.error('Set policy, planner and robotwin paths')
    plan=build_plan(a.output.resolve(),a.policy.resolve(),a.planner.resolve(),a.robotwin.resolve())
    if a.dry_run:
        print(json.dumps(plan,indent=2));return
    if a.output.exists(): raise FileExistsError('Choose a fresh output directory; existing runs are not overwritten')
    for path in [a.policy/'complete.json',a.policy/'recipe_config.json',a.planner/'model/.metadata',a.robotwin/'envs']:
        if not path.exists(): raise FileNotFoundError(path)
    recipe=json.loads((a.policy/'recipe_config.json').read_text())
    if recipe['data'].get('lookahead_ratio') != .15 or not recipe['data'].get('photometric_augmentation'):
        raise ValueError('Selected policy must carry 15% lookahead and ColorJitter')
    if a.planner.name != 'iter_000070000': raise ValueError('Selected planner directory must be iter_000070000; verify its provenance separately')
    paired=Path(plan['paired_episode_file'])
    if hashlib.sha256(paired.read_bytes()).hexdigest()!=plan['paired_manifest_sha256']:
        raise ValueError('Paired episode manifest checksum mismatch')
    a.output.mkdir(parents=True)
    (a.output/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(a.output/'PLAN.json')

if __name__=='__main__': main()
