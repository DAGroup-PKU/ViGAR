import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def build_plan(output, policy, planner, robotwin, *, nodes=1, gpus_per_node=8,
               episodes=100, seed=3, tasks=None, splits=None, simulators_per_pair=1,
               paired_episodes=None, fast_inference=False):
    if nodes < 1 or gpus_per_node < 2 or gpus_per_node % 2:
        raise ValueError('Use at least one node and an even number of GPUs per node, starting at 2')
    if episodes < 1 or seed < 0 or simulators_per_pair < 1:
        raise ValueError('Episodes and simulator count must be positive; seed must be nonnegative')
    plan = json.loads((REPO / 'configs/eval_plan.json').read_text())
    selected = tasks or plan['tasks']
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(plan['tasks']):
        raise ValueError('Select unique tasks from configs/eval_plan.json')
    splits = splits or ['clean', 'random']
    if not splits or len(set(splits)) != len(splits) or set(splits) - {'clean', 'random'}:
        raise ValueError('Splits must be clean and/or random')
    plan.update(source=str(REPO / 'vigar/source'), simroot=str(robotwin),
                checkpoints=[str(policy)], planner=str(planner),
                planner_release=str(REPO / 'subgoal_planner/inference_runtime'),
                planner_service_dir=str(REPO / 'evaluation/planner_service'),
                tasks=selected, splits=splits, nodes=nodes, gpus_per_node=gpus_per_node,
                episodes_per_split=episodes, episodes_per_checkpoint=len(selected)*len(splits)*episodes,
                seed=seed, start_seed=100000*(1+seed), max_seed_attempts=max(10000, episodes*100),
                simulators_per_model_pair=simulators_per_pair, fast_inference=fast_inference)
    if paired_episodes:
        rows = json.loads(Path(paired_episodes).read_text())
        for task in selected:
            for split in splits:
                items = rows.get(f'{split}/{task}', [])
                if len(items) != episodes or len({r['seed'] for r in items}) != episodes:
                    raise ValueError(f'{split}/{task}: expected {episodes} unique seeds')
                if any(type(r['seed']) is not int or r['seed'] < 0 or
                       not isinstance(r.get('instruction'), str) or not r['instruction'].strip() for r in items):
                    raise ValueError(f'{split}/{task}: invalid seed or instruction')
        plan['paired_episode_file'] = str(output / 'episodes.json')
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=os.environ.get('VIGAR_POLICY_CHECKPOINT'))
    parser.add_argument('--planner', type=Path, default=os.environ.get('SUBGOAL_PLANNER_CHECKPOINT'))
    parser.add_argument('--robotwin', type=Path, default=os.environ.get('ROBOTWIN_ROOT'))
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--gpus-per-node', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--splits', nargs='+', choices=['clean', 'random'])
    parser.add_argument('--simulators-per-pair', type=int, default=1)
    parser.add_argument('--paired-episodes', type=Path)
    parser.add_argument('--fast-inference', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not all([args.policy, args.planner, args.robotwin]):
        parser.error('Set policy, planner and robotwin paths')
    output = args.output.resolve()
    plan = build_plan(output, args.policy.resolve(), args.planner.resolve(), args.robotwin.resolve(),
                      nodes=args.nodes, gpus_per_node=args.gpus_per_node, episodes=args.episodes,
                      seed=args.seed, tasks=args.tasks, splits=args.splits,
                      simulators_per_pair=args.simulators_per_pair, paired_episodes=args.paired_episodes,
                      fast_inference=args.fast_inference)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    if output.exists():
        raise FileExistsError('Choose a new output directory; resume an existing run with evaluate.sh')
    for path in [args.policy/'complete.json', args.policy/'recipe_config.json',
                 args.planner/'model/.metadata', args.robotwin/'envs']:
        if not path.exists(): raise FileNotFoundError(path)
    metadata = json.loads((args.planner/'inference_config.json').read_text())
    if metadata.get('weight_variant') != 'regular':
        raise ValueError('The released planner uses regular weights')
    if not any((args.planner/'model').glob('*.distcp')):
        raise ValueError('Planner checkpoint shards are missing')
    output.mkdir(parents=True)
    if args.paired_episodes:
        (output/'episodes.json').write_bytes(args.paired_episodes.read_bytes())
    (output/'PLAN.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(output/'PLAN.json')


if __name__ == '__main__':
    main()
