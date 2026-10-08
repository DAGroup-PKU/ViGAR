"""Export one explicitly selected generated image per annotated stage."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

from common import CONFIG_SHA256, inside, load_prepared, sha, write
from verify_cache import verify


def export(root, decisions_path, output, dataset):
    contract, cases = load_prepared(root)
    if sha(dataset / 'meta/annotations.json') != contract['annotations_sha256']:
        raise ValueError('Dataset annotations changed after case preparation')
    decisions = json.loads(decisions_path.read_text())
    if decisions['cases_sha256'] != sha(root / 'cases.json'):
        raise ValueError('Selections refer to different prepared cases')
    selected = {row['key']: row for row in decisions['decisions']}
    if len(selected) != len(decisions['decisions']) or set(selected) != {case['key'] for case in cases}:
        raise ValueError('Each prepared stage must have exactly one explicit selection')
    rows, files = [], []
    for case in cases:
        folder = inside(root, case['key'])
        generated = json.loads((folder / 'generated.json').read_text())
        if generated['contract_sha256'] != sha(root / 'contract.json'):
            raise ValueError('Selected image belongs to a different generation contract')
        decision = selected[case['key']]
        matches = [candidate for candidate in generated['case']['candidates'] if candidate['id'] == decision['decision']]
        if len(matches) != 1:
            raise ValueError(f'Choose one generated candidate for {case["key"]}; unreviewed/rejected stages cannot be exported')
        candidate = matches[0]
        prepared = [item for item in case['candidates'] if item['id'] == candidate['id']]
        if len(prepared) != 1 or any(candidate.get(key) != prepared[0][key] for key in ('path', 'seed', 'variant')):
            raise ValueError('Candidate differs from its prepared specification')
        if candidate['variant'] != 'generated':
            raise ValueError('Expert or input-image controls cannot become generated goals')
        source = inside(folder, candidate['path'])
        digest = sha(source)
        if digest != candidate['sha256']:
            raise ValueError('Selected goal checksum mismatch')
        receipt = json.loads((folder / f'{candidate["id"]}.receipt.json').read_text())
        if receipt['contract_sha256'] != generated['contract_sha256'] or receipt['sha256'] != digest:
            raise ValueError('Candidate generation receipt does not match')
        if receipt.get('seed') != candidate['seed'] or receipt.get('input_sha256') != case.get('input_rgb.png_sha256') or receipt.get('gt_sent_to_planner') is not False:
            raise ValueError('Candidate generation provenance does not match')
        path = f'goals/{case["key"]}.png'
        score = folder / 'score.json'
        model = json.loads(score.read_text()).get('model_returned') if score.exists() else None
        rows.append(dict(case, goal_path=path, goal_sha256=digest, selected_candidate=candidate['id'],
                         selection_method='explicit_human_selection', human_approved=True,
                         selection_note=decision.get('override_reason', ''), score_model=model,
                         score_sha256=sha(score) if score.exists() else None))
        files.append((source, path))
    manifest = dict(schema='vigar-subgoal-cache/v1', complete=True, provisional=False,
                    task_config_sha256=CONFIG_SHA256, cases=rows, checkpoint=contract['checkpoint'],
                    checkpoint_metadata_sha256=contract['checkpoint_metadata_sha256'],
                    weight_variant=contract['weight_variant'], dataset=str(dataset.resolve()),
                    annotations_sha256=contract['annotations_sha256'], split='clean',
                    tasks=len({row['task'] for row in rows}), episodes=len({row['episode_index'] for row in rows}),
                    slots=len(rows), selection_policy='explicit_human_selection',
                    human_reviewed_goals=sum(row['human_approved'] for row in rows),
                    selected_score_models=dict(Counter(row['score_model'] for row in rows if row['score_model'])),
                    decisions_sha256=sha(decisions_path), generation_contract_sha256=sha(root / 'contract.json'),
                    no_gt_fallback=True)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    try:
        (staging / 'goals').mkdir()
        for source, path in files:
            shutil.copy2(source, staging / path)
        write(staging / 'manifest.json', manifest)
        write(staging / 'FROZEN.json', dict(manifest_sha256=sha(staging / 'manifest.json'), slots=len(rows)))
        verify(staging, dataset)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging)
        raise



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.decisions, args.output, args.dataset)
