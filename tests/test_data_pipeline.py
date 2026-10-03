"""Exercise the final task selection, dataset materialization and cache binding on CPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data'))
sys.path.insert(0, str(ROOT / 'data/goal_generation'))
from task_config import CONFIG, CONFIG_SHA256
from common import sha, write
from prepare import canvas, stage_cases
from export_cache import export
from verify_cache import verify

spec = importlib.util.spec_from_file_location('annotation_builder', ROOT / 'data/stage_annotation/build_dataset.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture(scope='module')
def built_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp('data_pipeline')
    base, raw, output = root / 'source', root / 'raw', root / 'annotated'
    (base / 'meta/episodes/chunk-000').mkdir(parents=True)
    (base / 'data').mkdir(); (base / 'videos').mkdir()
    write(base / 'meta/info.json', {'features': {}})
    write(base / 'meta/stats.json', {'action': {key: [1.] * 14 for key in ('mean', 'std', 'min', 'max')}})
    template = root / 'trajectory.hdf5'
    with h5py.File(template, 'w') as handle:
        handle['joint_action/left_gripper'] = np.zeros(12)
        handle['joint_action/right_gripper'] = np.zeros(12)
        handle['joint_action/vector'] = np.zeros((12, 14))
    episodes, boundaries = [], []
    for family_index, (task, config) in enumerate(sorted(CONFIG['tasks'].items())):
        source = raw / task / 'demo_clean/data'
        source.mkdir(parents=True)
        (source.parent / 'seed.txt').write_text('\n'.join(str(i) for i in range(50)))
        for local in range(50):
            eid = family_index * 50 + local
            (source / f'episode{local}.hdf5').symlink_to(template)
            episodes.append(dict(episode_index=eid, tasks=[f'{task} instruction {local}'], length=12))
            if config['mode'] == 'subgoal':
                texts = config['stage_texts']
                if task == 'place_bread_basket' and local < 9:
                    texts = texts[-1:]
                ends = [(i + 1) * 12 // len(texts) for i in range(len(texts))]
                stages = [dict(start_frame=start, end_frame=end, subtask_text=text)
                          for start, end, text in zip([0] + ends[:-1], ends, texts)]
                boundaries.append(dict(task=task, local_episode_index=local, seed=local, frame_count=12,
                                       stage_count=len(texts), stages=stages,
                                       contract_version=CONFIG['stage_contract_version'], task_config_sha256=CONFIG_SHA256))
    pq.write_table(pa.Table.from_pylist(episodes), base / 'meta/episodes/chunk-000/file-000.parquet')
    pq.write_table(pa.Table.from_pylist([dict(task_index=i, task=task) for i, task in enumerate(sorted(CONFIG['tasks']))]), base / 'meta/tasks.parquet')
    boundary_path = root / 'boundaries.jsonl'
    boundary_path.write_text('\n'.join(json.dumps(row) for row in boundaries))
    annotations, mapping, family_map, extras = builder.build_records(base, raw, boundary_path)
    builder.build_runtime(base, raw, output, annotations, mapping, family_map, extras['goal_graph'],
                          extras['audit'], extras['stage_boundary_rows'], extras['stage_boundaries_sha256'])
    return root, output, annotations, boundary_path


def test_direct_build_preserves_task_and_stage_semantics(built_dataset):
    _, output, annotations, _ = built_dataset
    manifest = json.loads((output / 'episode_manifest.json').read_text())
    assert manifest['summary']['multi_subtask_families'] == 19
    assert manifest['summary']['single_subtask_families'] == 31
    assert manifest['summary']['semantic_family_stage_units'] == 74
    assert manifest['summary']['by_split']['train']['subtasks'] == 3691
    assert (output / 'data').is_dir() and (output / 'meta/info.json').is_file()
    for row in annotations.values():
        if row['family'] in {'open_microwave', 'handover_block', 'place_object_basket'}:
            assert row['goal_mode'] == 'episode_goal'
            assert len(row['action_steps']) == 1
            assert row['action_steps'][0]['action_text'] == row['task_name']
            assert row['key_frame']['single'] == [11]
        if row['family'] == 'place_bread_basket' and row['source_local_episode_index'] < 9:
            assert row['action_steps'][0]['info']['semantic_stage_index'] == 1
    assert all(row['task_config_sha256'] == CONFIG_SHA256 for row in annotations.values())


def test_boundary_coverage_and_config_binding(built_dataset, tmp_path):
    _, _, _, source = built_dataset
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows) == 950
    missing = tmp_path / 'missing.jsonl'
    missing.write_text('\n'.join(json.dumps(row) for row in rows[:-1]))
    with pytest.raises(ValueError, match='coverage'):
        builder.load_stage_boundaries(missing)
    rows[0]['task_config_sha256'] = 'different'
    missing.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match='different task configuration'):
        builder.load_stage_boundaries(missing)


def test_goal_cases_follow_final_annotations(built_dataset):
    _, _, annotations, _ = built_dataset
    cases = stage_cases(annotations, candidates=2)
    assert len(cases) == 3691
    assert len({case['key'] for case in cases}) == 3691
    assert all(case['target_frame'] == case['end_frame_exclusive'] - 1 for case in cases)
    assert cases == stage_cases(annotations, candidates=2)
    for task in ('open_microwave', 'handover_block', 'place_object_basket'):
        selected = [case for case in cases if case['task'] == task]
        assert len(selected) == 50
        assert all(case['input_frame'] == 0 and case['target_frame'] == 11 for case in selected)


def test_rgb_canvas_preserves_official_channel_convention(tmp_path):
    import cv2
    path = tmp_path / 'image.hdf5'
    with h5py.File(path, 'w') as handle:
        for camera, color in [('head_camera', (255, 0, 0)), ('left_camera', (0, 255, 0)), ('right_camera', (0, 0, 255))]:
            rgb = np.full((24, 32, 3), color, dtype=np.uint8)
            ok, encoded = cv2.imencode('.png', rgb)
            assert ok
            handle.create_dataset(f'observation/{camera}/rgb', data=encoded[None])
        result = canvas(handle, 0)
    assert result.shape == (384, 320, 3)
    assert result[20, 20].tolist() == [255, 0, 0]
    assert result[300, 20].tolist() == [0, 255, 0]
    assert result[300, 300].tolist() == [0, 0, 255]


def test_export_uses_actual_checkpoint_and_selection_records(built_dataset, tmp_path):
    _, dataset, annotations, _ = built_dataset
    # Full coverage uses a shared synthetic image; these are CPU fixtures, not generated model results.
    root, output = tmp_path / 'generation', tmp_path / 'cache'
    root.mkdir()
    cases = stage_cases(annotations)
    image = root / 'synthetic.png'
    Image.new('RGB', (320, 384), (10, 20, 30)).save(image)
    write(root / 'cases.json', cases)
    write(root / 'contract.json', dict(task_config_sha256=CONFIG_SHA256, annotations_sha256=sha(dataset / 'meta/annotations.json'),
                                      checkpoint='selected-checkpoint', checkpoint_metadata_sha256='a' * 64, weight_variant='regular'))
    contract_sha = sha(root / 'contract.json')
    write(root / 'PREPARED.json', dict(contract_sha256=contract_sha, cases_sha256=sha(root / 'cases.json')))
    for case in cases:
        folder = root / case['key']; folder.mkdir()
        (folder / 'C1.png').hardlink_to(image)
        candidate = dict(case['candidates'][0], sha256=sha(image))
        write(folder / 'generated.json', dict(contract_sha256=contract_sha, case=dict(case, candidates=[candidate])))
        write(folder / 'C1.receipt.json', dict(contract_sha256=contract_sha, sha256=sha(image), seed=candidate['seed'], input_sha256=None, gt_sent_to_i2i=False))
    decisions = root / 'decisions.json'
    rows = [dict(key=case['key'], decision='C1', override_reason='') for case in cases]
    write(decisions, dict(cases_sha256=sha(root / 'cases.json'), decisions=rows))
    export(root, decisions, output, dataset)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['checkpoint'] == 'selected-checkpoint'
    assert manifest['human_reviewed_goals'] == 3691
    assert manifest['selected_score_models'] == {}
    assert verify(output, dataset)['passed']
    rows[0]['decision'] = 'reject_all'
    write(decisions, dict(cases_sha256=sha(root / 'cases.json'), decisions=rows))
    with pytest.raises(ValueError, match='Choose one generated candidate'):
        export(root, decisions, tmp_path / 'rejected', dataset)
    assert not (tmp_path / 'rejected').exists()
    manifest['cases'][0]['goal_path'] = '../outside.png'
    write(output / 'manifest.json', manifest)
    write(output / 'FROZEN.json', dict(manifest_sha256=sha(output / 'manifest.json'), slots=3691))
    with pytest.raises(ValueError, match='leaves its workspace'):
        verify(output, dataset)
