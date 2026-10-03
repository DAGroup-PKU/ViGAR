"""Exact cached episode rows, concurrent publication, invalidation and bounded maps."""

import os
import pickle
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from recipes.GoalWAM.data.dataset import LeRobotPolicyDataset
from recipes.GoalWAM.data.parquet_cache import ParquetEpisodeCache, cache_path, prepare_file


def make_parts(root, *, fixed=False):
    directory = root / "data/chunk-000"
    directory.mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(42)
    for episode in range(4):
        for frame in range(8):
            rows.append(
                {
                    "episode_index": episode,
                    "frame_index": frame,
                    "observation.state": rng.standard_normal(49).astype(np.float32).tolist(),
                    "action": rng.standard_normal(49).astype(np.float32).tolist(),
                    "action_valid_mask": (rng.random(49) > 0.2).tolist(),
                    "observation.state_valid_mask": (rng.random(49) > 0.2).tolist(),
                }
            )
    # Episode 1 spans files; frame rows are deliberately out of order.
    first, second = rows[:12][::-1], rows[12:][::-1]
    for index, values in enumerate((first, second)):
        table = pa.Table.from_pylist(values)
        if fixed:
            fields = [
                pa.field(f.name, pa.list_(pa.bool_() if "mask" in f.name else pa.float32(), 49))
                if f.name in ("observation.state", "action", "action_valid_mask", "observation.state_valid_mask")
                else f
                for f in table.schema
            ]
            table = table.cast(pa.schema(fields))
        pq.write_table(table, directory / f"file-{index:03d}.parquet", row_group_size=3)
    return sorted(directory.glob("*.parquet"))


def reader(root, cache):
    raw = object.__new__(LeRobotPolicyDataset)
    raw.entries = {
        "task": {"root": root, "info": {"data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}}
    }
    raw._rows = OrderedDict()
    raw._readers = OrderedDict()
    raw._parquet_cache = cache
    return raw


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("hint", [False, True])
def test_cached_rows_equal_scanner_across_parts_and_masks(tmp_path, fixed, hint):
    root = tmp_path / "source"
    make_parts(root, fixed=fixed)
    old, new = reader(root, None), reader(root, ParquetEpisodeCache(tmp_path / "cache", max_open_files=1))
    try:
        for episode in (0, 1, 2, 3, 0):
            metadata = dict(episode_index=episode, length=8)
            if hint:
                metadata.update({"data/chunk_index": 0, "data/file_index": 0 if episode < 2 else 1})
            ep = dict(name="task", metadata=metadata)
            expected, actual = old._episode_rows(ep), new._episode_rows(ep)
            assert set(expected) == set(actual)
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
            assert len(new._parquet_cache._tables) <= 1
        restored = pickle.loads(pickle.dumps(new._parquet_cache))
        assert not restored._tables
        assert restored.directory == new._parquet_cache.directory
        restored.close()
    finally:
        old.close()
        new.close()


def test_warm_cache_avoids_parquet_reads_and_concurrent_builds(tmp_path, monkeypatch):
    root = tmp_path / "source"
    source = make_parts(root)[0]
    directory = tmp_path / "cache"
    with ThreadPoolExecutor(max_workers=4) as pool:
        targets = list(pool.map(lambda _: prepare_file(source, directory), range(4)))
    assert len(set(targets)) == 1 and len(list(directory.glob("*.arrow"))) == 1
    assert not list(directory.glob("*.tmp"))
    before = targets[0].stat().st_mtime_ns

    def unexpected(*args, **kwargs):
        raise AssertionError("A warm IPC cache must not open the parquet decoder")

    monkeypatch.setattr(pq, "ParquetFile", unexpected)
    cache = ParquetEpisodeCache(directory)
    metadata = {"episode_index": 0, "length": 8, "data/chunk_index": 0, "data/file_index": 0}
    try:
        table = cache.read_episode(root, reader(root, None).entries["task"]["info"], metadata)
        assert len(table) == 8
        assert targets[0].stat().st_mtime_ns == before
    finally:
        cache.close()


def test_source_change_selects_new_cache_file(tmp_path):
    source = make_parts(tmp_path / "source")[0]
    directory = tmp_path / "cache"
    old = prepare_file(source, directory)
    table = pq.read_table(source).to_pylist()
    table[0]["action"][0] += 10
    pq.write_table(pa.Table.from_pylist(table), source)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    new = prepare_file(source, directory)
    assert new != old and new.is_file()
    with pa.memory_map(str(new)) as handle:
        assert pa.ipc.open_file(handle).read_all()["action"][0].as_py()[0] == table[0]["action"][0]


def test_failed_build_never_publishes_partial_file(tmp_path, monkeypatch):
    source = make_parts(tmp_path / "source")[0]
    directory = tmp_path / "cache"

    def fail(*args, **kwargs):
        raise RuntimeError("simulated read failure")

    monkeypatch.setattr(pq, "ParquetFile", fail)
    with pytest.raises(RuntimeError, match="simulated read failure"):
        prepare_file(source, directory)
    assert not cache_path(source, directory).exists()
    assert not list(directory.glob("*.tmp"))


def test_cached_reader_still_rejects_invalid_rows(tmp_path):
    root = tmp_path / "source"
    make_parts(root)
    raw = reader(root, ParquetEpisodeCache(tmp_path / "cache"))
    try:
        with pytest.raises(ValueError, match="noncontiguous"):
            raw._episode_rows(dict(name="task", metadata=dict(episode_index=0, length=9)))
    finally:
        raw.close()
