"""Reusable numeric Arrow files for random LeRobot episode reads.

Parquet may put hundreds of episodes in one compressed row group. An
uncompressed IPC sidecar permits memory-mapped reads without repeatedly
decompressing the full group. Source datasets remain immutable and untouched.
"""

import fcntl
import hashlib
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


COLUMNS = ("episode_index", "frame_index", "observation.state", "action")
OPTIONAL_COLUMNS = ("action_valid_mask", "observation.state_valid_mask")
CACHE_VERSION = 1


def cache_path(source, directory):
    source = Path(source).resolve()
    stat = source.stat()
    identity = f"{CACHE_VERSION}:{source}:{stat.st_size}:{stat.st_mtime_ns}"
    return Path(directory) / (hashlib.sha256(identity.encode()).hexdigest() + ".arrow")


def prepare_file(source, directory):
    """Serialize each source version once; flock and atomic rename protect readers."""
    target = cache_path(source, directory)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_file():
            return target
        fd, temporary = tempfile.mkstemp(prefix=target.stem + ".", suffix=".tmp", dir=target.parent)
        os.close(fd)
        try:
            with pq.ParquetFile(source) as parquet:
                names = parquet.schema_arrow.names
                for column in COLUMNS:
                    if column not in names:
                        raise ValueError(f"{source}: missing {column}")
                columns = [*COLUMNS, *(c for c in OPTIONAL_COLUMNS if c in names)]
                schema = pa.schema([parquet.schema_arrow.field(c) for c in columns])
                with pa.OSFile(temporary, "wb") as sink, pa.ipc.new_file(sink, schema) as writer:
                    for batch in parquet.iter_batches(batch_size=65536, columns=columns, use_threads=False):
                        writer.write_batch(batch.replace_schema_metadata(None))
            if cache_path(source, directory) != target:
                raise RuntimeError(f"{source}: source changed while preparing parquet cache")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return target


class ParquetEpisodeCache:
    """Bounded per-worker mmap handles; sidecars are shared across ranks/runs."""

    def __init__(self, directory, max_open_files=8):
        self.directory = str(directory)
        if type(max_open_files) is not int or max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.max_open_files = max_open_files
        self._tables = OrderedDict()
        self._files = OrderedDict()

    def __getstate__(self):
        # Spawn workers reopen mmaps themselves; no open Arrow objects cross processes.
        return dict(
            directory=self.directory, max_open_files=self.max_open_files, _tables=OrderedDict(), _files=OrderedDict()
        )

    def _read(self, source, episode):
        target = prepare_file(source, self.directory)
        key = str(target)
        if key not in self._tables:
            handle = pa.memory_map(key, "r")
            try:
                table = pa.ipc.open_file(handle).read_all()
                ids = table["episode_index"].combine_chunks()
                if ids.null_count:
                    raise ValueError(f"{source}: null episode_index")
                ids = ids.to_numpy(zero_copy_only=False)
                boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
                spans = {}
                for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
                    if start != end:
                        spans.setdefault(int(ids[start]), []).append((int(start), int(end - start)))
            except Exception:
                handle.close()
                raise
            self._tables[key] = (handle, table, spans)
            while len(self._tables) > self.max_open_files:
                self._tables.popitem(last=False)[1][0].close()
        self._tables.move_to_end(key)
        _, table, spans = self._tables[key]
        selected = [table.slice(start, length) for start, length in spans.get(episode, [])]
        # Arrow buffers retain their mmap ownership, even if the cache closes
        # its file handle. The caller sorts/copies only this episode's rows.
        return pa.concat_tables(selected) if selected else table.slice(0, 0)

    def read_episode(self, root, info, metadata):
        root = Path(root)
        episode = int(metadata["episode_index"])
        hinted = None
        if "data_path" in info and all(k in metadata for k in ("data/chunk_index", "data/file_index")):
            source = root / info["data_path"].format(
                chunk_index=int(metadata["data/chunk_index"]), file_index=int(metadata["data/file_index"])
            )
            if source.is_file():
                hinted = (source, self._read(source, episode))
                if len(hinted[1]) == metadata["length"]:
                    return hinted[1]
        # Historical manifests can lack file hints, and an episode may span
        # parts. In those cases preserve the original all-parts episode selection.
        key = str(root)
        if key not in self._files:
            self._files[key] = sorted((root / "data").rglob("*.parquet"))
            while len(self._files) > 64:
                self._files.popitem(last=False)
        self._files.move_to_end(key)
        if not self._files[key]:
            raise ValueError(f"{root}: no parquet data files")
        tables = [
            hinted[1] if hinted and source == hinted[0] else self._read(source, episode) for source in self._files[key]
        ]
        return pa.concat_tables(tables)

    def close(self):
        for handle, _, _ in self._tables.values():
            handle.close()
        self._tables.clear()
        self._files.clear()
