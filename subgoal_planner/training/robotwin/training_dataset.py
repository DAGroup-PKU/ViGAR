from functools import lru_cache
import gzip
import json
import os
import torch
from pathlib import Path
from sampling import UnifiedIndex
from fixed_ee_roi import VIEWS, ee_point, project, rectangular_canvas


class SelectedDataset:
    def __init__(self, base, records, geometry):
        self.base = base
        self.records = records
        self.geometry = geometry
        self.index = UnifiedIndex(base.episodes, base.samples, records)

    def __len__(self):
        return len(self.base)

    @lru_cache(maxsize=128)
    def mask(self, eid, frame):
        if eid not in self.geometry:
            if not self.records[eid].get("source_record"):
                raise ValueError(f"Unexpected missing ROI: {eid}")
            return (torch.zeros(384, 320), "missing_metadata_global_loss")
        target = self.geometry[eid][frame]
        views = {}
        for name, (camera, *_) in VIEWS.items():
            c = target["cameras"][camera]
            views[name] = dict(
                source_hw=c["source_hw"],
                ee_pixels={
                    arm: project(ee_point(p), c["extrinsic_cv"], c["intrinsic_cv"])
                    for (arm, p) in target["poses"].items()
                },
            )
        (mask, _) = rectangular_canvas(views)
        return (mask, "ee_window" if mask.any() else "offscreen_global_loss")

    def __getitem__(self, index):
        item = dict(self.base[self.index.resolve(index)])
        eid = int(item["episode_id"])
        frame = int(item["target_frame_index"])
        if int(item["n_orig_video_frames"]) != self.records[eid]["frame_count"]:
            raise ValueError("Training frame count drift")
        (mask, status) = self.mask(eid, frame)
        item["_loss_roi_pair"] = torch.stack([torch.zeros_like(mask), mask])
        return item


def get_dataset(**kwargs):
    from cosmos_framework.data.vfm.local_datasets import (
        episode_image_edit_dataset as upstream,
    )

    DATASET = Path(os.environ["EPISODE_IMAGE_EDIT_DATASET_PATH"])
    metadata = Path(os.environ["SUBGOAL_PLANNER_ROI_METADATA"])
    metadata_cache = Path(os.environ["SUBGOAL_PLANNER_METADATA_CACHE"])
    rows = json.loads((DATASET / "meta/episode_manifest.json").read_text())["episodes"]
    records = {int(r["episode_index"]): r for r in rows}
    geometry = {}
    with gzip.open(metadata, "rt") as f:
        for r in map(json.loads, f):
            if not r["full_state_alignment"]["state_alignment_verified"]:
                raise ValueError("Unverified geometry")
            geometry[r["episode_index"]] = {t["frame"]: t for t in r["targets"]}
    missing = set(records) - set(geometry)
    allowed = set(
        json.loads((metadata.parent / "roi_fallback_allowlist.json").read_text())[
            "episode_ids"
        ]
    )
    if missing != allowed or len(missing) != 1750:
        raise ValueError("Unexpected ROI coverage")
    loader = upstream._load_metadata_cache
    cache = loader(metadata_cache)
    info = json.loads((DATASET / "meta/info.json").read_text())
    for r in rows:
        if r.get("source_record"):
            eid = int(r["episode_index"])
            key = info["video_path"].format(
                episode_chunk=eid // info["chunks_size"],
                episode_index=eid,
                video_key="observation.images.concat_view_384x320",
            )
            cache[key] = dict(
                width=320, height=384, fps=30.0, total_frames=r["frame_count"]
            )
    upstream._load_metadata_cache = lambda _: dict(cache)
    writer = upstream._write_metadata_cache
    upstream._write_metadata_cache = lambda *_a, **_k: None
    try:
        base = upstream.get_episode_image_edit_dataset(**kwargs)
    finally:
        upstream._load_metadata_cache = loader
        upstream._write_metadata_cache = writer
    assert float(base.next_subgoal_tail_fraction) == 0.15
    ds = SelectedDataset(base, records, geometry)
    return ds
