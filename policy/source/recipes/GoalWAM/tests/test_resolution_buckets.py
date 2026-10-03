"""Bucket geometry, metadata, global step scheduling and consumed-cursor resume."""

import copy
import itertools
import json
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from recipes.GoalWAM.data.data_loader import BucketWindowSampler, StatefulWindowLoader, collate_samples
from recipes.GoalWAM.data.dataset import LeRobotPolicyDataset, LeRobotPolicySFTDataset
from recipes.GoalWAM.data.images import CAMERA_KEYS, compose_goal_image, letterbox
from recipes.GoalWAM.data.resolution import assign_bucket, episode_resolution, validate_buckets
from recipes.GoalWAM.tests.test_goal_sampling import GoalStreamDataset
from recipes.GoalWAM.tests.test_tail_windows import population
from recipes.GoalWAM.trainer.arguments import GoalWAMDataArguments
from recipes.GoalWAM.trainer.evaluator import inference_sample


BUCKETS = [(192, 320), (224, 288), (256, 256), (288, 224), (320, 192)]
CANVASES = [(288, 320), (352, 288), (384, 256), (448, 224), (480, 192)]


def variable_dataset(tmp_path, mode="multi_view", cameras=None, buckets=BUCKETS):
    base, _ = population(tmp_path)
    info_path = tmp_path / "data/meta/info.json"
    info = json.loads(info_path.read_text())
    info["video_layout"] = "per_episode_variable"
    for key in CAMERA_KEYS.values():
        if key in info["features"]:
            info["features"][key] = dict(dtype="video", shape=None)
    info_path.write_text(json.dumps(info))
    episode_path = tmp_path / "data/meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(episode_path).to_pylist()
    for i, row in enumerate(rows):
        h, w = BUCKETS[i % len(BUCKETS)]
        row.update(image_height=h * 2, image_width=w * 2)
        annotation = json.loads(row["annotation"])
        annotation["resolution"] = [h * 2, w * 2]
        row["annotation"] = json.dumps(annotation)
    pq.write_table(pa.Table.from_pylist(rows), episode_path)
    raw = LeRobotPolicyDataset(
        base.manifest,
        {"robotwin_aloha_agilex": str(tmp_path / "norm.json")},
        img_size=[224, 288],
        img_size_buckets=buckets,
        enable_cameras=cameras,
        training=True,
        include_tail_windows=True,
        goal_image_composition=mode,
    )
    raw._episode_rows = base._episode_rows
    raw._video = synthetic_video
    return raw


def synthetic_video(ep, camera, indices):
    # Different native sizes/aspects in one episode are valid; canonical episode
    # dimensions choose a shared bucket, never substitute for camera dimensions.
    h, w, value = {
        CAMERA_KEYS["head"]: (24, 32, 50),
        CAMERA_KEYS["left"]: (16, 32, 100),
        CAMERA_KEYS["right"]: (32, 16, 200),
    }[camera]
    return np.full((len(indices), h, w, 3), value, dtype=np.uint8)


@pytest.mark.parametrize(
    "mode,cameras",
    [
        ("multi_view", ["head", "left", "right"]),
        ("multi_view", ["head"]),
    ],
)
def test_all_episode_buckets_and_native_preparation(tmp_path, mode, cameras):
    raw = variable_dataset(tmp_path, mode, cameras)
    sft = LeRobotPolicySFTDataset(raw)
    for i, size in enumerate(BUCKETS):
        sample = raw[i * 64]
        expected_hw = size if cameras == ["head"] else CANVASES[i]
        assert tuple(sample["target_hw"].tolist()) == size
        assert sample["goal_frame"].shape == (3, 1, *expected_hw)
        assert sample["video"].shape == (3, 13, *expected_hw)
        for name, (y, x, h, w) in sample["camera_boxes"].items():
            pixels = sample["video_pixel_mask"][y : y + h, x : x + w]
            assert pixels.any()
            assert (
                sample["video"][..., y : y + h, x : x + w][..., pixels]
                .eq({"head": 50, "left": 100, "right": 200}[name])
                .all()
            )
        prepared = sft[i * 64]
        assert prepared["image_size"].tolist() == [*expected_hw, *expected_hw]
        assert prepared["sequence_plan"].vision_item_roles == ["goal", "default"]
        inference = inference_sample(prepared)
        assert torch.equal(inference["video"][0], sample["goal_frame"])
        assert not inference["video"][-1][:, 1:].any()
        # Tail duration variation is still legal inside one spatial bucket.
        batch = collate_samples([prepared, sft[i * 64 + 63]])
        assert [items[-1].shape[1] for items in batch["video"]] == [13, 1]
    with pytest.raises(ValueError, match="resolution buckets"):
        collate_samples([sft[0], sft[64]])
    bad = sft[64]
    bad["target_hw"] = sft[0]["target_hw"]
    with pytest.raises(ValueError, match="canvas shapes"):
        collate_samples([sft[0], bad])


def test_single_head_camera_uses_exact_bucket_and_letterbox():
    images = {"head": torch.full((1, 3, 240, 320), 101, dtype=torch.uint8)}
    assert assign_bucket((240, 320), BUCKETS) == (1, (224, 288))
    for size in BUCKETS:
        actual, mask, boxes = compose_goal_image(images, size, ["head"])
        expected, expected_mask = letterbox(images["head"], size)
        assert torch.equal(actual, expected) and torch.equal(mask, expected_mask)
        assert set(boxes) == {"head"}
    assert assign_bucket((200, 200), [(128, 256), (256, 128)]) == (0, (128, 256))


@pytest.mark.parametrize(
    "buckets", [[[223, 288]], [[224, 288], [224, 288]], [[0, 32]], [[True, 32]], [[224.0, 288]], [[32]], "bad"]
)
def test_bad_bucket_configuration(buckets):
    with pytest.raises(ValueError, match="img_size_buckets"):
        validate_buckets(buckets, [224, 288])


def test_missing_default_bucket_and_argument_wiring():
    with pytest.raises(ValueError, match="img_size must"):
        GoalWAMDataArguments(train_path="unused", img_size=[32, 32], img_size_buckets=BUCKETS)
    assert GoalWAMDataArguments(
        train_path="unused", img_size=[224, 288], img_size_buckets=BUCKETS
    ).img_size_buckets == [list(s) for s in BUCKETS]
    assert validate_buckets([], None) == ()


def test_authoritative_resolution_and_uniform_fallback():
    info = dict(video_layout="uniform", features={"a": {"shape": [24, 32, 3]}, "b": {"shape": [12, 16, 3]}})
    assert episode_resolution(info, {}, ["a", "b"], context="test") == (24, 32)
    assert episode_resolution(info, {"image_height": 320, "image_width": 192}, ["a", "b"], context="test") == (
        320,
        192,
    )
    info["features"]["b"]["shape"] = [16, 16, 3]
    with pytest.raises(ValueError, match="ambiguous"):
        episode_resolution(info, {}, ["a", "b"], context="test")
    for ep in (
        {"image_height": 32},
        {"image_height": None, "image_width": 32},
        {"image_height": 0, "image_width": 32},
    ):
        with pytest.raises(ValueError, match="resolution"):
            episode_resolution(info, ep, ["a"], context="test")
    info["video_layout"] = "per_episode_variable"
    with pytest.raises(ValueError, match="authoritative"):
        episode_resolution(info, {}, ["a"], context="test")


def test_variable_camera_dimensions_are_not_canonical_dimensions(tmp_path):
    raw = variable_dataset(tmp_path)
    ep = raw.episodes[0]
    entry = raw.entries[ep["name"]]
    entry["info"]["video_path"] = "{video_key}"

    class Reader:
        def read(self, timestamps):
            return synthetic_video(ep, CAMERA_KEYS["head"], timestamps)

    camera = CAMERA_KEYS["head"]
    raw._readers[str(entry["root"] / camera)] = Reader()
    ep["metadata"].update({f"videos/{camera}/{key}": 0 for key in ("chunk_index", "file_index", "from_timestamp")})
    frames = LeRobotPolicyDataset._video(raw, ep, camera, [0, 1])
    assert frames.shape == (2, 24, 32, 3)


class BucketStreamDataset(GoalStreamDataset):
    """Picklable small population for real worker/prefetch tests."""

    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        self.img_size_buckets = ((32, 64), (64, 32))
        self.episodes = [dict(bucket_id=0, weighted=8), dict(bucket_id=1, weighted=24)]

    def __len__(self):
        return 32

    def locate(self, index):
        return self.episodes[int(index >= 8)], index

    def resolution_bucket_record(self):
        return dict(version=1, buckets=self.img_size_buckets)

    def __getitem__(self, key):
        result = super().__getitem__(key)
        result["bucket"] = int(result["index"] >= 8)
        return result


@pytest.mark.parametrize("world,batch,accum", [(1, 3, 2), (2, 2, 3), (3, 4, 1)])
def test_optimizer_steps_match_across_ranks_and_preserve_window_weights(world, batch, accum):
    raw = BucketStreamDataset()
    samplers = [
        BucketWindowSampler(raw, r, world, 42, batch_size=batch, accumulation_steps=accum, include_position=True)
        for r in range(world)
    ]
    nsteps = 1200
    streams = [list(itertools.islice(iter(s), nsteps * batch * accum)) for s in samplers]
    counts = Counter()
    for start in range(0, nsteps * batch * accum, batch * accum):
        step = [row for stream in streams for row in stream[start : start + batch * accum]]
        assert len({index >= 8 for index, _ in step}) == 1
        assert len({pos for _, pos in step}) == world * batch * accum
        counts.update(index >= 8 for index, _ in step)
    assert counts[True] / sum(counts.values()) == pytest.approx(0.75, abs=0.045)
    # Mid-step resume reconstructs the same bucket and the next rank-local draw.
    for r, sampler in enumerate(samplers):
        sampler.cursor = 5
        resumed = BucketWindowSampler(
            raw, r, world, 42, batch_size=batch, accumulation_steps=accum, include_position=True
        )
        resumed.load_state_dict(sampler.state_dict())
        assert list(itertools.islice(iter(resumed), 17)) == streams[r][5:22]


def test_fixed_population_sampling_stays_in_subset():
    sampler = BucketWindowSampler(BucketStreamDataset(), 0, 1, 42, [1, 1, 10], batch_size=2, accumulation_steps=2)
    result = list(itertools.islice(iter(sampler), 1000))
    assert set(result) == {1, 10}
    assert result.count(1) / len(result) == pytest.approx(2 / 3, abs=0.08)


def collect_stream(workers, dropout, count, state=None, *, prefetch_factor=2):
    loader = StatefulWindowLoader(
        BucketStreamDataset(dropout),
        batch_size=2,
        accumulation_steps=3,
        num_workers=workers,
        prefetch_factor=prefetch_factor,
    )
    if workers:
        loader.loader.multiprocessing_context = "spawn"
    if state is not None:
        loader.load_state_dict(state)
    iterator = iter(loader)
    result = []
    try:
        for _ in range(count):
            batch = next(iterator)
            result.extend(
                [
                    {k: (batch[k][i].item() if isinstance(batch[k][i], torch.Tensor) else batch[k][i]) for k in batch}
                    for i in range(2)
                ]
            )
        saved = copy.deepcopy(loader.state_dict())
    finally:
        iterator.close()
        if workers:
            loader.loader._iterator._shutdown_workers()
    return result, saved


def test_prefetch_resume_and_goal_dropout_rng_independence():
    baseline, _ = collect_stream(0, 0.1, 7)
    first, state = collect_stream(2, 0.1, 2, prefetch_factor=1)
    rest, _ = collect_stream(2, 0.1, 5, state, prefetch_factor=4)
    assert first + rest == baseline
    no_dropout, _ = collect_stream(0, 0, 7)
    for a, b in zip(baseline, no_dropout, strict=True):
        assert {k: v for k, v in a.items() if k != "dropped"} == {k: v for k, v in b.items() if k != "dropped"}


def test_resume_rejects_changed_buckets_population_and_batch_geometry(tmp_path):
    raw = variable_dataset(tmp_path)
    loader = StatefulWindowLoader(LeRobotPolicySFTDataset(raw), batch_size=2, accumulation_steps=3)
    state = loader.state_dict()
    for batch, accum in [(1, 3), (2, 2)]:
        with pytest.raises(ValueError, match="bucket sampling"):
            StatefulWindowLoader(
                LeRobotPolicySFTDataset(raw), batch_size=batch, accumulation_steps=accum
            ).load_state_dict(state)
    raw.episodes[0]["source_hw"] = (512, 512)
    with pytest.raises(ValueError, match="bucket sampling"):
        StatefulWindowLoader(LeRobotPolicySFTDataset(raw), batch_size=2, accumulation_steps=3).load_state_dict(state)
    raw.img_size_buckets = ()
    with pytest.raises(ValueError, match="bucket sampling"):
        StatefulWindowLoader(LeRobotPolicySFTDataset(raw), batch_size=2, accumulation_steps=3).load_state_dict(state)


def test_bucket_loader_delivers_native_batches_in_distributed_accumulation(tmp_path):
    raw = variable_dataset(tmp_path)
    loaders = [
        StatefulWindowLoader(LeRobotPolicySFTDataset(raw), rank=rank, world_size=2, batch_size=2, accumulation_steps=3)
        for rank in range(2)
    ]
    iterators = [iter(loader) for loader in loaders]
    try:
        for _ in range(3):
            step_shapes = set()
            for _ in range(3):
                for iterator in iterators:
                    batch = next(iterator)
                    assert len(batch["video"]) == 2
                    for items, size in zip(batch["video"], batch["image_size"], strict=True):
                        assert items[0].shape[-2:] == items[1].shape[-2:]
                        assert size.tolist() == [*items[1].shape[-2:]] * 2
                        assert items[1].shape[1] in (1, 5, 9, 13)
                        step_shapes.add(items[1].shape[-2:])
            assert len(step_shapes) == 1
    finally:
        for iterator in iterators:
            iterator.close()
