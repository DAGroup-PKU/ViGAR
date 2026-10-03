"""Exact sparse video reads across goal gaps and shared-file episode offsets."""

import av
import numpy as np
import pytest

from recipes.GoalWAM.data.dataset import FrameReader


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("sparse_video") / "shared.mp4"
    rng = np.random.default_rng(42)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=30)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": "12", "bf": "2"}
        for _ in range(400):
            pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as source:
        reference = {
            round(float(f.pts * f.time_base) * 30): f.to_ndarray(format="rgb24") for f in source.decode(video=0)
        }
    return path, reference


class CountDecodes:
    def __init__(self, container):
        self.container = container
        self.frames = self.seeks = 0

    def seek(self, *args, **kwargs):
        self.seeks += 1
        return self.container.seek(*args, **kwargs)

    def decode(self, *args, **kwargs):
        for frame in self.container.decode(*args, **kwargs):
            self.frames += 1
            yield frame

    def close(self):
        self.container.close()


@pytest.mark.parametrize("rate", [1, 3])
def test_sparse_goal_exact_pixels_order_duplicates_and_cache(video, rate):
    path, reference = video
    reader = FrameReader(path, 30)
    reader.container = CountDecodes(reader.container)
    # Episode starts inside a shared video; target requests may be reordered.
    rollout = [61 + j * 4 * rate for j in range(13)]
    wanted = [371, *reversed(rollout), 371, rollout[0]]
    try:
        actual = reader.read([i / 30 for i in wanted])
        np.testing.assert_array_equal(actual, np.stack([reference[i] for i in wanted]))
        assert reader.container.seeks == 2
        assert reader.container.frames <= 48 * rate + 25
        before = reader.container.frames
        np.testing.assert_array_equal(reader.read([i / 30 for i in wanted]), actual)
        assert reader.container.frames == before
        # Partially cached requests still return the correct nearby and far RGB.
        more = [62, 371, 375, 62]
        np.testing.assert_array_equal(reader.read([i / 30 for i in more]), np.stack([reference[i] for i in more]))
    finally:
        reader.close()


def test_nearby_requests_stay_in_one_pass_and_cache_is_bounded(video):
    path, reference = video
    reader = FrameReader(path, 30)
    reader.container = CountDecodes(reader.container)
    try:
        wanted = list(range(150, 300))
        np.testing.assert_array_equal(reader.read([i / 30 for i in wanted]), np.stack([reference[i] for i in wanted]))
        assert reader.container.seeks == 1
        assert len(reader.cache) == 96
        np.testing.assert_array_equal(reader.read([150 / 30]), reference[150][None])
        assert len(reader.cache) == 96
    finally:
        reader.close()


def test_missing_goal_still_fails(video):
    path, _ = video
    reader = FrameReader(path, 30)
    try:
        with pytest.raises(ValueError, match="Missing video frames"):
            reader.read([0.0, 500 / 30])
    finally:
        reader.close()
