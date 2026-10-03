from types import SimpleNamespace

import numpy as np

from recipes.simulation.robotwin.common.rendering import head_video_frame


def test_video_capture_preserves_render_update_and_captures_only_head():
    calls = []

    class Camera:
        def take_picture(self):
            calls.append("capture")

        def get_picture(self, name):
            calls.append(name)
            return np.array([[[1.1, -0.1, 0.5, 1.0]]], dtype=np.float32)

    environment = SimpleNamespace(
        _update_render=lambda: calls.append("update"),
        cameras=SimpleNamespace(static_camera_list=[None, Camera()], head_camera_id=1),
    )
    rgb = head_video_frame(environment)
    assert calls == ["update", "capture", "Color"]
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb, [[[255, 0, 127]]])
