"""Head-camera video capture without rendering unused wrist-camera frames."""


def head_video_frame(environment):
    # Preserve the upstream render update, including crazy-light RNG updates.
    # Policy observations still use get_obs() and render all three cameras.
    environment._update_render()
    cameras = environment.cameras
    camera = cameras.static_camera_list[cameras.head_camera_id]
    camera.take_picture()
    return (camera.get_picture("Color") * 255).clip(0, 255).astype("uint8")[:, :, :3]
