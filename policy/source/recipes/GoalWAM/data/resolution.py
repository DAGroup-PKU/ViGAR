"""Episode aspect-ratio assignment, independent of decoding and sampling RNG."""

import math


def validate_buckets(buckets, img_size):
    if buckets is None:
        return ()
    if not isinstance(buckets, (list, tuple)):
        raise ValueError("img_size_buckets must be a list of [HEIGHT, WIDTH] pairs")
    result = []
    for size in buckets:
        if (
            not isinstance(size, (list, tuple))
            or len(size) != 2
            or any(type(v) is not int or v <= 0 or v % 32 for v in size)
        ):
            raise ValueError("img_size_buckets edges must be positive integers divisible by 32")
        result.append(tuple(size))
    if len(set(result)) != len(result):
        raise ValueError("img_size_buckets must not contain duplicate entries")
    if result and (img_size is None or tuple(img_size) not in result):
        raise ValueError("img_size must be present in img_size_buckets")
    return tuple(result)


def assign_bucket(source_hw, buckets):
    """Minimize letterbox padding; configuration order breaks ties."""
    h, w = source_hw
    if any(type(v) is not int or v <= 0 for v in (h, w)):
        raise ValueError("Source resolution must contain positive integer HEIGHT WIDTH")

    def padding(index):
        th, tw = buckets[index]
        scale = min(th / h, tw / w)
        return 1.0 - (h * scale) * (w * scale) / (th * tw), index

    index = min(range(len(buckets)), key=padding)
    return index, buckets[index]


def episode_resolution(info, episode, available_cameras, *, context):
    """Episode dimensions are canonical for bucketing, not each camera's decoded size."""
    h, w = episode.get("image_height"), episode.get("image_width")
    if h is not None or w is not None:
        if any(type(v) is not int or v <= 0 for v in (h, w)):
            raise ValueError(f"{context}: invalid or partial authoritative episode resolution")
        return h, w
    if info.get("video_layout") == "per_episode_variable":
        raise ValueError(f"{context}: missing authoritative episode resolution")
    shapes = [tuple(info["features"][key]["shape"][:2]) for key in available_cameras]
    # Uniform datasets may have different camera sizes but must agree on aspect
    # ratio if no canonical episode dimensions were supplied.
    if not shapes or any(not math.isclose(h / w, shapes[0][0] / shapes[0][1]) for h, w in shapes):
        raise ValueError(f"{context}: ambiguous dataset-level aspect ratios; provide episode image_height/image_width")
    return shapes[0]
