"""Verify the online state/control bridge against original HDF5 and 49D parquet."""

import argparse
import io
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import yaml
from scipy.spatial.transform import Rotation

from .geometry import ACTION_MASK, controller_actions, encode_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("/path/to/vigar/assets"))
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--recipe", type=Path, help="Also invert the recipe's relative joint/EEF representation")
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("horizon must be positive")
    layout = None
    if args.recipe:
        from recipes.ViGAR.data.relative_action import to_global_action, to_relative_action

        layout = yaml.safe_load(args.recipe.read_text())["data"]["action_layout"]
    records = []
    entries = list(yaml.safe_load(args.manifest.read_text()).items())[: args.tasks]
    for name, entry in entries:
        root = Path(entry["data_path"])
        task = root.parent.name
        episodes = pq.read_table(sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))[0]).to_pylist()
        for ep in (episodes[0], episodes[-1]):
            source = json.loads(ep["annotation"])["source"]
            archive = args.raw_root / task / source["archive"]
            member = f"{Path(source['archive']).stem}/data/episode{source['source_episode_index']}.hdf5"
            table = (
                pads.dataset(root / "data", format="parquet")
                .to_table(
                    columns=["frame_index", "observation.state", "action"],
                    filter=pads.field("episode_index") == ep["episode_index"],
                )
                .sort_by("frame_index")
            )
            stored = np.array(table["observation.state"].to_pylist(), dtype=np.float32)
            commands = np.array(table["action"].to_pylist(), dtype=np.float32)
            with zipfile.ZipFile(archive) as zipped, h5py.File(io.BytesIO(zipped.read(member)), "r") as h5:
                last_anchor = len(stored) - args.horizon - 1
                if last_anchor < 0:
                    raise ValueError(f"{name}: episode shorter than horizon + one observation")
                for frame in sorted({0, last_anchor // 2, last_anchor}):
                    obs = dict(observation={}, joint_action={}, endpose={})
                    matrix = np.asarray(h5["observation/head_camera/cam2world_gl"][frame])
                    for camera in ("head_camera", "left_camera", "right_camera"):
                        obs["observation"][camera] = dict(cam2world_gl=matrix, rgb=np.zeros((2, 2, 3), dtype=np.uint8))
                    for side in ("left", "right"):
                        for group, fields in [
                            ("joint_action", ("arm", "gripper")),
                            ("endpose", ("endpose", "gripper")),
                        ]:
                            for field in fields:
                                obs[group][f"{side}_{field}"] = np.asarray(h5[f"{group}/{side}_{field}"][frame])
                    actual = encode_observation(obs)["state"]
                    scalar_mask = np.ones(49, dtype=bool)
                    for lo in (29, 37, 45):
                        scalar_mask[lo : lo + 4] = False
                        angle = (
                            Rotation.from_quat(actual[lo : lo + 4]).inv()
                            * Rotation.from_quat(stored[frame, lo : lo + 4])
                        ).magnitude()
                        assert angle < 1e-6
                    error = float(abs(actual[scalar_mask] - stored[frame, scalar_mask]).max())
                    assert error < 1e-5
                    chunk = commands[frame : frame + args.horizon]
                    mask = np.tile(ACTION_MASK, (args.horizon, 1))
                    if layout is not None:
                        relative = to_relative_action(actual, chunk, layout)
                        reconstructed = to_global_action(actual, relative, layout)
                        # Check scalar channels and rotations separately: quaternion
                        # canonicalization may change sign without changing orientation.
                        valid_scalar = scalar_mask & ACTION_MASK
                        np.testing.assert_allclose(
                            reconstructed[:, valid_scalar], chunk[:, valid_scalar], rtol=0, atol=1e-6
                        )
                        for lo in (29, 37):
                            angle = (
                                (
                                    Rotation.from_quat(reconstructed[:, lo : lo + 4]).inv()
                                    * Rotation.from_quat(chunk[:, lo : lo + 4])
                                )
                                .magnitude()
                                .max()
                            )
                            assert angle < 1e-6
                        chunk = reconstructed
                    ee = controller_actions(chunk, mask, matrix, "ee")
                    joints = controller_actions(chunk, mask, matrix, "qpos")
                    for side, ee_lo, joint_lo in [("left", 0, 0), ("right", 8, 7)]:
                        future = slice(frame + 1, frame + args.horizon + 1)  # Stored command already shifted.
                        wanted = np.asarray(h5[f"endpose/{side}_endpose"][future])
                        np.testing.assert_allclose(ee[:, ee_lo : ee_lo + 3], wanted[:, :3], rtol=0, atol=1e-6)
                        angle = (
                            (
                                Rotation.from_quat(ee[:, ee_lo + np.array([4, 5, 6, 3])]).inv()
                                * Rotation.from_quat(wanted[:, [4, 5, 6, 3]])
                            )
                            .magnitude()
                            .max()
                        )
                        assert angle < 1e-6
                        np.testing.assert_allclose(
                            joints[:, joint_lo : joint_lo + 6],
                            h5[f"joint_action/{side}_arm"][future],
                            rtol=0,
                            atol=1e-6,
                        )
                        np.testing.assert_allclose(
                            ee[:, ee_lo + 7],
                            np.asarray(h5[f"endpose/{side}_gripper"][future]).reshape(-1),
                            rtol=0,
                            atol=1e-6,
                        )
                    records.append(
                        dict(
                            dataset=name,
                            episode=ep["episode_index"],
                            frame=frame,
                            state_scalar_max_abs=error,
                            passed=True,
                        )
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                passed=True,
                windows=len(records),
                horizon=args.horizon,
                relative_roundtrip=layout is not None,
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"PASS: {len(records)} real windows, both EEF and joint inverse commands")


if __name__ == "__main__":
    main()
