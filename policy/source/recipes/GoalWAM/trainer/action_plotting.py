"""Headless physical-action trajectory plots for GoalWAM."""

import logging
import os
from io import BytesIO
from typing import Any


os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..data.action_layout import ActionLayout


logger = logging.getLogger(__name__)


def equal_xyz_limits(
    points: np.ndarray,
    *,
    min_span: float = 1e-2,
    padding_fraction: float = 0.15,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return centered XYZ limits with one numeric span for all three axes."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 3), dtype=np.float64)
    elif points.ndim == 1 and points.shape[0] == 3:
        points = points.reshape(1, 3)
    elif points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape (..., 3), got {points.shape}.")
    else:
        points = points.reshape(-1, 3)

    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        points = np.zeros((1, 3), dtype=np.float64)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    data_span = max(float(np.max(maxs - mins)), float(min_span))
    half_span = data_span * (0.5 + float(padding_fraction))
    return (
        (float(centers[0] - half_span), float(centers[0] + half_span)),
        (float(centers[1] - half_span), float(centers[1] + half_span)),
        (float(centers[2] - half_span), float(centers[2] + half_span)),
    )


def set_equal_3d_axes(
    ax: Any,
    points: np.ndarray,
    *,
    min_span: float = 1e-2,
    padding_fraction: float = 0.15,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Apply equal numeric scales and a cubic display box to a 3D axis."""
    limits = equal_xyz_limits(points, min_span=min_span, padding_fraction=padding_fraction)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_zlim(limits[2])
    # Matplotlib's default 3D box aspect is 4:4:3, which visually compresses Z
    # even when the three numeric limit spans are identical.
    ax.set_box_aspect((1, 1, 1))
    return limits


def _figure_to_png_bytes(fig) -> bytes:
    """Encoded PNG rather than a ``PILImage``: rendered rows cross ranks in the
    gather below, and a pickled ``PILImage`` carries raw pixels (~750 KiB per
    plot) while the PNG stays around 50 KiB."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    plt.close(fig)
    return buffer.getvalue()


def _plot_chassis_bev(pred, gt, action_layout: ActionLayout):
    """Render chassis XY in NWU: X+ forward/north, Y+ left/west.

    Matplotlib's horizontal axis is world Y and its vertical axis is world X.
    The horizontal axis is inverted so positive Y appears on the viewer's left.
    """
    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        px = pred[:, action_layout.chassis_x]
        py = pred[:, action_layout.chassis_y]
        pw = pred[:, action_layout.chassis_angle]
        gx = gt[:, action_layout.chassis_x]
        gy = gt[:, action_layout.chassis_y]
        gw = gt[:, action_layout.chassis_angle]

        # Plot as (world_Y, world_X), then invert x-axis so Y+ is visually left.
        ax.plot(gy, gx, "-o", color="tab:green", markersize=3, linewidth=1.2, label="GT")
        ax.plot(py, px, "-o", color="tab:red", markersize=3, linewidth=1.2, label="Pred")

        span = max(np.ptp(np.concatenate([gx, px])), np.ptp(np.concatenate([gy, py])), 1e-6)
        arrow_len = max(0.05 * span, 1e-2)
        for x, y, w, c in ((gx[0], gy[0], gw[0], "tab:green"), (px[0], py[0], pw[0], "tab:red")):
            # Heading yaw is NWU (cos -> X forward, sin -> Y left). On the
            # (world_Y, world_X) plot this becomes (sin, cos).
            ax.arrow(y, x, arrow_len * np.sin(w), arrow_len * np.cos(w), color=c, head_width=arrow_len * 0.4)
        ax.scatter([gy[0], py[0]], [gx[0], px[0]], c=["tab:green", "tab:red"], marker="s", s=40)
        ax.scatter([gy[-1], py[-1]], [gx[-1], px[-1]], c=["tab:green", "tab:red"], marker="X", s=40)

        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_xaxis()
        ax.set_xlabel("Y (Left/W)")
        ax.set_ylabel("X (Fwd/N)")
        ax.set_title(f"Chassis BEV NWU (action {action_layout.chassis_x}-{action_layout.chassis_angle})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return _figure_to_png_bytes(fig)
    except Exception as exc:
        logger.info(f"[goalwam eval] Failed to plot chassis BEV: {exc}")
        plt.close("all")
        return None


def _plot_eef_3d(pred, gt, pos_slice, title):
    """Render EEF delta in anchor-camera NWU: X+ forward, Y+ left, Z+ up."""
    try:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        p = pred[:, pos_slice]
        g = gt[:, pos_slice]
        ax.plot(g[:, 0], g[:, 1], g[:, 2], "-o", color="tab:green", markersize=2.5, linewidth=1.2, label="GT")
        ax.plot(p[:, 0], p[:, 1], p[:, 2], "-o", color="tab:red", markersize=2.5, linewidth=1.2, label="Pred")
        ax.scatter([g[0, 0], p[0, 0]], [g[0, 1], p[0, 1]], [g[0, 2], p[0, 2]], c=["tab:green", "tab:red"], marker="s")
        ax.scatter(
            [g[-1, 0], p[-1, 0]], [g[-1, 1], p[-1, 1]], [g[-1, 2], p[-1, 2]], c=["tab:green", "tab:red"], marker="X"
        )
        set_equal_3d_axes(ax, np.concatenate([g, p], axis=0))
        ax.set_xlabel("camera-t X (forward)")
        ax.set_ylabel("camera-t Y (left)")
        ax.set_zlabel("camera-t Z (up)")
        # View from behind/right so the NWU-style camera axes read clearly.
        ax.view_init(elev=22.0, azim=-135.0)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return _figure_to_png_bytes(fig)
    except Exception as exc:
        logger.info(f"[goalwam eval] Failed to plot EEF 3D ({title}): {exc}")
        plt.close("all")
        return None
