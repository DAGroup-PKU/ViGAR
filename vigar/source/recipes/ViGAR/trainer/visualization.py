"""Inspectable action plots, GT/predicted future frames, and a local HTML gallery."""

import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_sample_visuals(output, sample_id, *, prediction, target, mask, layout, rgb, target_video, goal, caption, fps):
    import imageio.v3 as iio

    from .action_plotting import _plot_chassis_bev, _plot_eef_3d

    output = Path(output)
    prefix = f"sample_{sample_id:03d}"
    media = {}
    for name, group in (("left_eef_3d", layout.left_eef_pos), ("right_eef_3d", layout.right_eef_pos)):
        valid = mask[:, group].all(-1)
        if valid.any():
            png = _plot_eef_3d(prediction[valid], target[valid], group, name.replace("_", " ") + " — GT / prediction")
            if png is not None:
                path = output / f"{prefix}_{name}.png"
                path.write_bytes(png)
                media[name] = path.name
    chassis = slice(layout.chassis_xy.start, layout.chassis_angle + 1)
    valid = mask[:, chassis].all(-1)
    if valid.any():
        png = _plot_chassis_bev(prediction[valid], target[valid], layout)
        if png is not None:
            path = output / f"{prefix}_chassis_bev.png"
            path.write_bytes(png)
            media["chassis_bev"] = path.name
    pred_frames = rgb.permute(1, 2, 3, 0).numpy()
    gt_frames = target_video.permute(1, 2, 3, 0).numpy()
    h, w = pred_frames.shape[1:3]
    chosen = np.linspace(0, len(pred_frames) - 1, 5, dtype=int)
    sheet = Image.new("RGB", (w * len(chosen), 2 * (h + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (label, frames) in enumerate((("GT", gt_frames), ("Prediction", pred_frames))):
        for col, frame in enumerate(chosen):
            offset = row * (h + 28)
            draw.text(
                (col * w + 5, offset + 5),
                f"{label}: {'current' if frame == 0 else 'future'} +{frame / fps:.2f}s",
                fill="black",
            )
            sheet.paste(Image.fromarray(frames[frame]), (col * w, offset + 28))
    path = output / f"{prefix}_future_frames.png"
    sheet.save(path)
    media["future_frames"] = path.name
    movie = []
    for t, (gt, pred) in enumerate(zip(gt_frames, pred_frames, strict=True)):
        frame = Image.new("RGB", (w * 2, h + 28), "white")
        frame.paste(Image.fromarray(gt), (0, 28))
        frame.paste(Image.fromarray(pred), (w, 28))
        draw = ImageDraw.Draw(frame)
        draw.text((5, 5), f"GT +{t / fps:.2f}s", fill="black")
        draw.text((w + 5, 5), "Prediction", fill="black")
        movie.append(np.asarray(frame))
    path = output / f"{prefix}_future_frames.mp4"
    iio.imwrite(path, np.stack(movie), fps=fps, codec="libx264", macro_block_size=1)
    media["future_video"] = path.name
    condition = Image.new("RGB", (w * 2, h + 28), "white")
    condition.paste(Image.fromarray(gt_frames[0]), (0, 28))
    condition.paste(Image.fromarray(goal[:, 0].permute(1, 2, 0).numpy()), (w, 28))
    draw = ImageDraw.Draw(condition)
    draw.text((5, 5), "Current observation", fill="black")
    draw.text((w + 5, 5), "Oracle terminal goal", fill="black")
    path = output / f"{prefix}_conditioning.png"
    condition.save(path)
    media["conditioning"] = path.name
    return media


def write_gallery(output, rows, iteration, weights):
    parts = [
        "<!doctype html><meta charset='utf-8'><title>ViGAR evaluation</title>",
        "<style>body{font:16px sans-serif;max-width:1600px;margin:24px auto;padding:0 16px}img,video{max-width:100%}.plots img{max-width:48%}pre{white-space:pre-wrap}section{border-top:1px solid #bbb;padding:24px 0}</style>",
        f"<h1>ViGAR evaluation: update {iteration}, {html.escape(weights)}</h1>",
        "<p>Ground truth and predictions use matching future timestamps. EEF plots show deltas in the anchor camera frame. Unavailable actions are omitted.</p>",
    ]
    for row in rows:
        if not row.get("media"):
            continue
        parts.append(
            f"<section><h2>Sample {row['sample_id']}: {html.escape(row['dataset'])}</h2><p>{html.escape(row['caption'])}</p>"
        )
        media = row["media"]
        for name in ("conditioning", "future_frames"):
            parts.append(f"<img loading='lazy' alt='{name}' src='{html.escape(media[name])}'>")
        parts.append(
            f"<video controls preload='none' src='{html.escape(media['future_video'])}'></video><div class='plots'>"
        )
        for name in ("left_eef_3d", "right_eef_3d", "chassis_bev"):
            if name in media:
                parts.append(f"<img loading='lazy' alt='{name}' src='{html.escape(media[name])}'>")
        parts.append("</div><pre>" + html.escape(json.dumps(row["metrics"], indent=2)) + "</pre></section>")
    (Path(output) / "index.html").write_text("\n".join(parts))
