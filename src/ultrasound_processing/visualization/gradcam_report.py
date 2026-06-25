from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - float(np.min(heatmap))
    max_value = float(np.max(heatmap))
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap


def _jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    import matplotlib

    cmap = matplotlib.colormaps.get_cmap("jet")
    colored = cmap(_normalize_heatmap(heatmap))[..., :3]
    return (colored * 255).astype(np.uint8)


def save_overlay(
    *,
    image_path: str | Path,
    heatmap: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.45,
) -> None:
    image = Image.open(image_path).convert("RGB")
    heatmap_rgb = Image.fromarray(_jet_colormap(heatmap)).resize(image.size, Image.BILINEAR)
    overlay = Image.blend(image, heatmap_rgb, alpha=alpha)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output)


def write_html_report(entries: list[dict[str, Any]], *, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries:
        overlay = Path(str(entry["overlay_image"]))
        try:
            overlay_src = overlay.relative_to(output.parent)
        except ValueError:
            overlay_src = overlay
        overlay_href = str(overlay_src).replace("\\", "/")
        rows.append(
            "\n".join(
                [
                    "<section>",
                    f"<h2>Patient {html.escape(str(entry.get('patient_id', 'unknown')))} - {html.escape(str(entry.get('region', 'unknown')))}</h2>",
                    f"<p>Source: {html.escape(str(entry.get('source_image', '')))}</p>",
                    f'<img src="{html.escape(overlay_href)}" alt="Grad-CAM overlay" />',
                    "</section>",
                ]
            )
        )
    body = "\n".join(rows) if rows else "<p>No Grad-CAM entries were generated.</p>"
    output.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                '<meta charset="utf-8" />',
                "<title>Grad-CAM Report</title>",
                "<style>",
                "body { font-family: Arial, sans-serif; margin: 2rem; color: #1f2933; }",
                "section { margin-bottom: 2rem; }",
                "img { max-width: 520px; width: 100%; border: 1px solid #d5d9df; }",
                "p { color: #52606d; }",
                "</style>",
                "</head>",
                "<body>",
                "<h1>Grad-CAM Report</h1>",
                body,
                "</body>",
                "</html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
