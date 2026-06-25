from pathlib import Path

import numpy as np
from PIL import Image

from ultrasound_processing.visualization.gradcam_report import save_overlay, write_html_report


def test_save_overlay_and_write_html_report(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    overlay_path = tmp_path / "overlay.png"
    report_path = tmp_path / "report.html"
    Image.new("RGB", (16, 16), color=(100, 80, 60)).save(image_path)
    heatmap = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)

    save_overlay(image_path=image_path, heatmap=heatmap, output_path=overlay_path)
    write_html_report(
        [
            {
                "patient_id": "101",
                "region": "A",
                "source_image": str(image_path),
                "overlay_image": str(overlay_path),
            }
        ],
        output_path=report_path,
    )

    assert overlay_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "Grad-CAM Report" in html
    assert "101" in html
    assert "overlay.png" in html
