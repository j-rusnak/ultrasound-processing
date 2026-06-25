from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultrasound_processing.config import load_config
from ultrasound_processing.pipeline import configured_device, model_from_config
from ultrasound_processing.training.checkpoints import load_checkpoint
from ultrasound_processing.visualization.gradcam import apply_grad_cam
from ultrasound_processing.visualization.gradcam_report import save_overlay, write_html_report


def _get_layer(model: torch.nn.Module, dotted_path: str) -> torch.nn.Module:
    layer = model
    for part in dotted_path.split("."):
        layer = getattr(layer, part)
    return layer


def _read_image_list(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--image-list")
    parser.add_argument("--target-layer", default="outc")
    parser.add_argument("--output-dir", default="artifacts/results/gradcam_report")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, default_path=ROOT / "configs" / "default.yaml")
    output_dir = ROOT / args.output_dir
    if args.dry_run:
        print(json.dumps({"model": config["model"]["name"], "output_dir": args.output_dir, "dry_run": True}))
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --dry-run is used")
    if not args.image_list:
        raise SystemExit("--image-list is required unless --dry-run is used")

    model = model_from_config(config)
    device = configured_device(config)
    metadata = load_checkpoint(args.checkpoint, model=model, map_location=device, strict=False)
    if isinstance(metadata.get("model"), torch.nn.Module):
        model = metadata["model"]
    target_layer = _get_layer(model, args.target_layer)

    entries = []
    for row in _read_image_list(Path(args.image_list)):
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        patient_id = row.get("patient_id") or image_path.parent.name
        region = row.get("region") or "unknown"
        overlay_path = output_dir / f"{patient_id}_{region}_gradcam.png"
        heatmap = apply_grad_cam(
            image_path,
            model,
            target_layer,
            image_size=int(config["data"].get("image_size", 256)),
            device=device,
        )
        save_overlay(image_path=image_path, heatmap=heatmap, output_path=overlay_path)
        entries.append(
            {
                "patient_id": patient_id,
                "region": region,
                "source_image": str(image_path),
                "overlay_image": str(overlay_path),
            }
        )

    report_path = output_dir / "index.html"
    write_html_report(entries, output_path=report_path)
    print(json.dumps({"report": str(report_path), "overlays": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
