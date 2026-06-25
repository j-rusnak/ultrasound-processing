from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultrasound_processing.config import load_config, resolve_project_path
from ultrasound_processing.pipeline import configured_device, model_from_config
from ultrasound_processing.training.checkpoints import load_checkpoint
from ultrasound_processing.visualization.gradcam import apply_grad_cam


def _get_layer(model: torch.nn.Module, dotted_path: str) -> torch.nn.Module:
    layer = model
    for part in dotted_path.split("."):
        layer = getattr(layer, part)
    return layer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--target-layer", default="outc")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config, default_path=ROOT / "configs" / "default.yaml")
    model = model_from_config(config)
    device = configured_device(config)
    metadata = load_checkpoint(args.checkpoint, model=model, map_location=device, strict=False)
    if isinstance(metadata.get("model"), torch.nn.Module):
        model = metadata["model"]

    heatmap = apply_grad_cam(
        args.image,
        model,
        _get_layer(model, args.target_layer),
        image_size=int(config["data"].get("image_size", 256)),
        device=device,
    )
    output = Path(args.output) if args.output else resolve_project_path(config["artifacts"]["results_dir"], base_dir=ROOT) / "gradcam.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output, heatmap, cmap="jet")
    print(output)


if __name__ == "__main__":
    main()
