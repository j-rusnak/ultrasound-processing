from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultrasound_processing.config import load_config, resolve_project_path
from ultrasound_processing.evaluation.predict import predict_batches, write_predictions
from ultrasound_processing.pipeline import build_loader, build_prediction_dataset, configured_device, model_from_config
from ultrasound_processing.training.checkpoints import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config, default_path=ROOT / "configs" / "default.yaml")
    model = model_from_config(config)
    device = configured_device(config)
    metadata = load_checkpoint(args.checkpoint, model=model, map_location=device, strict=False)
    if isinstance(metadata.get("model"), torch.nn.Module):
        model = metadata["model"]
    dataset = build_prediction_dataset(config, base_dir=ROOT)
    loader = build_loader(dataset, config, shuffle=False)
    rows = predict_batches(model, loader, device=device)
    output = Path(args.output) if args.output else resolve_project_path(config["artifacts"]["results_dir"], base_dir=ROOT) / "predictions.csv"
    write_predictions(rows, output)
    print(json.dumps({"predictions": str(output), "count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
