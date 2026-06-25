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
from ultrasound_processing.evaluation.statistics import summarize_predictions
from ultrasound_processing.pipeline import build_loader, build_prediction_dataset, configured_device, model_from_config
from ultrasound_processing.tracking import append_run_record, current_git_commit, default_split_id
from ultrasound_processing.training.checkpoints import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, default_path=ROOT / "configs" / "default.yaml")
    model = model_from_config(config)
    device = configured_device(config)
    if args.dry_run:
        print(json.dumps({"model": config["model"]["name"], "device": str(device), "dry_run": True}))
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --dry-run is used")

    metadata = load_checkpoint(args.checkpoint, model=model, map_location=device, strict=False)
    if isinstance(metadata.get("model"), torch.nn.Module):
        model = metadata["model"]
    dataset = build_prediction_dataset(config, base_dir=ROOT)
    loader = build_loader(dataset, config, shuffle=False)
    rows = predict_batches(model, loader, device=device)
    output = Path(args.output) if args.output else resolve_project_path(config["artifacts"]["results_dir"], base_dir=ROOT) / "predictions.csv"
    write_predictions(rows, output)
    metrics = summarize_predictions(output)
    run_log = resolve_project_path(config["artifacts"]["results_dir"], base_dir=ROOT) / "runs.jsonl"
    record = append_run_record(
        run_log,
        event="evaluate",
        config=config,
        metrics=metrics,
        checkpoint_path=args.checkpoint,
        predictions_path=output,
        split_id=default_split_id(config),
        command=" ".join(sys.argv),
        git_commit=current_git_commit(ROOT),
    )
    print(json.dumps({"predictions": str(output), "metrics": metrics, "run_id": record["run_id"]}, indent=2))


if __name__ == "__main__":
    main()
