from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultrasound_processing.config import load_config, resolve_project_path
from ultrasound_processing.pipeline import build_datasets, build_loader, configured_device, model_from_config
from ultrasound_processing.training.checkpoints import save_checkpoint
from ultrasound_processing.training.trainer import set_seed, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, default_path=ROOT / "configs" / "default.yaml")
    set_seed(int(config.get("seed", 42)))
    model = model_from_config(config)
    device = configured_device(config)
    if args.dry_run:
        print(json.dumps({"model": config["model"]["name"], "device": str(device), "dry_run": True}))
        return

    train_dataset, val_dataset = build_datasets(config, base_dir=ROOT)
    train_loader = build_loader(train_dataset, config, shuffle=True)
    val_loader = build_loader(val_dataset, config, shuffle=False)
    history = train_model(model, train_loader, val_loader, config, device=device)

    checkpoint_dir = resolve_project_path(config["artifacts"]["checkpoint_dir"], base_dir=ROOT)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"{config['model']['name']}_{config['data']['output']}_{stamp}.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        epoch=int(config["training"]["epochs"]),
        config=config,
        metrics={"train_loss": history["train_loss"][-1], "validation_loss": history["validation_loss"][-1]},
    )
    print(json.dumps({"checkpoint": str(checkpoint_path), "history": history}, indent=2))


if __name__ == "__main__":
    main()
