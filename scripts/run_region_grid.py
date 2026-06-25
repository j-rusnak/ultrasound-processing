from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiments" / "umb_region_grid"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    parser.add_argument("--results-dir", default="artifacts/results/region_grid")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    configs = sorted(CONFIG_ROOT.glob("*.yaml"))
    if not configs:
        raise SystemExit(f"no configs found under {CONFIG_ROOT}")
    if not args.train and not args.evaluate:
        args.evaluate = True

    for config in configs:
        experiment = config.stem
        if args.train:
            checkpoint = Path(args.checkpoint_dir) / f"{experiment}.pt"
            command = [
                sys.executable,
                "scripts/train.py",
                "--config",
                str(config.relative_to(ROOT)),
                "--checkpoint-output",
                str(checkpoint),
            ]
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
        if args.evaluate:
            checkpoint = Path(args.checkpoint_dir) / f"{experiment}.pt"
            output = Path(args.results_dir) / f"{experiment}_predictions.csv"
            command = [
                sys.executable,
                "scripts/evaluate.py",
                "--config",
                str(config.relative_to(ROOT)),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
            ]
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
