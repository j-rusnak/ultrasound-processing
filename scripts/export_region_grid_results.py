from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultrasound_processing.evaluation.region_grid import collect_region_grid_metrics, write_metrics_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", default="artifacts/results/region_grid")
    parser.add_argument("--pattern", default="*_predictions.csv")
    parser.add_argument("--output-csv", default="artifacts/results/region_grid_metrics.csv")
    parser.add_argument("--output-md", default="artifacts/results/region_grid_metrics.md")
    args = parser.parse_args()

    rows = collect_region_grid_metrics(ROOT / args.predictions_dir, pattern=args.pattern)
    write_metrics_table(rows, csv_path=ROOT / args.output_csv, markdown_path=ROOT / args.output_md)
    print(json.dumps({"rows": len(rows), "csv": args.output_csv, "markdown": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
