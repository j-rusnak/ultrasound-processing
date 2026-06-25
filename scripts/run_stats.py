from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultrasound_processing.evaluation.statistics import summarize_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_predictions(args.predictions), indent=2))


if __name__ == "__main__":
    main()
