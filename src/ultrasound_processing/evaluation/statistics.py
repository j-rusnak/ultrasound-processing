from __future__ import annotations

import csv
from pathlib import Path

import torch

from ultrasound_processing.evaluation.metrics import regression_summary


def summarize_predictions(path: str | Path) -> dict[str, float]:
    predictions: list[float] = []
    targets: list[float] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            predictions.append(float(row["prediction"]))
            targets.append(float(row["target"]))

    y_pred = torch.tensor(predictions, dtype=torch.float32)
    y_true = torch.tensor(targets, dtype=torch.float32)
    return regression_summary(y_true, y_pred)
