from __future__ import annotations

import csv
from pathlib import Path

import torch

from ultrasound_processing.evaluation.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r_squared,
    root_mean_squared_error,
)


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
    return {
        "mae": float(mean_absolute_error(y_true, y_pred).item()),
        "mse": float(mean_squared_error(y_true, y_pred).item()),
        "rmse": float(root_mean_squared_error(y_true, y_pred).item()),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred).item()),
        "r2": float(r_squared(y_true, y_pred).item()),
    }
