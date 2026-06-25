from __future__ import annotations

import csv
from pathlib import Path

import torch

from ultrasound_processing.evaluation.metrics import regression_summary


REGION_GRID_FIELDS = [
    "experiment",
    "model",
    "output",
    "region",
    "mae",
    "mse",
    "rmse",
    "mape",
    "r2",
    "bland_altman_bias",
    "bland_altman_lower_loa",
    "bland_altman_upper_loa",
]


def _parse_prediction_name(path: Path) -> tuple[str, str, str, str]:
    stem = path.stem
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"prediction filename must include model, output, and region: {path.name}")
    output = parts[-2].upper()
    region = parts[-1].upper()
    model = "_".join(parts[:-2])
    return stem, model, output, region


def summarize_prediction_file(path: str | Path) -> dict[str, float | str]:
    prediction_path = Path(path)
    predictions: list[float] = []
    targets: list[float] = []
    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            predictions.append(float(row["prediction"]))
            targets.append(float(row["target"]))

    if not predictions:
        raise ValueError(f"prediction file has no rows: {prediction_path}")

    experiment, model, output, region = _parse_prediction_name(prediction_path)
    metrics = regression_summary(
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(predictions, dtype=torch.float32),
    )
    return {
        "experiment": experiment,
        "model": model,
        "output": output,
        "region": region,
        **metrics,
    }


def collect_region_grid_metrics(
    predictions_dir: str | Path,
    *,
    pattern: str = "*_predictions.csv",
) -> list[dict[str, float | str]]:
    root = Path(predictions_dir)
    rows = [summarize_prediction_file(path) for path in sorted(root.glob(pattern))]
    output_order = {"FM": 0, "FFM": 1}
    region_order = {"B": 0, "A": 1, "Q": 2, "BA": 3, "BQ": 4, "AQ": 5, "BAQ": 6}
    return sorted(
        rows,
        key=lambda row: (
            str(row["model"]),
            output_order.get(str(row["output"]), 99),
            region_order.get(str(row["region"]), 99),
        ),
    )


def write_metrics_table(
    rows: list[dict[str, float | str]],
    *,
    csv_path: str | Path,
    markdown_path: str | Path | None = None,
) -> None:
    csv_output = Path(csv_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGION_GRID_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if markdown_path is not None:
        markdown_output = Path(markdown_path)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "| " + " | ".join(REGION_GRID_FIELDS) + " |",
            "| " + " | ".join(["---"] * len(REGION_GRID_FIELDS)) + " |",
        ]
        for row in rows:
            values = []
            for field in REGION_GRID_FIELDS:
                value = row[field]
                if isinstance(value, float):
                    value = f"{value:.4f}"
                values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
