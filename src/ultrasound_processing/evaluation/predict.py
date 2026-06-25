from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from ultrasound_processing.training.trainer import move_batch_to_device, model_inputs_from_batch


def predict_batches(model: torch.nn.Module, loader, *, device: str | torch.device = "cpu") -> list[dict[str, Any]]:
    device = torch.device(device)
    model.to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(model_inputs_from_batch(batch)).detach().cpu().view(-1)
            targets = batch["target"].detach().cpu().view(-1)
            patient_ids = batch["patient_id"]
            if isinstance(patient_ids, str):
                patient_ids = [patient_ids]
            for patient_id, prediction, target in zip(patient_ids, outputs, targets):
                rows.append(
                    {
                        "patient_id": str(patient_id),
                        "prediction": float(prediction.item()),
                        "target": float(target.item()),
                    }
                )
    return rows


def write_predictions(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "prediction", "target"])
        writer.writeheader()
        writer.writerows(rows)
