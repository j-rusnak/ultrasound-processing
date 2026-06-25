from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ultrasound_processing.config import resolve_project_path
from ultrasound_processing.data.datasets import MultiUNetPatientDataset, PatientDataset
from ultrasound_processing.data.labels import load_labels
from ultrasound_processing.data.splits import split_patient_ids
from ultrasound_processing.models.factory import create_model


def model_from_config(config: dict[str, Any]) -> torch.nn.Module:
    model_config = dict(config["model"])
    name = model_config.pop("name")
    input_channels = int(model_config.pop("input_channels", 3))
    output_dim = int(model_config.pop("output_dim", 1))
    freeze_backbone = model_config.pop("freeze_backbone", None)
    normalized_name = name.lower().replace("-", "_").replace("+", "_plus")
    metadata_models = {
        "effnet_ft",
        "effnet_lp",
        "efficientnet_b1_finetune",
        "efficientnet_b1_linear_probe",
        "resnet18_ft",
        "resnet18_lp",
        "resnet18_finetune",
        "resnet18_linear_probe",
        "unet",
        "attunet",
        "attention_unet",
        "unet_plus_plus",
    }
    if config["training"].get("use_weight_length", False) and normalized_name in metadata_models:
        model_config.setdefault("metadata_dim", 2)
    return create_model(
        name,
        input_channels=input_channels,
        output_dim=output_dim,
        freeze_backbone=freeze_backbone,
        **model_config,
    )


def _patient_ids(image_root: Path) -> list[str]:
    return sorted(path.name for path in image_root.iterdir() if path.is_dir())


def _dataset_kwargs(config: dict[str, Any], patient_ids: list[str], *, base_dir: str | Path | None = None) -> dict[str, Any]:
    data = config["data"]
    image_root = resolve_project_path(data["image_root"], base_dir=base_dir)
    labels = load_labels(resolve_project_path(data["labels_csv"], base_dir=base_dir))
    return {
        "image_root": image_root,
        "labels": labels,
        "patient_ids": patient_ids,
        "image_size": int(data.get("image_size", 256)),
        "number_of_images": int(data.get("number_of_images", 3)),
        "output": data.get("output", "FM"),
        "augment": int(data.get("augment", 0)),
        "threshold": data.get("threshold", False),
        "speckle": bool(data.get("speckle", False)),
        "despeckle": bool(data.get("despeckle", False)),
        "crop": data.get("crop", [1.0, 1.0, 1.0]),
    }


def build_datasets(config: dict[str, Any], *, base_dir: str | Path | None = None):
    data = config["data"]
    image_root = resolve_project_path(data["image_root"], base_dir=base_dir)
    patients = _patient_ids(image_root)
    train_ids, val_ids = split_patient_ids(
        patients,
        train_fraction=float(data.get("train_fraction", 0.9)),
        seed=int(config.get("seed", 42)),
        test_patient_ids=data.get("test_patient_ids") or None,
    )

    if config["model"]["name"].lower().replace("-", "_") == "multi_unet":
        dataset_class = MultiUNetPatientDataset
        train = dataset_class(**_dataset_kwargs(config, train_ids, base_dir=base_dir))
        val = dataset_class(**_dataset_kwargs(config, val_ids, base_dir=base_dir))
    else:
        dataset_class = PatientDataset
        train = dataset_class(
            region_combination=data.get("region", "BAQ"),
            **_dataset_kwargs(config, train_ids, base_dir=base_dir),
        )
        val = dataset_class(
            region_combination=data.get("region", "BAQ"),
            **_dataset_kwargs(config, val_ids, base_dir=base_dir),
        )
    return train, val


def build_prediction_dataset(config: dict[str, Any], *, base_dir: str | Path | None = None):
    data = config["data"]
    image_root = resolve_project_path(data["image_root"], base_dir=base_dir)
    patients = data.get("test_patient_ids") or _patient_ids(image_root)
    if config["model"]["name"].lower().replace("-", "_") == "multi_unet":
        return MultiUNetPatientDataset(**_dataset_kwargs(config, patients, base_dir=base_dir))
    return PatientDataset(
        region_combination=data.get("region", "BAQ"),
        **_dataset_kwargs(config, patients, base_dir=base_dir),
    )


def build_loader(dataset, config: dict[str, Any], *, shuffle: bool = False) -> DataLoader:
    training = config["training"]
    return DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(training.get("num_workers", 0)),
    )


def configured_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "cpu"))
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)
