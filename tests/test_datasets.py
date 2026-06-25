from pathlib import Path

import torch
from PIL import Image

from ultrasound_processing.data.datasets import MultiUNetPatientDataset, PatientDataset
from ultrasound_processing.data.labels import LabelRecord


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), color=(120, 80, 40)).save(path)


def test_patient_dataset_returns_stacked_images_and_target(tmp_path: Path) -> None:
    root = tmp_path / "cropped_images"
    patient_dir = root / "101"
    _write_image(patient_dir / "101_AB1_R.jpg")
    _write_image(patient_dir / "101_BICEP1_R.jpg")
    _write_image(patient_dir / "101_QUAD1_R.jpg")
    labels = {101: LabelRecord(fm=12.0, ffm=22.0, weight=7.0, length=55.0)}

    dataset = PatientDataset(
        image_root=root,
        labels=labels,
        patient_ids=["101"],
        image_size=16,
        region_combination="BAQ",
        number_of_images=1,
        output="FM",
    )

    sample = dataset[0]

    assert sample["patient_id"] == "101"
    assert sample["images"].shape == (3, 3, 16, 16)
    assert torch.equal(sample["target"], torch.tensor([12.0]))
    assert torch.equal(sample["metadata"], torch.tensor([7.0, 55.0]))


def test_multi_unet_dataset_keeps_regions_separate(tmp_path: Path) -> None:
    root = tmp_path / "cropped_images"
    patient_dir = root / "101"
    _write_image(patient_dir / "101_AB1_R.jpg")
    _write_image(patient_dir / "101_BICEP1_R.jpg")
    _write_image(patient_dir / "101_QUAD1_R.jpg")
    labels = {101: LabelRecord(fm=12.0, ffm=22.0, weight=7.0, length=55.0)}

    dataset = MultiUNetPatientDataset(
        image_root=root,
        labels=labels,
        patient_ids=["101"],
        image_size=16,
        number_of_images=1,
        output="FFM",
    )

    sample = dataset[0]

    assert sample["images_a"].shape == (1, 3, 16, 16)
    assert sample["images_b"].shape == (1, 3, 16, 16)
    assert sample["images_q"].shape == (1, 3, 16, 16)
    assert torch.equal(sample["target"], torch.tensor([22.0]))
