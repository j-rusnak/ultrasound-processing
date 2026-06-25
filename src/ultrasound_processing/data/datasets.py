from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from ultrasound_processing.data.labels import LabelRecord
from ultrasound_processing.data.transforms import image_to_tensor


REGION_ORDER = ("A", "B", "Q")
REGION_PATTERNS = {
    "A": ("_A", "AB"),
    "B": ("_B", "BICEP"),
    "Q": ("_Q", "QUAD"),
}


def _patient_sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):012d}") if value.isdigit() else (1, value)


def _normalize_patient_ids(patient_ids: Iterable[str | int]) -> list[str]:
    return sorted((str(patient_id) for patient_id in patient_ids), key=_patient_sort_key)


class _BasePatientDataset(Dataset):
    def __init__(
        self,
        *,
        image_root: str | Path,
        labels: Mapping[int, LabelRecord],
        patient_ids: Iterable[str | int] | None = None,
        image_size: int = 256,
        number_of_images: int = 3,
        output: str = "FM",
        augment: int = 0,
        threshold: bool | int = False,
        speckle: bool = False,
        despeckle: bool = False,
        crop: Iterable[float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.image_root = Path(image_root)
        self.labels = dict(labels)
        self.image_size = int(image_size)
        self.number_of_images = int(number_of_images)
        self.output = output.upper()
        self.augment = int(augment)
        self.threshold = threshold
        self.speckle = speckle
        self.despeckle = despeckle
        self.crop = list(crop)
        if len(self.crop) != 3:
            raise ValueError("crop must contain abdomen, bicep, and quadriceps fractions")
        if self.output not in {"FM", "FFM"}:
            raise ValueError("output must be 'FM' or 'FFM'")

        if patient_ids is None:
            if not self.image_root.exists():
                raise FileNotFoundError(f"image root does not exist: {self.image_root}")
            patient_ids = [path.name for path in self.image_root.iterdir() if path.is_dir()]
        self.patients = [
            patient_id
            for patient_id in _normalize_patient_ids(patient_ids)
            if int(patient_id) in self.labels
        ]

    def __len__(self) -> int:
        return len(self.patients)

    def _label_record(self, patient_id: str) -> LabelRecord:
        return self.labels[int(patient_id)]

    def _target_tensor(self, patient_id: str) -> torch.Tensor:
        return torch.tensor([self._label_record(patient_id).target(self.output)], dtype=torch.float32)

    def _metadata_tensor(self, patient_id: str) -> torch.Tensor:
        label = self._label_record(patient_id)
        return torch.tensor([label.weight, label.length], dtype=torch.float32)

    def _crop_fraction(self, region: str) -> float:
        return self.crop[REGION_ORDER.index(region)]

    def _matching_files(self, patient_path: Path, region: str) -> list[Path]:
        patterns = tuple(pattern.upper() for pattern in REGION_PATTERNS[region])
        files = []
        for path in sorted(patient_path.iterdir()):
            name = path.name.upper()
            if path.is_file() and any(pattern in name for pattern in patterns):
                files.append(path)
        return files[: self.number_of_images]

    def _load_image(self, path: Path, *, region: str) -> torch.Tensor:
        image = Image.open(path)
        width, height = image.size
        crop_height = max(1, round(height * self._crop_fraction(region)))
        image = image.crop((0, 0, width, crop_height))
        if self.speckle:
            image = image.filter(ImageFilter.MedianFilter(size=5))
        if self.despeckle:
            try:
                import cv2
                import numpy as np
            except ImportError as exc:
                raise RuntimeError("despeckle=True requires opencv-python and numpy") from exc
            image_np = np.array(image.convert("RGB"))
            image_np = cv2.fastNlMeansDenoisingColored(
                image_np,
                None,
                h=10,
                templateWindowSize=7,
                searchWindowSize=21,
            )
            image = Image.fromarray(image_np)
        return image_to_tensor(image, image_size=self.image_size, threshold=self.threshold)

    def _augment_tensor(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        target_count = self.augment if self.augment > 0 else 1
        augmented = [tensor]
        candidates = [
            torch.flip(tensor, dims=[2]),
            torch.flip(tensor, dims=[1]),
            torch.rot90(tensor, k=1, dims=[1, 2]),
            torch.rot90(tensor, k=3, dims=[1, 2]),
            torch.rot90(tensor, k=2, dims=[1, 2]),
            torch.flip(torch.rot90(tensor, k=1, dims=[1, 2]), dims=[2]),
            torch.flip(torch.rot90(tensor, k=3, dims=[1, 2]), dims=[1]),
        ]
        while len(augmented) < target_count:
            augmented.append(candidates[(len(augmented) - 1) % len(candidates)])
        return augmented[:target_count]

    def _load_region(self, patient_path: Path, region: str) -> torch.Tensor:
        tensors: list[torch.Tensor] = []
        for path in self._matching_files(patient_path, region):
            tensors.extend(self._augment_tensor(self._load_image(path, region=region)))
        if not tensors:
            raise FileNotFoundError(f"no {region} images found in {patient_path}")
        return torch.stack(tensors)


class PatientDataset(_BasePatientDataset):
    def __init__(
        self,
        *,
        region_combination: str = "BAQ",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        regions = [region for region in REGION_ORDER if region in region_combination.upper()]
        if not regions:
            raise ValueError("region_combination must include at least one of A, B, or Q")
        self.regions = regions

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        patient_id = self.patients[idx]
        patient_path = self.image_root / patient_id
        images = [self._load_region(patient_path, region) for region in self.regions]
        return {
            "patient_id": patient_id,
            "images": torch.cat(images, dim=0),
            "target": self._target_tensor(patient_id),
            "metadata": self._metadata_tensor(patient_id),
        }


class MultiUNetPatientDataset(_BasePatientDataset):
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        patient_id = self.patients[idx]
        patient_path = self.image_root / patient_id
        return {
            "patient_id": patient_id,
            "images_a": self._load_region(patient_path, "A"),
            "images_b": self._load_region(patient_path, "B"),
            "images_q": self._load_region(patient_path, "Q"),
            "target": self._target_tensor(patient_id),
            "metadata": self._metadata_tensor(patient_id),
        }


TestPatientDataset = PatientDataset
MultiUNetTestPatientDataset = MultiUNetPatientDataset
