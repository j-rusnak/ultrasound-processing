from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image


def _as_triplet(value: float | Sequence[float]) -> torch.Tensor:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = [float(value)] * 3
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError("normalization values must contain 1 or 3 entries")
    return torch.tensor(values, dtype=torch.float32).view(3, 1, 1)


def image_to_tensor(
    image: Image.Image,
    *,
    image_size: int = 256,
    normalize_mean: float | Sequence[float] = 0.5,
    normalize_std: float | Sequence[float] = 0.5,
    threshold: int | float | bool = False,
) -> torch.Tensor:
    """Resize a PIL image and convert it to a normalized CHW float tensor."""

    image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).contiguous()
    if threshold is not False:
        tensor = torch.where(tensor < float(threshold) / 255.0, torch.zeros_like(tensor), tensor)
    mean = _as_triplet(normalize_mean)
    std = _as_triplet(normalize_std)
    return (tensor - mean) / std
