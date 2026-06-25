from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def ensure_batched_image_set(images: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    """Normalize image inputs to B x N x C x H x W."""

    if isinstance(images, torch.Tensor):
        if images.ndim == 5:
            return images.float()
        if images.ndim == 4:
            return images.unsqueeze(1).float()
        if images.ndim == 3:
            return images.unsqueeze(0).unsqueeze(0).float()
        raise ValueError(f"expected 3D, 4D, or 5D image tensor, got {images.ndim}D")

    tensors = []
    for image in images:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"expected each image to be 3D or 4D, got {image.ndim}D")
        tensors.append(image.float())
    if not tensors:
        raise ValueError("image sequence is empty")
    return torch.stack(tensors, dim=1)


def optional_metadata(
    *,
    metadata: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    length: torch.Tensor | None = None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if metadata is not None:
        if metadata.ndim == 1:
            metadata = metadata.unsqueeze(0)
        return metadata.float().to(device)
    if weight is None or length is None:
        return None
    weight = weight.reshape(batch_size, 1).float().to(device)
    length = length.reshape(batch_size, 1).float().to(device)
    return torch.cat([weight, length], dim=1)


class TinyConvBackbone(nn.Module):
    def __init__(self, input_channels: int = 3, feature_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.projection = nn.Linear(64, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


class ImageSetRegressor(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        feature_dim: int,
        output_dim: int = 1,
        metadata_dim: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.metadata_dim = metadata_dim
        self.regression = nn.Linear(feature_dim + metadata_dim, output_dim)

    def encode_images(self, image_set: torch.Tensor) -> torch.Tensor:
        batch_size, image_count, channels, height, width = image_set.shape
        flat = image_set.reshape(batch_size * image_count, channels, height, width)
        features = self.encoder(flat)
        return features.reshape(batch_size, image_count, -1).mean(dim=1)

    def forward(
        self,
        images: torch.Tensor | Sequence[torch.Tensor],
        weight: torch.Tensor | None = None,
        length: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_set = ensure_batched_image_set(images)
        features = self.encode_images(image_set)
        extra = optional_metadata(
            metadata=metadata,
            weight=weight,
            length=length,
            batch_size=features.shape[0],
            device=features.device,
        )
        if extra is not None:
            features = torch.cat([features, extra], dim=1)
        return self.regression(features)


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False
