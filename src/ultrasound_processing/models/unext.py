from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ultrasound_processing.models.common import ensure_batched_image_set


class UNextS(nn.Module):
    def __init__(self, input_channels: int = 3, output_dim: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, 1, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.regression = nn.Linear(64, output_dim)

    def forward(self, images: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        image_set = ensure_batched_image_set(images)
        batch_size, image_count, channels, height, width = image_set.shape
        flat = image_set.reshape(batch_size * image_count, channels, height, width)
        encoded = self.encoder(flat)
        decoded = F.interpolate(self.decoder(encoded), size=(height, width), mode="bilinear", align_corners=False)
        features = self.pool(decoded).flatten(start_dim=1)
        features = features.reshape(batch_size, image_count, -1).mean(dim=1)
        return self.regression(features)


UNext_S = UNextS
