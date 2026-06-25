from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ultrasound_processing.models.common import ensure_batched_image_set, optional_metadata


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 1,
        *,
        base_channels: int = 16,
        metadata_dim: int = 0,
    ) -> None:
        super().__init__()
        self.metadata_dim = metadata_dim
        self.inc = DoubleConv(input_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.up1 = Up(base_channels * 12, base_channels * 4)
        self.up2 = Up(base_channels * 6, base_channels * 2)
        self.up3 = Up(base_channels * 3, base_channels)
        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.regression = nn.Linear(64 + metadata_dim, output_dim)

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return torch.sigmoid(self.outc(x))

    def forward(
        self,
        images: torch.Tensor | Sequence[torch.Tensor],
        weight: torch.Tensor | None = None,
        length: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_set = ensure_batched_image_set(images)
        batch_size, image_count, channels, height, width = image_set.shape
        flat = image_set.reshape(batch_size * image_count, channels, height, width)
        maps = self.forward_map(flat)
        features = self.pool(maps).flatten(start_dim=1)
        features = features.reshape(batch_size, image_count, -1).mean(dim=1)
        extra = optional_metadata(
            metadata=metadata,
            weight=weight,
            length=length,
            batch_size=batch_size,
            device=features.device,
        )
        if extra is not None:
            features = torch.cat([features, extra], dim=1)
        return self.regression(features)
