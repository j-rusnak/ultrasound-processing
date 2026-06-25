from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from ultrasound_processing.models.unet import UNet


class LinearRegression(nn.Module):
    def __init__(self, input_size: int = 3, output_size: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RegressionNN(nn.Module):
    def __init__(self, input_size: int = 3, output_size: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 7),
            nn.ReLU(),
            nn.Linear(7, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiUNetEnsemble(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 3,
        output_dim: int = 1,
        abdomen_model: nn.Module | None = None,
        bicep_model: nn.Module | None = None,
        quad_model: nn.Module | None = None,
        combiner: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.abdomen_model = abdomen_model or UNet(input_channels=input_channels, output_dim=output_dim)
        self.bicep_model = bicep_model or UNet(input_channels=input_channels, output_dim=output_dim)
        self.quad_model = quad_model or UNet(input_channels=input_channels, output_dim=output_dim)
        self.combiner = combiner or RegressionNN(input_size=3 * output_dim, output_size=output_dim)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        abdomen = self.abdomen_model(inputs["images_a"])
        bicep = self.bicep_model(inputs["images_b"])
        quad = self.quad_model(inputs["images_q"])
        return self.combiner(torch.cat([abdomen, bicep, quad], dim=1))
