from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MAPELoss(nn.Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        denominator = torch.clamp(torch.abs(target), min=1e-8)
        return torch.mean(torch.abs((target - input) / denominator)) * 100


class CustomLoss(nn.Module):
    def __init__(self, weight_mse: float = 0.6, weight_l1: float = 0.4) -> None:
        super().__init__()
        self.weight_mse = weight_mse
        self.weight_l1 = weight_l1
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight_mse * self.mse_loss(input, target) + self.weight_l1 * self.l1_loss(input, target)


def non_negative_mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse_loss = nn.MSELoss()(output, target)
    penalty = torch.mean(F.relu(-output))
    return mse_loss + penalty


def get_loss(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "mse":
        return nn.MSELoss()
    if normalized == "mae":
        return nn.L1Loss()
    if normalized == "mape":
        return MAPELoss()
    if normalized == "custom":
        return CustomLoss()
    raise ValueError(f"unknown loss: {name}")
