from __future__ import annotations

from typing import Any

from torch import nn

from ultrasound_processing.models.attention_unet import AttentionUNet
from ultrasound_processing.models.efficientnet import EfficientNetB1Model
from ultrasound_processing.models.ensemble import MultiUNetEnsemble
from ultrasound_processing.models.resnet import ResNet18Model
from ultrasound_processing.models.rnn import RNN
from ultrasound_processing.models.unet import UNet
from ultrasound_processing.models.unet_plus_plus import UNetPlusPlus
from ultrasound_processing.models.unext import UNextS


def _normalize(name: str) -> str:
    return name.lower().replace("-", "_").replace("+", "_plus").replace(" ", "_")


def create_model(
    name: str,
    *,
    input_channels: int = 3,
    output_dim: int = 1,
    freeze_backbone: bool | None = None,
    **kwargs: Any,
) -> nn.Module:
    normalized = _normalize(name)
    if normalized in {"effnet_ft", "efficientnet_b1_finetune"}:
        return EfficientNetB1Model(
            input_channels=input_channels,
            output_dim=output_dim,
            freeze_backbone=False if freeze_backbone is None else freeze_backbone,
            **kwargs,
        )
    if normalized in {"effnet_lp", "efficientnet_b1_linear_probe"}:
        return EfficientNetB1Model(
            input_channels=input_channels,
            output_dim=output_dim,
            freeze_backbone=True if freeze_backbone is None else freeze_backbone,
            **kwargs,
        )
    if normalized in {"resnet18_ft", "resnet18_finetune"}:
        return ResNet18Model(
            input_channels=input_channels,
            output_dim=output_dim,
            freeze_backbone=False if freeze_backbone is None else freeze_backbone,
            **kwargs,
        )
    if normalized in {"resnet18_lp", "resnet18_linear_probe"}:
        return ResNet18Model(
            input_channels=input_channels,
            output_dim=output_dim,
            freeze_backbone=True if freeze_backbone is None else freeze_backbone,
            **kwargs,
        )
    if normalized == "unet":
        return UNet(input_channels=input_channels, output_dim=output_dim, **kwargs)
    if normalized in {"attunet", "attention_unet"}:
        return AttentionUNet(input_channels=input_channels, output_dim=output_dim, **kwargs)
    if normalized in {"unet_plus_plus", "unetplusplus", "unet__plus__plus"}:
        return UNetPlusPlus(input_channels=input_channels, output_dim=output_dim, **kwargs)
    if normalized == "rnn":
        return RNN(input_channels=input_channels, output_dim=output_dim, **kwargs)
    if normalized in {"unext_s", "unexts"}:
        return UNextS(input_channels=input_channels, output_dim=output_dim, **kwargs)
    if normalized == "multi_unet":
        return MultiUNetEnsemble(input_channels=input_channels, output_dim=output_dim, **kwargs)
    raise ValueError(f"unknown model: {name}")
