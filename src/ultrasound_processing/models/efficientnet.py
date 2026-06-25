from __future__ import annotations

from torch import nn

from ultrasound_processing.models.common import ImageSetRegressor, TinyConvBackbone, freeze_module

try:
    from torchvision import models as tv_models
except Exception:
    tv_models = None


def _efficientnet_b1_encoder(input_channels: int) -> tuple[nn.Module, int]:
    if tv_models is None or input_channels != 3:
        encoder = TinyConvBackbone(input_channels=input_channels, feature_dim=128)
        return encoder, encoder.feature_dim
    try:
        model = tv_models.efficientnet_b1(weights=None)
    except TypeError:
        model = tv_models.efficientnet_b1(pretrained=False)
    feature_dim = model.classifier[1].in_features
    model.classifier = nn.Identity()
    return model, feature_dim


class EfficientNetB1Model(ImageSetRegressor):
    def __init__(
        self,
        *,
        input_channels: int = 3,
        output_dim: int = 1,
        freeze_backbone: bool = False,
        metadata_dim: int = 0,
    ) -> None:
        encoder, feature_dim = _efficientnet_b1_encoder(input_channels)
        if freeze_backbone:
            freeze_module(encoder)
        super().__init__(
            encoder=encoder,
            feature_dim=feature_dim,
            output_dim=output_dim,
            metadata_dim=metadata_dim,
        )
