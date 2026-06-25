from __future__ import annotations

from ultrasound_processing.models.unet import UNet


class AttentionUNet(UNet):
    """UNet-compatible attention model slot.

    The notebook used this architecture as a regression head over ultrasound
    image sets. This class preserves the public model option and tensor
    contract while sharing the portable UNet regression path.
    """


AttUNet = AttentionUNet
