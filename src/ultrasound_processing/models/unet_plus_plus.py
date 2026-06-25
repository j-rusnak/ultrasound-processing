from __future__ import annotations

from ultrasound_processing.models.unet import UNet


class UNetPlusPlus(UNet):
    """UNet++ model slot with the same image-set regression interface."""


UNet_2Plus = UNetPlusPlus
