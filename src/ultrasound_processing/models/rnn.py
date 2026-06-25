from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ultrasound_processing.models.common import ensure_batched_image_set


class RNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 1,
        *,
        pooled_size: int = 32,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.pooled_size = pooled_size
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        self.rnn = nn.LSTM(
            input_size=input_channels * pooled_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, images: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        image_set = ensure_batched_image_set(images)
        batch_size, image_count, channels, height, width = image_set.shape
        flat = image_set.reshape(batch_size * image_count, channels, height, width)
        pooled = self.pool(flat)
        sequence = pooled.permute(0, 2, 1, 3).reshape(batch_size * image_count, self.pooled_size, -1)
        encoded, _ = self.rnn(sequence)
        per_image = self.fc(encoded[:, -1, :])
        return per_image.reshape(batch_size, image_count, -1).mean(dim=1)
