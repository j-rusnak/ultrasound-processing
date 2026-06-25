from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultrasound_processing.data.transforms import image_to_tensor


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output

    def _save_gradient(self, module, input_grad, output_grad) -> None:
        self.gradients = output_grad[0]

    def __call__(self, images: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        output = self.model(images)
        output.sum().backward(retain_graph=True)
        if self.gradients is None or self.activations is None:
            raise RuntimeError("target layer did not produce Grad-CAM activations")

        gradients = self.gradients.detach()
        activations = self.activations.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activations).sum(dim=1))
        heatmap = torch.nn.functional.interpolate(
            heatmap.unsqueeze(1),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / torch.clamp(heatmap.max(), min=1e-8)
        return heatmap.cpu().numpy()


def load_image_tensor(path: str | Path, *, image_size: int = 256) -> torch.Tensor:
    image = Image.open(path)
    return image_to_tensor(image, image_size=image_size).unsqueeze(0).unsqueeze(0)


def apply_grad_cam(
    image_path: str | Path,
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    *,
    image_size: int = 256,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    device = torch.device(device)
    model.to(device)
    model.eval()
    images = load_image_tensor(image_path, image_size=image_size).to(device)
    grad_cam = GradCAM(model, target_layer)
    try:
        return grad_cam(images)
    finally:
        grad_cam.close()
