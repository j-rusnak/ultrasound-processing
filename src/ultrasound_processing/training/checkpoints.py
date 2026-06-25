from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    config: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "config": config or {},
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, output_path)


def _torch_load(path: Path, map_location: str | torch.device | None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    checkpoint = _torch_load(checkpoint_path, map_location)

    if isinstance(checkpoint, nn.Module):
        return {"model": checkpoint, "epoch": None, "config": {}, "metrics": {}}

    if not isinstance(checkpoint, dict):
        raise ValueError(f"unsupported checkpoint format: {checkpoint_path}")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if model is not None:
        model.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint.get("epoch"),
        "config": checkpoint.get("config", {}),
        "metrics": checkpoint.get("metrics", {}),
    }
