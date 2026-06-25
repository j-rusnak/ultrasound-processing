from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from ultrasound_processing.training.losses import get_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    return batch


def model_inputs_from_batch(batch: Mapping[str, Any]) -> torch.Tensor | dict[str, torch.Tensor]:
    if {"images_a", "images_b", "images_q"}.issubset(batch.keys()):
        return {
            "images_a": batch["images_a"],
            "images_b": batch["images_b"],
            "images_q": batch["images_q"],
        }
    return batch["images"]


def _epoch(
    *,
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_metadata: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        target = batch["target"].float()
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        model_inputs = model_inputs_from_batch(batch)
        if use_metadata and "metadata" in batch:
            output = model(model_inputs, metadata=batch["metadata"]).view_as(target)
        else:
            output = model(model_inputs).view_as(target)
        loss = criterion(output, target)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().item())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    config: dict[str, Any],
    *,
    device: str | torch.device | None = None,
) -> dict[str, list[float]]:
    training = config["training"]
    device = torch.device(device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    criterion = get_loss(training.get("criterion", "mse"))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training.get("learning_rate", 0.001)))
    scheduler = None
    if training.get("adaptive_lr", True):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(training.get("scheduler_gamma", 0.95)),
        )

    history = {"train_loss": [], "validation_loss": []}
    for _ in range(int(training.get("epochs", 1))):
        history["train_loss"].append(
            _epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                use_metadata=bool(training.get("use_weight_length", False)),
                optimizer=optimizer,
            )
        )
        with torch.no_grad():
            history["validation_loss"].append(
                _epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    use_metadata=bool(training.get("use_weight_length", False)),
                )
            )
        if scheduler is not None:
            scheduler.step()
    return history
