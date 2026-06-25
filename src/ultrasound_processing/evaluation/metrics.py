from __future__ import annotations

import torch


def _as_float_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)


def mean_absolute_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true = _as_float_tensor(y_true)
    y_pred = _as_float_tensor(y_pred)
    return torch.mean(torch.abs(y_true - y_pred))


def mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true = _as_float_tensor(y_true)
    y_pred = _as_float_tensor(y_pred)
    return torch.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_percentage_error(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y_true = _as_float_tensor(y_true)
    y_pred = _as_float_tensor(y_pred)
    denominator = torch.clamp(torch.abs(y_true), min=eps)
    return torch.mean(torch.abs((y_true - y_pred) / denominator)) * 100


def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true = _as_float_tensor(y_true)
    y_pred = _as_float_tensor(y_pred)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if torch.isclose(ss_total, torch.tensor(0.0, device=ss_total.device)):
        return torch.tensor(0.0, device=ss_total.device)
    return 1 - (ss_residual / ss_total)


def bland_altman_stats(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    y_true = _as_float_tensor(y_true)
    y_pred = _as_float_tensor(y_pred)
    differences = y_pred - y_true
    bias = torch.mean(differences)
    sd = torch.std(differences, unbiased=False)
    return {
        "bias": float(bias.item()),
        "lower_loa": float((bias - 1.96 * sd).item()),
        "upper_loa": float((bias + 1.96 * sd).item()),
    }


def regression_summary(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    bland_altman = bland_altman_stats(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred).item()),
        "mse": float(mean_squared_error(y_true, y_pred).item()),
        "rmse": float(root_mean_squared_error(y_true, y_pred).item()),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred).item()),
        "r2": float(r_squared(y_true, y_pred).item()),
        "bland_altman_bias": bland_altman["bias"],
        "bland_altman_lower_loa": bland_altman["lower_loa"],
        "bland_altman_upper_loa": bland_altman["upper_loa"],
    }
