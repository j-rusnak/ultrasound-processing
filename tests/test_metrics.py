import pytest
import torch

from ultrasound_processing.evaluation.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r_squared,
    root_mean_squared_error,
)


def test_regression_metrics_match_known_values() -> None:
    y_true = torch.tensor([2.0, 4.0, 6.0])
    y_pred = torch.tensor([1.0, 5.0, 7.0])

    assert mean_absolute_error(y_true, y_pred).item() == pytest.approx(1.0)
    assert mean_squared_error(y_true, y_pred).item() == pytest.approx(1.0)
    assert root_mean_squared_error(y_true, y_pred).item() == pytest.approx(1.0)
    assert mean_absolute_percentage_error(y_true, y_pred).item() == pytest.approx(
        ((1 / 2) + (1 / 4) + (1 / 6)) / 3 * 100
    )
    assert r_squared(y_true, y_pred).item() == pytest.approx(0.625)
