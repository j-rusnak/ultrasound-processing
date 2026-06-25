from pathlib import Path

import pytest

from ultrasound_processing.evaluation.region_grid import collect_region_grid_metrics, write_metrics_table


def test_collect_region_grid_metrics_from_prediction_files(tmp_path: Path) -> None:
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    (predictions_dir / "unet_fm_baq_predictions.csv").write_text(
        "\n".join(
            [
                "patient_id,prediction,target",
                "101,1.0,1.5",
                "202,2.0,1.5",
            ]
        ),
        encoding="utf-8",
    )

    rows = collect_region_grid_metrics(predictions_dir)

    assert len(rows) == 1
    row = rows[0]
    assert row["experiment"] == "unet_fm_baq"
    assert row["model"] == "unet"
    assert row["output"] == "FM"
    assert row["region"] == "BAQ"
    assert row["mae"] == pytest.approx(0.5)
    assert row["mse"] == pytest.approx(0.25)
    assert row["rmse"] == pytest.approx(0.5)
    assert row["mape"] == pytest.approx(33.33333333333333)
    assert row["r2"] == pytest.approx(0.0)
    assert row["bland_altman_bias"] == pytest.approx(0.0)
    assert row["bland_altman_lower_loa"] == pytest.approx(-0.98)
    assert row["bland_altman_upper_loa"] == pytest.approx(0.98)


def test_write_metrics_table_creates_csv_and_markdown(tmp_path: Path) -> None:
    rows = [
        {
            "experiment": "unet_fm_a",
            "model": "unet",
            "output": "FM",
            "region": "A",
            "mae": 0.1,
            "mse": 0.01,
            "rmse": 0.1,
            "mape": 2.0,
            "r2": 0.9,
            "bland_altman_bias": 0.01,
            "bland_altman_lower_loa": -0.1,
            "bland_altman_upper_loa": 0.1,
        }
    ]
    csv_path = tmp_path / "metrics.csv"
    md_path = tmp_path / "metrics.md"

    write_metrics_table(rows, csv_path=csv_path, markdown_path=md_path)

    assert "experiment,model,output,region" in csv_path.read_text(encoding="utf-8")
    assert "| experiment | model | output | region |" in md_path.read_text(encoding="utf-8")
