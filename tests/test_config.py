from pathlib import Path

import pytest

from ultrasound_processing.config import ConfigError, load_config


def test_load_config_merges_default_and_experiment(tmp_path: Path) -> None:
    default = tmp_path / "default.yaml"
    experiment = tmp_path / "experiment.yaml"

    default.write_text(
        """
seed: 42
device: cpu
data:
  image_root: data/cropped_images
  labels_csv: data/Data_11.5.23_modified.csv
  region: BAQ
model:
  name: unet
training:
  epochs: 90
  batch_size: 1
artifacts:
  checkpoint_dir: artifacts/checkpoints
  results_dir: artifacts/results
""".strip(),
        encoding="utf-8",
    )
    experiment.write_text(
        """
data:
  region: A
training:
  epochs: 2
""".strip(),
        encoding="utf-8",
    )

    config = load_config(experiment, default_path=default)

    assert config["seed"] == 42
    assert config["data"]["image_root"] == "data/cropped_images"
    assert config["data"]["region"] == "A"
    assert config["training"]["epochs"] == 2
    assert config["training"]["batch_size"] == 1


def test_load_config_rejects_missing_required_sections(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="missing required section"):
        load_config(config)
