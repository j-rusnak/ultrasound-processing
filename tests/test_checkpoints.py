from pathlib import Path

import torch

from ultrasound_processing.models.factory import create_model
from ultrasound_processing.training.checkpoints import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_restores_model_weights(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    model = create_model("rnn", input_channels=3, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        config={"model": {"name": "rnn"}},
        metrics={"mae": 1.25},
    )

    restored = create_model("rnn", input_channels=3, output_dim=1)
    metadata = load_checkpoint(checkpoint_path, model=restored, map_location="cpu")

    assert metadata["epoch"] == 3
    assert metadata["config"]["model"]["name"] == "rnn"
    assert metadata["metrics"]["mae"] == 1.25
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
