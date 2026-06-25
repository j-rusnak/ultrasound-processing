import pytest
import torch

from ultrasound_processing.models.factory import create_model


@pytest.mark.parametrize(
    "model_name",
    [
        "efficientnet_b1_finetune",
        "efficientnet_b1_linear_probe",
        "resnet18_finetune",
        "resnet18_linear_probe",
        "unet",
        "attention_unet",
        "unet_plus_plus",
        "rnn",
        "unext_s",
    ],
)
def test_model_factory_models_accept_batched_image_sets(model_name: str) -> None:
    torch.manual_seed(0)
    model = create_model(model_name, input_channels=3, output_dim=1)
    model.eval()
    images = torch.randn(2, 2, 3, 32, 32)

    with torch.no_grad():
        output = model(images)

    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()


def test_multi_unet_ensemble_accepts_separate_regions() -> None:
    torch.manual_seed(0)
    model = create_model("multi_unet", input_channels=3, output_dim=1)
    model.eval()
    images = {
        "images_a": torch.randn(2, 1, 3, 32, 32),
        "images_b": torch.randn(2, 1, 3, 32, 32),
        "images_q": torch.randn(2, 1, 3, 32, 32),
    }

    with torch.no_grad():
        output = model(images)

    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()
