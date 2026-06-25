# ultrasound-processing

PyTorch project structure for the human body composition ultrasound experiments that were originally developed in `Copy_of_Human_Body_Composition_FM_FFM.ipynb`.

## Setup

Create a Python environment, then install the project in editable mode:

```powershell
pip install -e ".[dev]"
```

The implementation is config driven. The default config points at local, gitignored data paths:

```text
data/cropped_images/
data/Data_11.5.23_modified.csv
artifacts/checkpoints/
artifacts/results/
```

## Required Google Drive Downloads

Download these from `Ultrasound Files- Minnesota + Boston Collaboration` in the project Google Drive:

```text
cropped_images/
Data_11.5.23_modified.csv
saved_models/
  *.pt
  MICCAI_models/
    *.pt
```

Place them locally as:

```text
data/cropped_images/
data/Data_11.5.23_modified.csv
artifacts/checkpoints/
```

The image folder should contain one subfolder per `Study_ID`. The label CSV must contain `Study_ID`, `FM`, `FFM`, `Weight_visit`, and `Length_visit`.

## Commands

Train:

```powershell
python scripts/train.py --config configs/experiments/unet_fm_baq.yaml
```

Evaluate:

```powershell
python scripts/evaluate.py --config configs/experiments/unet_fm_baq.yaml --checkpoint artifacts/checkpoints/model.pt
```

Predict:

```powershell
python scripts/predict.py --config configs/experiments/unet_fm_baq.yaml --checkpoint artifacts/checkpoints/model.pt
```

Grad-CAM:

```powershell
python scripts/gradcam.py --config configs/experiments/unet_fm_baq.yaml --checkpoint artifacts/checkpoints/model.pt --image data/cropped_images/58041919/58041919_AB1_R.jpg
```

Statistics:

```powershell
python scripts/run_stats.py --predictions artifacts/results/predictions.csv
```

## Notes

The original notebook is preserved under `notebooks/archive/` for provenance. New reusable code lives under `src/ultrasound_processing/`.
