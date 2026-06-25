# ultrasound-processing

PyTorch research pipeline for automated newborn body composition prediction from ultrasound images. The project estimates fat mass (FM) and fat-free mass (FFM) from biceps, abdomen, and quadriceps ultrasound scans, using air displacement plethysmography (ADP/Pea Pod) measurements as the clinical reference labels.

The codebase was split out of `Copy_of_Human_Body_Composition_FM_FFM.ipynb` and aligned with the prior papers in `../past-papers/`. The goal is to make the research reproducible: controlled configs, patient-wise data splits, reusable model definitions, checkpointed experiments, metrics, and Grad-CAM outputs.

## Research Goals

- Build portable, non-ionizing ultrasound models for infant nutritional and growth assessment.
- Predict both FM and FFM from multiple anatomical sites: biceps (`B`), abdomen (`A`), and quadriceps (`Q`).
- Compare image-region combinations such as `A`, `B`, `Q`, `BA`, `BQ`, `AQ`, and `BAQ`.
- Evaluate modified U-Net, Attention U-Net, CNN backbones, recurrent/image-set models, and multi-region ensembles.
- Report regression accuracy with MAE, MSE, RMSE, MAPE, and Bland-Altman agreement.
- Use Grad-CAM to check whether predictions rely on anatomically plausible regions such as subcutaneous fat and muscle layers.

## Research Context

The local paper context establishes the project direction:

- `UMB2025.pdf`: demonstrates ultrasound-based FM/FFM prediction in 65 preterm infants using 721 biceps, abdomen, and quadriceps images. The strongest reported FM result used all three regions (`BAQ`), while abdomen-only images were especially strong for FFM in that analysis.
- `EMBC2026_Attention_Mechanisms_for_Body_Composition_Prediction.pdf`: compares spatial attention (Attention U-Net) against global attention (ViT/MedViT) and non-attention baselines. It motivates Attention U-Net support and Grad-CAM interpretability in this repo.
- `RangerLab-IEEEAccess-2025.jpg`: frames the broader IEEE Access project as automated body composition prediction in newborns using ultrasound images.

See [docs/research_context.md](docs/research_context.md) for a fuller summary of the papers and how they map to repo features.

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

The image folder should contain one subfolder per `Study_ID`. Each patient should have ultrasound images from abdomen, biceps, and quadriceps when running full `BAQ` experiments. The label CSV must contain `Study_ID`, `FM`, `FFM`, `Weight_visit`, and `Length_visit`.

## Common Experiments

Train the baseline modified U-Net for FM from all three regions:

```powershell
python scripts/train.py --config configs/experiments/unet_fm_baq.yaml
```

Train the Attention U-Net experiment motivated by the attention-mechanism paper:

```powershell
python scripts/train.py --config configs/experiments/attention_unet_fm_baq.yaml
```

Train the abdomen-only FFM experiment motivated by the UMB anatomical-region analysis:

```powershell
python scripts/train.py --config configs/experiments/unet_ffm_abdomen.yaml
```

Evaluate a checkpoint:

```powershell
python scripts/evaluate.py --config configs/experiments/unet_fm_baq.yaml --checkpoint artifacts/checkpoints/model.pt
```

Generate predictions:

```powershell
python scripts/predict.py --config configs/experiments/unet_fm_baq.yaml --checkpoint artifacts/checkpoints/model.pt
```

Run Grad-CAM:

```powershell
python scripts/gradcam.py --config configs/experiments/unet_fm_baq.yaml --checkpoint artifacts/checkpoints/model.pt --image data/cropped_images/58041919/58041919_AB1_R.jpg
```

Summarize prediction metrics:

```powershell
python scripts/run_stats.py --predictions artifacts/results/predictions.csv
```

## Project Layout

```text
configs/                  Experiment configs for region/output/model choices
data/                     Local, gitignored image and label inputs
artifacts/                Local, gitignored checkpoints and result files
notebooks/archive/        Original monolithic notebook for provenance
scripts/                  Thin CLI entrypoints
src/ultrasound_processing/ Reusable package code
tests/                    Synthetic-data unit tests
```

## Notes

Do not commit private medical images, labels, checkpoints, or generated result files. Keep source papers outside the repo or add them only when licensing allows redistribution. The original notebook is preserved under `notebooks/archive/` for provenance; new reusable code should live under `src/ultrasound_processing/`.
