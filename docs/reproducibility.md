# Reproducibility Workflow

This project should make each experiment traceable from config to prediction table.

## 1. Prepare Local Data

Place the private Google Drive export locally:

```text
data/cropped_images/
data/Data_11.5.23_modified.csv
artifacts/checkpoints/
```

The data and artifacts directories are gitignored.

## 2. Run a Config

Train a model:

```powershell
python scripts/train.py --config configs/experiments/umb_region_grid/unet_fm_baq.yaml
```

For region-grid runs, save the checkpoint to the fixed name expected by the grid runner:

```powershell
python scripts/train.py --config configs/experiments/umb_region_grid/unet_fm_baq.yaml --checkpoint-output artifacts/checkpoints/unet_fm_baq.pt
```

Evaluate a checkpoint:

```powershell
python scripts/evaluate.py --config configs/experiments/umb_region_grid/unet_fm_baq.yaml --checkpoint artifacts/checkpoints/unet_fm_baq.pt --output artifacts/results/region_grid/unet_fm_baq_predictions.csv
```

Both scripts append a JSONL record to:

```text
artifacts/results/runs.jsonl
```

Each run record includes the model name, output, region, metrics, artifact paths, split ID, config fingerprint, command, and current git commit when available.

## 3. Export UMB-Style Region Tables

After generating prediction CSVs for the region grid:

```powershell
python scripts/export_region_grid_results.py --predictions-dir artifacts/results/region_grid
```

This writes:

```text
artifacts/results/region_grid_metrics.csv
artifacts/results/region_grid_metrics.md
```

Prediction files should be named like:

```text
unet_fm_b_predictions.csv
unet_fm_a_predictions.csv
unet_fm_q_predictions.csv
unet_fm_ba_predictions.csv
unet_fm_bq_predictions.csv
unet_fm_aq_predictions.csv
unet_fm_baq_predictions.csv
unet_ffm_b_predictions.csv
...
```

## 4. Generate Grad-CAM Reports

Create an image list CSV with columns:

```text
patient_id,region,image_path
```

An example is provided at:

```text
docs/gradcam_image_list_example.csv
```

Generate an HTML report:

```powershell
python scripts/gradcam_report.py --config configs/experiments/unet_fm_baq.yaml --checkpoint artifacts/checkpoints/model.pt --image-list docs/gradcam_image_list_example.csv --target-layer outc
```

The report is written under:

```text
artifacts/results/gradcam_report/
```

## 5. Compare Against Papers Carefully

Paper-level reproduction depends on matching patient splits, selected images per anatomical site, preprocessing, model implementation, training schedule, and checkpoints. Use the paper metrics as benchmarks, not as guaranteed outputs from a different split or checkpoint.
