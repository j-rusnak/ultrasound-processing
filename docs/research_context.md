# Research Context

This repo supports automated newborn body composition prediction from portable ultrasound images. The target outputs are fat mass (FM) and fat-free mass (FFM), with air displacement plethysmography (ADP/Pea Pod) measurements used as reference labels.

## Source Papers

The local context folder `../past-papers/` contains the papers used to frame this project:

- `UMB2025.pdf`: "Enhancing Newborn Health Assessment: Ultrasound-based Body Composition Prediction Using Deep Learning Techniques"
- `EMBC2026_Attention_Mechanisms_for_Body_Composition_Prediction.pdf`: "Spatial and Global Attention Mechanisms for Ultrasound-Based Estimation of Newborn Body Composition"
- `RangerLab-IEEEAccess-2025.jpg`: first-page image for the IEEE Access article "Developing a Deep Learning Approach for Automated Body Composition Prediction in Newborns Using Ultrasound Images"

The papers should be treated as local research references. Do not commit copied papers into this repo unless redistribution is explicitly allowed.

## Problem Framing

Newborn and preterm infant growth assessment requires more than weight, length, or BMI because those measurements do not distinguish FM from FFM. The papers position ultrasound as a portable, affordable, non-ionizing alternative to less accessible reference methods such as DXA, CT, or ADP.

The research objective for this repo is to make the machine learning workflow reproducible and extensible:

- load patient-level ultrasound image sets from abdomen, biceps, and quadriceps;
- predict infant-level FM or FFM;
- compare anatomical region combinations;
- compare architecture families, especially U-Net and attention-based variants;
- report clinical/regression agreement metrics;
- produce interpretable visualizations with Grad-CAM.

## Dataset Assumptions

The papers describe a cohort with ultrasound images from clinically stable preterm infants. The repo expects the working data export to preserve the same structure:

- one folder per `Study_ID`;
- multiple images per anatomical site;
- abdomen images identified by `A`/`AB` naming patterns;
- biceps images identified by `B`/`BICEP` naming patterns;
- quadriceps images identified by `Q`/`QUAD` naming patterns;
- CSV labels containing `FM`, `FFM`, `Weight_visit`, and `Length_visit`.

The checked-in tests use synthetic images only. Real clinical data remains local and gitignored.

## Model Direction

The UMB paper motivates the modified U-Net baseline and anatomical-region experiments. It reports strong FM performance when biceps, abdomen, and quadriceps are combined (`BAQ`), and strong FFM performance from abdomen-only images in that analysis.

The EMBC attention paper motivates:

- `attention_unet` as a first-class model option;
- comparison against non-attention CNN baselines;
- future ViT/MedViT-style global attention support;
- Grad-CAM as a required interpretability workflow.

The current code includes U-Net-compatible Attention U-Net and UNet++ model slots, ResNet/EfficientNet image-set regressors, RNN/UNeXt-style alternatives, and a multi-UNet ensemble for separate region models.

## Evaluation Targets

Experiments should report:

- MAE, MSE, RMSE, and MAPE for FM and FFM;
- Bland-Altman bias and limits of agreement when comparing predictions to reference labels;
- region-combination comparisons for `A`, `B`, `Q`, `BA`, `BQ`, `AQ`, and `BAQ`;
- Grad-CAM visual checks for anatomically plausible attention.

Metrics in the papers are useful benchmarks, but exact reproduction depends on matching the same patient split, image selection, preprocessing, checkpoint, and model implementation.
