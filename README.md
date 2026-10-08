# RetinaCheck by Vivek Pateriya

A new educational retinal-image classification implementation.

Copyright (c) 2026 Vivek Pateriya for this project's newly authored code.

## Setup

Use Python 3.11 and PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train and evaluate

```powershell
.\.venv\Scripts\python.exe retina.py train --labels ..\archive\train.csv --images ..\archive\colored_images --epochs 1
```

One epoch checks the workflow. For a longer experiment, use `--epochs 20`.
The backend saves the training/validation split, best model, final model,
validation accuracy, and confusion matrix in `artifacts/`.

## Predict

```powershell
.\.venv\Scripts\python.exe retina.py predict --image ..\archive\colored_images\Moderate\000c1434d8d7.png
```

The model returns five probabilities. These scores are not calibrated medical
confidence. This educational model has no clinical validation.

## How it works

CSV labels match image IDs. Each grade contributes 20 percent of its images
to validation. Training uses class weights to address imbalance. The shared
image decoder resizes RGB images to 160 by 160 and normalizes pixels. Three
convolution layers extract features. Global average pooling and a small dense
classifier produce five softmax scores. Prediction applies the same decoder.

This version needs a newly trained model. Earlier model files use another
architecture and cannot be substituted.

## Data and dependencies

Dataset: [Kaggle Diabetic Retinopathy 224x224 2019 data](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data/data).
Dataset ownership and terms belong to its provider. TensorFlow, NumPy, and
Pillow retain their respective licenses.

## Verification status

Files were created locally, but terminal connection is unavailable.
Runtime and training verification must run in the working PowerShell window.
## Web interface

The white-and-teal interface includes a dark-mode switch. The browser remembers
your theme choice. Upload previews and probability bars work in either theme.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000. Prediction requires the new trained model at
`artifacts/retina.keras`. Uploads are temporary and removed after prediction.
