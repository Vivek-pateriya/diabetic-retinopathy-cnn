# RetinaCheck by Vivek Pateriya

A new educational retinal-image classification implementation.

Copyright (c) 2026 Vivek Pateriya for this project's newly authored code.

## Run the app (no training required)

Install Python 3.11 and Git, then run these commands in Windows PowerShell:

```powershell
git clone https://github.com/Vivek-pateriya/diabetic-retinopathy-cnn.git
cd diabetic-retinopathy-cnn
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000 and upload a retinal fundus PNG or JPEG.
The trained model is included at `artifacts/retina.keras`, so no dataset
download or retraining is required for prediction. Keep PowerShell running.
For later launches, run only `.\.venv\Scripts\python.exe app.py` from the
project directory. Dependencies require an internet connection on first setup.

## Optional: train and evaluate your own model

Download the dataset linked below and adjust the label/image paths as needed.
Training replaces the bundled model and metrics; back them up first.

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

The bundled model uses this architecture. Its stored validation accuracy is
67.12% on 733 images. Validation was also used for model selection; it is not
an independent clinical test. Recognition of advanced grades is limited.

## Data and dependencies

Dataset: [Kaggle Diabetic Retinopathy 224x224 2019 data](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data/data).
Dataset ownership and terms belong to its provider. TensorFlow, NumPy, and
Pillow retain their respective licenses.

## Web interface

The white-and-teal interface includes a dark-mode switch. The browser remembers
your theme choice. Upload previews and probability bars work in either theme.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000. Prediction uses the bundled trained model at
`artifacts/retina.keras`. Uploads are temporary and removed after prediction.
