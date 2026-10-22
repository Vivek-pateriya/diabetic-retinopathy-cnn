# RetinaCheck by Vivek Pateriya

A new educational retinal-image classification implementation.

Copyright (c) 2026 Vivek Pateriya for this project's newly authored code.

## Run the app (no training required)

Install 64-bit CPython and Git, then run these commands in Windows PowerShell.
Python 3.13 is recommended for stable TensorFlow; Python 3.14 uses a TensorFlow
release candidate (see compatibility below).

```powershell
git clone https://github.com/Vivek-pateriya/diabetic-retinopathy-cnn.git
cd diabetic-retinopathy-cnn
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000 and upload a retinal fundus PNG or JPEG.
The trained model is included at `artifacts/retina.keras`, so no dataset
download or retraining is required for prediction. Keep PowerShell running.
For later launches, run only `.\.venv\Scripts\python.exe app.py` from the
project directory. Dependencies require an internet connection on first setup.

## Python and TensorFlow compatibility

| Python | TensorFlow selected by requirements.txt | Status |
| --- | --- | --- |
| 3.10-3.13 | 2.21.0 | Stable release |
| 3.14 | 2.22.0rc0 | Pre-release; use for testing, not production |

For Python 3.14, replace `py -3.13 -m venv .venv` above with
`py -3.14 -m venv .venv`. The dependency markers select the matching TensorFlow
automatically; no global `--pre` flag is needed. Python 3.15+, free-threaded
Python builds, and Windows ARM64 are not supported by this setup.

When upgrading Python, create a **new** environment instead of reusing one
created by an older interpreter. To keep the old environment intact:

```powershell
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install --upgrade pip
.\.venv314\Scripts\python.exe -m pip install -r requirements.txt
.\.venv314\Scripts\python.exe -m unittest discover -s tests -v
.\.venv314\Scripts\python.exe app.py
```

These commands run from the project directory. For later commands, use
`.venv314` instead of `.venv` when that is the environment you created.
If you already cloned this repository, run `git pull --ff-only` inside it
before following the upgrade commands so you have the updated requirements.
If `.venv314` is already set up, run only
`.\.venv314\Scripts\python.exe app.py` to launch the app again.
MSYS2 Python is not a substitute for standard Windows CPython wheels.
If Windows reports a missing DLL, install the Microsoft Visual C++
Redistributable required by TensorFlow.

Sources: [TensorFlow stable wheels](https://pypi.org/project/tensorflow/2.21.0/)
and [Python 3.14 pre-release wheels](https://pypi.org/project/tensorflow/2.22.0rc0/).

## Optional: train and evaluate your own model

This section assumes you completed the setup above and are inside the project
directory. If you cloned the repository but have not set it up yet, run these
commands from the directory containing `diabetic-retinopathy-cnn`:

```powershell
cd diabetic-retinopathy-cnn
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Setup is required only once. If `.venv` and dependencies are already ready,
do not recreate the environment; continue with training below. If the web app
is running in this terminal, stop it with Ctrl+C first or use another terminal
opened in the project directory.

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

## Verification status

The bundled model was checked locally: the homepage returned HTTP 200 and
an image upload returned HTTP 200 with a grade and five probabilities.
On 22 October 2026, the compatibility suite passed on Python 3.14.6 with
TensorFlow 2.22.0rc0 and Keras nightly 3.16.0.dev2026102204, and on the existing
Python 3.11.9 / TensorFlow 2.16.1 environment. Checks cover bundled-model
inference, a two-image training update with temporary model saving, and Flask
image upload. Python 3.13 / TensorFlow 2.21.0 is supported by the dependency
configuration but was not runtime-tested on this machine.

## Web interface

The white-and-teal interface includes a dark-mode switch. The browser remembers
your theme choice. Upload previews and probability bars work in either theme.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000. Prediction uses the bundled trained model at
`artifacts/retina.keras`. Uploads are temporary and removed after prediction.
