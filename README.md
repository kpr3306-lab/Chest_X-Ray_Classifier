# ThoraxIQ — Chest X-ray Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-4.2%2B-092E20?logo=django&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--18-EE4C2C?logo=pytorch&logoColor=white)

A Django web app that classifies chest X-ray images into **COVID19**, **NORMAL**, **PNEUMONIA** or **TUBERCULOSIS** using a fine-tuned ResNet-18.

> [!WARNING]
> This is an educational project. It is **not a medical device** and has **not been clinically validated**. Do not use it for diagnosis.

---

## Contents
- [Features](#features)
- [Results](#results)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Train your own model](#train-your-own-model)
- [Tests](#tests)
- [Configuration](#configuration)
- [Deployment](#deployment)

## Features
- **Upload** — drag & drop or browse, image preview, file validation, optional patient details (name, ID, age, sex, blood type, notes).
- **Result** — predicted class, confidence, and a score bar for every class (plus the runner-up).
- **History** — all past predictions with a calendar view; delete one or all. History is private to each browser session.
- **Model performance dashboard** — confusion matrix, accuracy / precision / recall / F1, ROC-AUC and training curves, all read from real test-set metrics.
- **About** — how the model was built and its limitations.

## Results
Evaluated on the held-out test set (**771 images**) with the shipped model `detector/ml_models/resnet18_chestxray_4class.pth`.

| Metric | Value |
|---|---|
| Accuracy | **89.2%** |
| Macro ROC-AUC (one-vs-rest) | **0.985** |

| Class | Precision | Recall | ROC-AUC | Test images |
|---|---|---|---|---|
| COVID19 | 0.981 | 0.962 | 1.000 | 106 |
| NORMAL | 0.959 | 0.692 | 0.966 | 234 |
| PNEUMONIA | 0.845 | 0.982 | 0.975 | 390 |
| TUBERCULOSIS | 0.911 | 1.000 | 1.000 | 41 |

**Confusion matrix** (rows = actual, columns = predicted):

| | COVID19 | NORMAL | PNEUMONIA | TUBERCULOSIS |
|---|---|---|---|---|
| **COVID19** | 102 | 0 | 2 | 2 |
| **NORMAL** | 2 | 162 | 68 | 2 |
| **PNEUMONIA** | 0 | 7 | 383 | 0 |
| **TUBERCULOSIS** | 0 | 0 | 0 | 41 |

The main weakness: 68 NORMAL X-rays are predicted as PNEUMONIA, so NORMAL recall is only 69%. The model leans towards flagging disease rather than missing it.

Training ran for 10 epochs (5 head-only + 5 fine-tuning); final validation accuracy was 97.1%. The full log is in [training_log.txt](training_log.txt).

## How it works
**Model** — ImageNet-pretrained ResNet-18 with its final layer replaced by a 4-way classifier.

**Preprocessing** (same for training evaluation and the web app, see `eval_transform()` in [detector/ml.py](detector/ml.py)):
grayscale → 3 channels → resize to 224×224 → ImageNet normalisation.
Training images also get random horizontal flip, ±10° rotation and brightness jitter.

**Training recipe** ([train.py](train.py)):
1. Stratified 80/20 split of `train/` for validation (fixed seed 42).
2. Cross-entropy loss with class weights, so rarer classes count more.
3. Phase 1 — backbone frozen, train only the final layer (Adam, lr 1e-3, 5 epochs).
4. Phase 2 — also unfreeze `layer4` (Adam, lr 1e-4, 5 epochs).
5. Keep the checkpoint with the lowest validation loss.

**Web app** — PyTorch is imported lazily, so pages that don't need the model (home, about, history) load fast. Uploads are saved under random file names and only served to the session that uploaded them.

## Project structure
```
.
├── pneumodetect/          Django project (settings, urls, wsgi)
├── detector/              Django app
│   ├── ml.py              model loading, preprocessing, inference
│   ├── views.py           pages: home, upload, result, history, dashboard, about
│   ├── models.py          Prediction model (label, scores, patient details)
│   ├── forms.py           upload form and validation
│   ├── templates/         HTML templates
│   ├── static/            CSS, JS (charts, calendar, upload), images
│   ├── ml_models/         trained weights (.pth) and metrics.json
│   └── tests.py           test suite (model stubbed)
├── train.py               trains the 4-class model
├── export_metrics.py      evaluates the model and writes metrics.json
├── training_log.txt       log of the training run behind the shipped model
├── requirements.txt
└── Procfile               gunicorn start command for deployment
```

## Getting started
```bash
git clone https://github.com/kpr3306-lab/Chest_X-Ray_Classifier.git
cd Chest_X-Ray_Classifier

python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

python manage.py migrate
python manage.py runserver
```
Open http://127.0.0.1:8000.

The trained weights are already included at `detector/ml_models/resnet18_chestxray_4class.pth` (~45 MB). If you replace them, keep that exact file name. Older 2- or 3-class weights will not load, because their final layer has the wrong number of outputs.

## Train your own model
**Dataset:** Kaggle [Chest X-Ray (Pneumonia, Covid-19, Tuberculosis)](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis). Expected layout:
```
Data/
  Train/  COVID19/  NORMAL/  PNEUMONIA/  TUBERCULOSIS/
  Test/   COVID19/  NORMAL/  PNEUMONIA/  TUBERCULOSIS/
```
Folder names are matched case-insensitively, and `TURBERCULOSIS` / `TB` / `COVID` are accepted too (see `FOLDER_ALIASES` in [detector/ml.py](detector/ml.py)). An unknown or missing class folder stops the script with a clear message. The dataset's own `Val` folder is not used.

```bash
pip install scikit-learn
python train.py --data "C:\path\to\Data"
python export_metrics.py --data "C:\path\to\Data" --log training_log.txt
```

`train.py` writes the weights to `detector/ml_models/resnet18_chestxray_4class.pth` and the epoch log to `training_log.txt`. A GPU is used automatically if available.

| `train.py` option | Default | Meaning |
|---|---|---|
| `--data` | required | folder containing `train/` and `test/` |
| `--epochs-head` | 5 | epochs with the backbone frozen |
| `--epochs-finetune` | 5 | epochs with `layer4` unfrozen |
| `--batch-size` | 32 | batch size |
| `--workers` | 0 | DataLoader workers (0 is safest on Windows) |
| `--seed` | 42 | random seed |
| `--no-pretrained` | off | random weights instead of ImageNet (offline smoke tests only) |

`export_metrics.py` evaluates the model on `test/` and writes the confusion matrix, macro and per-class ROC-AUC, and (with `--log`) the training curves to `detector/ml_models/metrics.json`, which the dashboard reads. The `--log` file must come from the same run that produced the `.pth`. Without it, the dashboard shows that the curves are missing instead of inventing them.

## Tests
```bash
python manage.py test
```
The model is stubbed in the tests, so they run without PyTorch.

## Configuration
Settings are read from environment variables (see [pneumodetect/settings.py](pneumodetect/settings.py)):

| Variable | Default | Purpose |
|---|---|---|
| `DEBUG` | `1` | set to `0` in production |
| `SECRET_KEY` | dev-only key | **required** when `DEBUG=0` |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | comma-separated host names |
| `CSRF_TRUSTED_ORIGINS` | empty | comma-separated origins, e.g. `https://example.com` |
| `DB_PATH` | `db.sqlite3` | SQLite database file |
| `MEDIA_ROOT` | `media/` | where uploaded X-rays are stored |
| `RENDER_EXTERNAL_HOSTNAME` | — | set by Render; added to allowed hosts and CSRF origins automatically |

## Deployment
Example using [Render](https://render.com):

1. Create a new **Web Service** from this repository.
2. Build command:
   ```
   pip install -r requirements.txt && python manage.py collectstatic --noinput && python manage.py migrate
   ```
3. Start command:
   ```
   gunicorn pneumodetect.wsgi --workers 1 --threads 2 --timeout 120
   ```
4. Environment variables: `DEBUG=0`, `SECRET_KEY=<long random string>`.

Notes:
- Free hosting disks are ephemeral: uploads and history reset on each redeploy or restart. That is fine for a demo.
- **Predictions need more than 512 MB of RAM.** On Render's free 512 MB instance the pages load, but uploading an X-ray fails with *Internal Server Error* because loading PyTorch and the model is too heavy. Use an instance with more RAM (e.g. a paid Render plan).
- If you deploy a model trained with different transforms, update `eval_transform()` in [detector/ml.py](detector/ml.py) to match.

## Disclaimer
ThoraxIQ was built as a course project for learning purposes. Its predictions are model output scores, not clinical probabilities, and must not be used to make medical decisions.
