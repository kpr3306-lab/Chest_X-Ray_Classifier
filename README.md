# ThoraxIQ — Chest X-ray Classification

Educational Django app that classifies chest X-rays into COVID19 / NORMAL / PNEUMONIA / TUBERCULOSIS with a fine-tuned ResNet-18.
**Not a medical device. Not clinically validated.**

Pages: Home, Upload (drag & drop + preview + validation), Result, History (with delete), Model performance dashboard, About.

## 1. Run locally

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
pip install scikit-learn         # only needed for export_metrics.py

# train the 4-class model first (section 2), or copy trained weights here (keep this exact name):
#   detector/ml_models/resnet18_chestxray_4class.pth
# NOTE: older 2- or 3-class weights cannot be loaded - their final layer has the wrong number of outputs.

python manage.py migrate
python manage.py runserver
```
Open http://127.0.0.1:8000

## 2. Train the 4-class model and fill the dashboard with REAL numbers

Dataset: Kaggle "Chest X-Ray (Pneumonia, Covid-19, Tuberculosis)" (jtiptj). It has Train / Test / Val folders, each with one sub-folder per class:
```
Data/
  Train/  COVID19/  NORMAL/  PNEUMONIA/  TUBERCULOSIS/
  Test/   COVID19/  NORMAL/  PNEUMONIA/  TUBERCULOSIS/
```
Capitalisation of `Train`/`Test` and the class-folder spelling do not matter (`TURBERCULOSIS` and `TB` are accepted too, see `FOLDER_ALIASES`
in `detector/ml.py`). Any unknown or missing class folder stops the script with a clear message. The dataset's own `Val` folder is not used.
```bash
python train.py --data "C:\path\to\Data"            # writes detector/ml_models/resnet18_chestxray_4class.pth and training_log.txt
python export_metrics.py --data "C:\path\to\Data" --log training_log.txt
```
`train.py` follows the same recipe as the original notebook (80/20 validation split, class weights, 5 epochs head + 5 epochs fine-tuning
of `layer4`, best checkpoint by validation loss). `export_metrics.py` evaluates the model on `test/` and writes the 4x4 confusion matrix,
macro one-vs-rest ROC-AUC and per-class AUCs to `metrics.json`; accuracy, precision, recall and F1 are calculated from the confusion matrix.
`--log` must come from the same run that produced the `.pth`. Without it the dashboard says the curves are missing instead of inventing them.
After changing the model code, run `python manage.py migrate` once (stores the per-class scores in one JSON field; adding a class later needs no new migration).

## 3. Tests
```bash
python manage.py test
```
(The model is stubbed in the tests, so they run without PyTorch.)

## 4. Deploy (Render example)
1. Push this folder to GitHub (include the `.pth` and `metrics.json`; the `.pth` is ~45 MB).
2. New **Web Service** on Render. Build command:
   `pip install -r requirements.txt && python manage.py collectstatic --noinput && python manage.py migrate`
   Start command: `gunicorn pneumodetect.wsgi --workers 1 --threads 2 --timeout 120`
3. Environment variables: `DEBUG=0`, `SECRET_KEY=<long random string>`.
   (`RENDER_EXTERNAL_HOSTNAME` is picked up automatically for ALLOWED_HOSTS and CSRF.)

Notes
- Free hosting disks are ephemeral: uploads and history reset on each redeploy/restart. That is fine for a demo.
- PyTorch is memory-hungry. If a 512 MB instance runs out of memory, use a host with more RAM
  (for example Hugging Face Spaces with Docker).
- History is private per browser session; uploaded images are only served to the session that uploaded them.

## Preprocessing note
`detector/ml.py` uses the training evaluation transform: grayscale (3 ch) -> resize 224x224 -> normalise (ImageNet).
If you ever deploy a model trained with different transforms, change `eval_transform()` to match.
