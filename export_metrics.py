"""Export REAL test-set metrics for the dashboard.

Loads the deployed 4-class model (detector/ml_models/resnet18_chestxray_4class.pth), evaluates it
on the test folder (COVID19 / NORMAL / PNEUMONIA / TUBERCULOSIS sub-folders) and writes detector/ml_models/metrics.json.

Usage (run from this folder):
    python export_metrics.py --data "C:\\path\\to\\Data"
    python export_metrics.py --data "C:\\path\\to\\Data" --log training_log.txt

--data is the folder that contains test/ (any capitalisation, e.g. Test). Class folders may be spelled
COVID19, NORMAL, PNEUMONIA and TUBERCULOSIS (or TURBERCULOSIS, TB) in any letter case.

--log is optional: a text file containing the epoch lines your training cell printed, e.g.
    [head] ep 1/5 | train loss 0.231 acc 0.912 | val loss 0.180 acc 0.931 | 42s
It must come from the SAME training run that produced the saved model.
Needs scikit-learn (pip install scikit-learn) in addition to requirements.txt.
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

LINE = re.compile(
    r"\[(?P<phase>[\w-]+)\]\s*ep\s*(?P<ep>\d+)/(?P<total>\d+)\s*\|\s*"
    r"train loss\s*(?P<tl>[\d.]+)\s*acc\s*(?P<ta>[\d.]+)\s*\|\s*"
    r"val loss\s*(?P<vl>[\d.]+)\s*acc\s*(?P<va>[\d.]+)"
)


def parse_log(path):
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "phases": []}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        hist["train_loss"].append(float(m["tl"]))
        hist["train_acc"].append(float(m["ta"]))
        hist["val_loss"].append(float(m["vl"]))
        hist["val_acc"].append(float(m["va"]))
        hist["phases"].append(m["phase"])
    return hist if hist["train_loss"] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder that contains test/ (with one sub-folder per class inside)")
    ap.add_argument("--log", help="optional training log text file")
    args = ap.parse_args()

    import numpy as np
    import torch
    from sklearn.metrics import confusion_matrix, roc_auc_score
    from torch.utils.data import DataLoader

    from detector.ml import (CLASS_NAMES, MODEL_PATH, eval_transform, find_split_dir, load_model,
                             make_imagefolder)

    test_dir = find_split_dir(args.data, "test")
    if test_dir is None:
        sys.exit(f"Could not find a test/ folder inside {args.data}")
    try:
        ds = make_imagefolder(test_dir, eval_transform())
    except ValueError as e:
        sys.exit(str(e))

    k = len(CLASS_NAMES)
    model = load_model()
    labels, probs = [], []                      # probs: one row of k softmax scores per image
    with torch.inference_mode():
        for x, y in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0):
            probs += torch.softmax(model(x), dim=1).tolist()
            labels += y.tolist()
    probs = np.asarray(probs)
    preds = probs.argmax(axis=1)                # same arg-max rule as detector.ml.predict

    cm = confusion_matrix(labels, preds, labels=list(range(k)))   # rows = actual, cols = predicted
    # One-vs-rest ROC-AUC: per class, and the macro average (each class weighted equally)
    auc_macro = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro",
                                    labels=list(range(k))))
    y_arr = np.asarray(labels)
    auc_per_class = []
    for i in range(k):
        auc_per_class.append(float(roc_auc_score((y_arr == i).astype(int), probs[:, i]))
                             if 0 < (y_arr == i).sum() < len(y_arr) else None)

    out = {
        "model_file": MODEL_PATH.name,
        "exported_at": datetime.now().strftime("%d %b %Y, %H:%M"),
        "n_test": len(labels),
        "class_names": CLASS_NAMES,
        "confusion_matrix": cm.tolist(),        # rows = actual, cols = predicted
        "roc_auc": auc_macro,                   # macro one-vs-rest
        "roc_auc_per_class": auc_per_class,
        "history": None,
    }

    if args.log:
        hist = parse_log(args.log)
        if hist is None:
            print("WARNING: no epoch lines recognised in the log; training curves will be omitted.")
        else:
            out["history"] = hist
            print(f"Parsed {len(hist['train_loss'])} epochs from the log.")

    target = MODEL_PATH.parent / "metrics.json"
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")

    total = int(cm.sum())
    print(f"Test images: {total}")
    print(f"Accuracy:    {np.trace(cm) / total:.3f}")
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        rec = tp / cm[i].sum() if cm[i].sum() else 0.0
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        print(f"  {name:<10} recall {rec:.3f}  precision {prec:.3f}  ({cm[i].sum()} images)")
    print(f"ROC-AUC (macro one-vs-rest): {auc_macro:.3f}")
    print("  per class: " + ", ".join(f"{n} {a:.3f}" for n, a in zip(CLASS_NAMES, auc_per_class) if a is not None))
    print("Confusion matrix (rows = actual, cols = predicted, order " + ", ".join(CLASS_NAMES) + "):")
    print(cm)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
