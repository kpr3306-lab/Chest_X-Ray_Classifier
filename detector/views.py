import json
import logging
import mimetypes
from datetime import date

from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Count
from django.db.models.functions import TruncDate
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST
from PIL import Image

from . import ml
from .forms import MAX_UPLOAD_MB, XrayUploadForm
from .models import Prediction

logger = logging.getLogger(__name__)

MAX_RECORDS_PER_SESSION = 50
METRICS_PATH = ml.MODEL_PATH.parent / "metrics.json"


# ---------------------------------------------------------------- helpers ----
def _ensure_session(request):
    if not request.session.session_key:
        request.session.create()
    return request.session.session_key


def _own_prediction_or_404(request, pk):
    """Visitors can only access records created in their own browser session."""
    key = request.session.session_key
    if not key:
        raise Http404
    return get_object_or_404(Prediction, pk=pk, session_key=key)


def _ratio(n, d):
    return n / d if d else 0.0


def _pct(x):
    return f"{x * 100:.1f}%"


def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _session_qs(request):
    key = request.session.session_key
    return Prediction.objects.filter(session_key=key) if key else Prediction.objects.none()


def _date_filter(request, qs):
    """Apply ?date=YYYY-MM-DD (local time zone); an invalid value is ignored."""
    try:
        day = date.fromisoformat(request.GET.get("date", ""))
    except ValueError:
        return qs, None
    return qs.filter(created_at__date=day), day


# ------------------------------------------------------------------ pages ----
def home(request):
    """Overview dashboard: this session's latest prediction plus the exported test metrics."""
    qs, day = _date_filter(request, _session_qs(request))
    recent = list(qs.order_by("-created_at", "-pk")[:MAX_RECORDS_PER_SESSION])  # pk breaks same-timestamp ties
    total = qs.count()
    counts = {row["label"]: row["n"] for row in qs.values("label").annotate(n=Count("id"))}
    totals = [{"name": n, "css": n.lower(), "n": counts.get(n, 0),
               "pct": round(_ratio(counts.get(n, 0), total) * 100, 1)} for n in ml.CLASS_NAMES]
    return render(request, "detector/home.html", {
        "latest": recent[0] if recent else None, "recent": recent, "day": day,
        "total": total, "totals": totals, "m": _load_metrics(),
    })


def about(request):
    return render(request, "detector/about.html")


def upload(request):
    key = _ensure_session(request)
    if request.method == "POST":
        form = XrayUploadForm(request.POST, request.FILES)
        if Prediction.objects.filter(session_key=key).count() >= MAX_RECORDS_PER_SESSION:
            messages.error(
                request,
                f"You've reached the limit of {MAX_RECORDS_PER_SESSION} saved predictions. "
                "Delete some from the History page to continue.",
            )
        elif form.is_valid():
            f = form.cleaned_data["image"]
            try:
                result = ml.predict(Image.open(f))
            except FileNotFoundError:
                logger.exception("Model file missing")
                messages.error(
                    request,
                    "The trained model file is missing on the server, so predictions are unavailable.",
                )
            except Exception:
                logger.exception("Prediction failed")
                messages.error(
                    request,
                    "Something went wrong while analysing that image. Please try a different file.",
                )
            else:
                f.seek(0)
                pred = Prediction.objects.create(
                    session_key=key,
                    image=f,
                    original_name=f.name[:255],
                    **{k: form.cleaned_data[k] for k in XrayUploadForm.PATIENT_FIELDS},
                    **result,
                )
                return redirect("detector:result", pk=pred.pk)
    else:
        form = XrayUploadForm()
    return render(request, "detector/upload.html", {"form": form, "max_mb": MAX_UPLOAD_MB})


def result(request, pk):
    pred = _own_prediction_or_404(request, pk)
    return render(request, "detector/result.html", {"pred": pred})


def prediction_image(request, pk):
    """Serve an uploaded X-ray only to the session that uploaded it."""
    pred = _own_prediction_or_404(request, pk)
    content_type = mimetypes.guess_type(pred.image.name)[0] or "application/octet-stream"
    response = FileResponse(pred.image.open("rb"), content_type=content_type)
    response["Cache-Control"] = "private, max-age=3600"
    return response


def history(request):
    qs, day = _date_filter(request, _session_qs(request))
    page = Paginator(qs, 15).get_page(request.GET.get("page"))
    return render(request, "detector/history.html",
                  {"page": page, "total": _session_qs(request).count(), "day": day})


def calendar_data(request):
    """Per-day counts for the toolbar calendar: {"2026-10-31": {"n": 3, "bad": 1}}; bad = not NORMAL."""
    days = {}
    for row in (_session_qs(request).annotate(day=TruncDate("created_at"))
                .values("day", "label").annotate(n=Count("id"))):
        d = days.setdefault(row["day"].isoformat(), {"n": 0, "bad": 0})
        d["n"] += row["n"]
        if row["label"] != "NORMAL":
            d["bad"] += row["n"]
    return JsonResponse({"days": days})


@require_POST
def delete_prediction(request, pk):
    pred = _own_prediction_or_404(request, pk)
    pred.delete()
    messages.success(request, "Prediction and its uploaded image were deleted.")
    return redirect("detector:history")


@require_POST
def delete_all(request):
    key = request.session.session_key
    if key:
        for pred in Prediction.objects.filter(session_key=key):
            pred.delete()  # per-object delete so image files are removed too
    messages.success(request, "All your predictions and uploaded images were deleted.")
    return redirect("detector:history")


# -------------------------------------------------------------- dashboard ----
def _load_metrics():
    """Read metrics.json (written by export_metrics.py) and derive display values from
    the N x N confusion matrix (rows = actual class, columns = predicted class).
    Nothing here is hard-coded."""
    if not METRICS_PATH.exists():
        return None
    try:
        data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        cm = data["confusion_matrix"]
        names = data.get("class_names") or ml.CLASS_NAMES
        n = len(names)
        if n != ml.NUM_CLASSES or len(cm) != n or any(len(row) != n for row in cm):
            raise ValueError(f"metrics.json is not a {ml.NUM_CLASSES}-class matrix (old 2-class export?)")
        cm = [[int(v) for v in row] for row in cm]
    except (ValueError, KeyError, TypeError):
        logger.exception("metrics.json is unreadable or stale - re-run export_metrics.py")
        return None

    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(n))
    row_sums = [sum(cm[i]) for i in range(n)]                       # actual images per class
    col_sums = [sum(cm[i][j] for i in range(n)) for j in range(n)]  # predicted images per class

    per_class = []
    for i, name in enumerate(names):
        prec, rec = _ratio(cm[i][i], col_sums[i]), _ratio(cm[i][i], row_sums[i])
        per_class.append({"name": name, "prec": prec, "rec": rec, "f1": _f1(prec, rec),
                          "support": row_sums[i], "tp": cm[i][i]})

    macro_prec = sum(c["prec"] for c in per_class) / n
    macro_rec = sum(c["rec"] for c in per_class) / n
    macro_f1 = sum(c["f1"] for c in per_class) / n

    cards = [
        {"label": "Test accuracy", "value": _pct(_ratio(correct, total)), "hint": f"{correct} of {total} correct"},
        {"label": "Macro recall", "value": _pct(macro_rec), "hint": "average of the per-class recalls"},
        {"label": "Macro precision", "value": _pct(macro_prec), "hint": "average of the per-class precisions"},
        {"label": "Macro F1", "value": _pct(macro_f1), "hint": "each class counts equally, whatever its size"},
    ]
    if data.get("roc_auc") is not None:
        cards.append({"label": "ROC-AUC (macro, one-vs-rest)", "value": f"{data['roc_auc']:.3f}",
                      "hint": "threshold-independent separability"})
    for c in per_class:
        cards.append({"label": f"Recall, {c['name']}", "value": _pct(c["rec"]),
                      "hint": f"{c['tp']} of {c['support']} {c['name']} images caught"})

    def cell(value, row_total, correct_cell):
        return {"value": value, "alpha": round(0.12 + 0.78 * _ratio(value, row_total), 2), "correct": correct_cell}

    # Where each class's mistakes went, e.g. "12 COVID19 images were predicted as PNEUMONIA".
    confusions = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j]:
                confusions.append({"actual": names[i], "predicted": names[j], "count": cm[i][j]})
    confusions.sort(key=lambda c: -c["count"])

    return {
        "cards": cards,
        "class_names": names,
        "classes": [
            {"name": c["name"], "precision": _pct(c["prec"]), "recall": _pct(c["rec"]),
             "f1": _pct(c["f1"]), "support": c["support"]}
            for c in per_class
        ],
        "cm_rows": [
            {"label": names[i], "cells": [cell(cm[i][j], row_sums[i], i == j) for j in range(n)]}
            for i in range(n)
        ],
        "confusions": confusions,
        "roc_auc_per_class": [
            {"name": names[i], "auc": a} for i, a in enumerate(data.get("roc_auc_per_class") or [])
            if a is not None
        ],
        "n_test": total,
        "roc_auc": data.get("roc_auc"),
        "accuracy": round(_ratio(correct, total) * 100, 1),
        "macro_f1": round(macro_f1 * 100, 1),
        "history": data.get("history"),
        "model_file": data.get("model_file", ml.MODEL_PATH.name),
        "exported_at": data.get("exported_at"),
    }


def dashboard(request):
    return render(request, "detector/dashboard.html", {"m": _load_metrics()})
