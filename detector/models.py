import uuid

from django.db import models

SAFE_EXTENSIONS = {"jpg", "jpeg", "png"}
SEX_CHOICES = [("F", "Female"), ("M", "Male"), ("O", "Other")]
BLOOD_TYPES = [(b, b) for b in ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]]


def xray_upload_path(instance, filename):
    """Store uploads under a random name so user-supplied filenames never touch the disk."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
    if ext not in SAFE_EXTENSIONS:
        ext = "png"
    return f"xrays/{uuid.uuid4().hex}.{ext}"


class Prediction(models.Model):
    # Ties a record to one browser session so visitors only ever see their own history.
    session_key = models.CharField(max_length=40, db_index=True)
    image = models.ImageField(upload_to=xray_upload_path)
    original_name = models.CharField(max_length=255, blank=True)
    label = models.CharField(max_length=20)   # one of detector.ml.CLASS_NAMES
    # Softmax probability of the *predicted* class (model output score, not a clinical probability)
    confidence = models.FloatField()
    # Softmax score of every class, e.g. {"COVID19": 0.03, "NORMAL": 0.05, ...} (they sum to 1).
    # One JSON field means adding a class later needs no new column or migration.
    scores = models.JSONField(default=dict)
    status = models.CharField(max_length=20, default="Completed")
    created_at = models.DateTimeField(auto_now_add=True)

    # Optional patient details entered with the upload (demo data; stays scoped to the session).
    patient_name = models.CharField(max_length=100, blank=True)
    patient_id = models.CharField("Patient ID", max_length=40, blank=True)
    patient_age = models.PositiveSmallIntegerField("Age", null=True, blank=True)
    patient_sex = models.CharField("Sex", max_length=1, blank=True, choices=SEX_CHOICES)
    patient_blood_type = models.CharField("Blood type", max_length=3, blank=True, choices=BLOOD_TYPES)
    patient_notes = models.CharField("Symptoms / notes", max_length=500, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.label} ({self.confidence:.2f}) - {self.created_at:%Y-%m-%d %H:%M}"

    @property
    def confidence_pct(self):
        return round(self.confidence * 100, 1)

    @property
    def score_rows(self):
        """[{name, pct, css}] for every class in CLASS_NAMES order, for the result page bars."""
        from . import ml
        return [{"name": n, "pct": round(float(self.scores.get(n, 0.0)) * 100, 1), "css": n.lower()}
                for n in ml.CLASS_NAMES]

    @property
    def runner_up(self):
        """Second-highest class score row, or None."""
        others = [r for r in self.score_rows if r["name"] != self.label]
        return max(others, key=lambda r: r["pct"]) if others else None

    @property
    def patient_display(self):
        return self.patient_name or "Unnamed patient"

    @property
    def patient_ref(self):
        """User-entered patient ID, else a stable record-based reference."""
        return self.patient_id or f"PD-{self.pk:05d}"

    def delete(self, *args, **kwargs):
        """Delete the record AND the uploaded image file."""
        storage, name = self.image.storage, self.image.name
        super().delete(*args, **kwargs)
        if name:
            storage.delete(name)
