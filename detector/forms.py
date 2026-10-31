from django import forms
from django.core.exceptions import ValidationError

from .models import BLOOD_TYPES, SEX_CHOICES

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_UPLOAD_MB = 5
MAX_PIXELS = 25_000_000  # reject absurdly large images


class XrayUploadForm(forms.Form):
    image = forms.ImageField(
        label="Chest X-ray image",
        widget=forms.FileInput(
            attrs={"accept": ".jpg,.jpeg,.png,image/jpeg,image/png", "id": "xray-input"}
        ),
    )
    patient_name = forms.CharField(label="Patient name", max_length=100, required=False,
                                   widget=forms.TextInput(attrs={"placeholder": "e.g. Jane Doe", "autocomplete": "off"}))
    patient_id = forms.CharField(label="Patient ID", max_length=40, required=False,
                                 widget=forms.TextInput(attrs={"placeholder": "Auto-generated if empty", "autocomplete": "off"}))
    patient_age = forms.IntegerField(label="Age", min_value=0, max_value=120, required=False,
                                     widget=forms.NumberInput(attrs={"placeholder": "Years"}))
    patient_sex = forms.ChoiceField(label="Sex", choices=[("", "Not specified")] + SEX_CHOICES, required=False)
    patient_blood_type = forms.ChoiceField(label="Blood type", choices=[("", "Unknown")] + BLOOD_TYPES, required=False)
    patient_notes = forms.CharField(label="Symptoms / notes", max_length=500, required=False,
                                    widget=forms.Textarea(attrs={"rows": 3, "placeholder": "Optional, e.g. cough, fever for 5 days"}))

    PATIENT_FIELDS = ["patient_name", "patient_id", "patient_age", "patient_sex", "patient_blood_type", "patient_notes"]

    def clean_image(self):
        f = self.cleaned_data["image"]

        ext = f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise ValidationError("Only JPG, JPEG and PNG files are supported.")

        if f.size > MAX_UPLOAD_MB * 1024 * 1024:
            raise ValidationError(f"File is too large. The maximum size is {MAX_UPLOAD_MB} MB.")

        # forms.ImageField has already verified this is a real image with Pillow;
        # also check the *actual* format, not just the file extension.
        pil = getattr(f, "image", None)
        if pil is None or pil.format not in ALLOWED_FORMATS:
            raise ValidationError("The file content is not a valid JPEG or PNG image.")
        if pil.width * pil.height > MAX_PIXELS:
            raise ValidationError("The image dimensions are too large.")
        return f
