import io
import json
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from PIL import Image

from . import views
from .models import Prediction

TMP_MEDIA = tempfile.mkdtemp()
FAKE = {"label": "PNEUMONIA", "confidence": 0.92,
        "scores": {"COVID19": 0.02, "NORMAL": 0.04, "PNEUMONIA": 0.92, "TUBERCULOSIS": 0.02}}
FAKE_COVID = {"label": "COVID19", "confidence": 0.81,
              "scores": {"COVID19": 0.81, "NORMAL": 0.04, "PNEUMONIA": 0.10, "TUBERCULOSIS": 0.05}}
FAKE_TB = {"label": "TUBERCULOSIS", "confidence": 0.7,
           "scores": {"COVID19": 0.05, "NORMAL": 0.05, "PNEUMONIA": 0.2, "TUBERCULOSIS": 0.7}}


def make_image(fmt="PNG", name="xray.png", size=(64, 64)):
    buf = io.BytesIO()
    Image.new("L", size, 128).save(buf, format=fmt)
    return SimpleUploadedFile(name, buf.getvalue(), content_type=f"image/{fmt.lower()}")


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class FlowTests(TestCase):
    def test_static_pages_render(self):
        for name in ["home", "upload", "history", "dashboard", "about"]:
            self.assertEqual(self.client.get(reverse(f"detector:{name}")).status_code, 200, name)

    def test_home_contains_required_content(self):
        html = self.client.get(reverse("detector:home")).content.decode()
        self.assertIn("ThoraxIQ", html)
        self.assertIn("Upload X-ray", html)
        self.assertIn("How the AI model works", html)

    def test_overview_shows_latest_prediction_and_totals(self):
        for fake in (FAKE, FAKE_COVID):
            with mock.patch("detector.ml.predict", return_value=fake):
                self.client.post(reverse("detector:upload"), {"image": make_image()})
        r = self.client.get(reverse("detector:home"))
        self.assertEqual(r.context["latest"].label, "COVID19")
        self.assertEqual(r.context["latest"].runner_up["name"], "PNEUMONIA")   # 10% beats 5% and 4%
        self.assertEqual({t["name"]: t["n"] for t in r.context["totals"]},
                         {"COVID19": 1, "NORMAL": 0, "PNEUMONIA": 1, "TUBERCULOSIS": 0})
        self.assertContains(r, "Attention Required")
        self.assertNotContains(Client().get(reverse("detector:home")), "Attention Required")

    @mock.patch("detector.ml.predict", return_value=FAKE)
    def test_patient_details_saved_and_shown(self, _):
        self.client.post(reverse("detector:upload"), {
            "image": make_image(), "patient_name": "Jane Test", "patient_id": "P-77", "patient_age": "41",
            "patient_sex": "F", "patient_blood_type": "AB-", "patient_notes": "cough"})
        pred = Prediction.objects.get()
        self.assertEqual((pred.patient_name, pred.patient_ref, pred.patient_age, pred.patient_blood_type),
                         ("Jane Test", "P-77", 41, "AB-"))
        for url in (reverse("detector:result", args=[pred.pk]), reverse("detector:history"), reverse("detector:home")):
            html = self.client.get(url).content.decode()
            self.assertIn("Jane Test", html, url)
            self.assertIn("P-77", html, url)
        # blank details fall back to a generated reference
        self.client.post(reverse("detector:upload"), {"image": make_image()})
        anon = Prediction.objects.exclude(pk=pred.pk).get()
        self.assertEqual((anon.patient_display, anon.patient_ref), ("Unnamed patient", f"PD-{anon.pk:05d}"))

    @mock.patch("detector.ml.predict", return_value=FAKE)
    def test_invalid_patient_age_rejected(self, predict):
        r = self.client.post(reverse("detector:upload"), {"image": make_image(), "patient_age": "400"})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(Prediction.objects.exists())
        predict.assert_not_called()

    def test_date_filter_and_calendar_data(self):
        for fake in (FAKE, FAKE_COVID):
            with mock.patch("detector.ml.predict", return_value=fake):
                self.client.post(reverse("detector:upload"), {"image": make_image()})
        old = Prediction.objects.get(label="COVID19")
        Prediction.objects.filter(pk=old.pk).update(created_at=old.created_at - timedelta(days=3))
        old_day = timezone.localdate(old.created_at - timedelta(days=3)).isoformat()
        today = timezone.localdate().isoformat()

        r = self.client.get(reverse("detector:home"), {"date": old_day})
        self.assertEqual([p.label for p in r.context["recent"]], ["COVID19"])
        self.assertContains(r, "Clear filter")
        r = self.client.get(reverse("detector:history"), {"date": today})
        self.assertEqual([p.label for p in r.context["page"].object_list], ["PNEUMONIA"])
        r = self.client.get(reverse("detector:history"), {"date": "not-a-date"})   # ignored, not a crash
        self.assertEqual(len(r.context["page"].object_list), 2)

        days = self.client.get(reverse("detector:calendar_data")).json()["days"]
        self.assertEqual(days, {old_day: {"n": 1, "bad": 1}, today: {"n": 1, "bad": 1}})
        self.assertEqual(Client().get(reverse("detector:calendar_data")).json(), {"days": {}})

    @mock.patch("detector.ml.predict", return_value=FAKE)
    def test_upload_predict_view_delete(self, _):
        r = self.client.post(reverse("detector:upload"), {"image": make_image()})
        pred = Prediction.objects.get()
        self.assertRedirects(r, reverse("detector:result", args=[pred.pk]))
        self.assertEqual((pred.label, pred.confidence_pct), ("PNEUMONIA", 92.0))

        page = self.client.get(reverse("detector:result", args=[pred.pk])).content.decode()
        self.assertIn("PNEUMONIA", page)
        self.assertIn("92.0%", page)
        self.assertIn("not a medical diagnosis", page)
        img = self.client.get(reverse("detector:image", args=[pred.pk]))
        self.assertEqual(img.status_code, 200)
        img.close()  # a real server closes the FileResponse; the test client doesn't, and Windows locks open files
        self.assertIn("PNEUMONIA", self.client.get(reverse("detector:history")).content.decode())

        path = Path(pred.image.path)
        self.assertTrue(path.exists())
        self.client.post(reverse("detector:delete", args=[pred.pk]))
        self.assertFalse(Prediction.objects.exists())
        self.assertFalse(path.exists(), "uploaded file should be deleted with the record")

    @mock.patch("detector.ml.predict", return_value=FAKE)
    def test_other_sessions_cannot_see_records(self, _):
        self.client.post(reverse("detector:upload"), {"image": make_image()})
        pred = Prediction.objects.get()
        other = Client()
        self.assertEqual(other.get(reverse("detector:result", args=[pred.pk])).status_code, 404)
        self.assertEqual(other.get(reverse("detector:image", args=[pred.pk])).status_code, 404)
        self.assertEqual(other.post(reverse("detector:delete", args=[pred.pk])).status_code, 404)
        self.assertNotIn("PNEUMONIA", other.get(reverse("detector:history")).content.decode())

    @mock.patch("detector.ml.predict", return_value=FAKE)
    def test_delete_all(self, _):
        for _i in range(2):
            self.client.post(reverse("detector:upload"), {"image": make_image()})
        self.assertEqual(Prediction.objects.count(), 2)
        self.client.post(reverse("detector:delete_all"))
        self.assertEqual(Prediction.objects.count(), 0)

    @mock.patch("detector.ml.predict", return_value=FAKE)
    def test_validation_rejects_bad_uploads(self, predict):
        bad_ext = SimpleUploadedFile("scan.gif", make_image("PNG").read(), content_type="image/gif")
        not_image = SimpleUploadedFile("fake.png", b"this is not an image", content_type="image/png")
        wrong_fmt = make_image("GIF", "scan.png")                       # GIF bytes with .png name
        huge = SimpleUploadedFile("big.png", make_image().read() + b"0" * (5 * 1024 * 1024), content_type="image/png")
        for f in (bad_ext, not_image, wrong_fmt, huge):
            r = self.client.post(reverse("detector:upload"), {"image": f})
            self.assertEqual(r.status_code, 200)
            self.assertFalse(Prediction.objects.exists())
        predict.assert_not_called()

    @mock.patch("detector.ml.predict", side_effect=FileNotFoundError("nope"))
    def test_missing_model_is_handled(self, _):
        r = self.client.post(reverse("detector:upload"), {"image": make_image()}, follow=True)
        self.assertContains(r, "model file is missing")
        self.assertFalse(Prediction.objects.exists())

    def test_jpeg_accepted(self):
        with mock.patch("detector.ml.predict", return_value=FAKE):
            self.client.post(reverse("detector:upload"), {"image": make_image("JPEG", "a.jpg")})
        self.assertEqual(Prediction.objects.count(), 1)

    def test_covid_prediction_is_stored_and_shown(self):
        with mock.patch("detector.ml.predict", return_value=FAKE_COVID):
            self.client.post(reverse("detector:upload"), {"image": make_image()})
        pred = Prediction.objects.get()
        self.assertEqual(pred.label, "COVID19")
        self.assertEqual([r["pct"] for r in pred.score_rows], [81.0, 4.0, 10.0, 5.0])
        page = self.client.get(reverse("detector:result", args=[pred.pk])).content.decode()
        self.assertIn("badge-covid19", page)
        self.assertIn("81.0%", page)


    def test_tuberculosis_prediction_is_stored_and_shown(self):
        with mock.patch("detector.ml.predict", return_value=FAKE_TB):
            self.client.post(reverse("detector:upload"), {"image": make_image()})
        pred = Prediction.objects.get()
        page = self.client.get(reverse("detector:result", args=[pred.pk])).content.decode()
        self.assertIn("badge-tuberculosis", page)
        self.assertIn("bar-tuberculosis", page)
        self.assertIn("70.0%", page)
        self.assertIn("badge-tuberculosis", self.client.get(reverse("detector:history")).content.decode())


class DashboardTests(TestCase):
    def test_empty_state_without_metrics(self):
        with mock.patch.object(views, "METRICS_PATH", Path("/nonexistent/metrics.json")):
            self.assertContains(self.client.get(reverse("detector:dashboard")), "haven")

    def test_metrics_derived_from_confusion_matrix(self):
        # rows = actual, cols = predicted; order COVID19, NORMAL, PNEUMONIA, TUBERCULOSIS
        cm = [[80, 5, 15, 0], [4, 200, 16, 0], [10, 20, 360, 10], [2, 3, 5, 30]]   # 100 / 220 / 400 / 40
        data = {"model_file": "m.pth", "class_names": ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"],
                "confusion_matrix": cm, "roc_auc": 0.97, "roc_auc_per_class": [0.95, 0.98, 0.97, 0.99],
                "history": {"train_loss": [.5, .3], "val_loss": [.4, .3], "train_acc": [.8, .9],
                            "val_acc": [.85, .9], "phases": ["head", "finetune"]}}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "metrics.json"
            p.write_text(json.dumps(data))
            with mock.patch.object(views, "METRICS_PATH", p):
                html = self.client.get(reverse("detector:dashboard")).content.decode()
                m = views._load_metrics()
        # accuracy 670/760 = 88.2%; COVID19 recall 80/100 = 80.0%; COVID19 precision 80/96 = 83.3%;
        # TUBERCULOSIS recall 30/40 = 75.0%
        for expected in ["88.2%", "80.0%", "83.3%", "75.0%", "0.970", "history-data", "TUBERCULOSIS", "predicted as"]:
            self.assertIn(expected, html)
        self.assertEqual(m["n_test"], 760)
        self.assertEqual([c["support"] for c in m["classes"]], [100, 220, 400, 40])
        self.assertEqual(m["confusions"][0], {"actual": "PNEUMONIA", "predicted": "NORMAL", "count": 20})

    def test_stale_three_class_metrics_are_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "metrics.json"
            p.write_text(json.dumps({"confusion_matrix": [[80, 5, 15], [4, 200, 16], [10, 20, 370]], "roc_auc": 0.96}))
            with mock.patch.object(views, "METRICS_PATH", p):
                self.assertIsNone(views._load_metrics())
                self.assertContains(self.client.get(reverse("detector:dashboard")), "haven")


class ModelFileTests(TestCase):
    def test_class_order_matches_imagefolder(self):
        from . import ml
        self.assertEqual(ml.CLASS_NAMES, sorted(ml.CLASS_NAMES))   # ImageFolder sorts alphabetically
        self.assertEqual(ml.NUM_CLASSES, 4)
        self.assertEqual(ml.CLASS_NAMES[-1], "TUBERCULOSIS")

    def test_folder_aliases(self):
        from . import ml
        for name in ["Turberculosis", "TURBERCULOSIS", "tb", "Tuberculosis"]:
            self.assertEqual(ml._canon(name), "TUBERCULOSIS")
        self.assertEqual(ml._canon("covid-19"), "COVID19")
        self.assertIsNone(ml._canon("viral_pneumonia"))


class ExportScriptTests(TestCase):
    def test_parse_log(self):
        import export_metrics
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("noise\n[head] ep 1/5 | train loss 0.231 acc 0.912 | val loss 0.180 acc 0.931 | 42s\n"
                    "[finetune] ep 1/5 | train loss 0.100 acc 0.960 | val loss 0.120 acc 0.955 | 50s\n")
        h = export_metrics.parse_log(f.name)
        self.assertEqual(h["phases"], ["head", "finetune"])
        self.assertEqual(h["val_acc"], [0.931, 0.955])
