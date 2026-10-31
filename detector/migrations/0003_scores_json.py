from django.db import migrations, models


def to_json(apps, schema_editor):
    """Fold the per-class float columns into the new `scores` JSON field."""
    Prediction = apps.get_model("detector", "Prediction")
    for p in Prediction.objects.all():
        p.scores = {"COVID19": p.prob_covid19, "NORMAL": p.prob_normal, "PNEUMONIA": p.prob_pneumonia}
        p.save(update_fields=["scores"])


class Migration(migrations.Migration):

    dependencies = [
        ("detector", "0002_three_class_scores"),
    ]

    operations = [
        migrations.AddField(
            model_name="prediction",
            name="scores",
            field=models.JSONField(default=dict),
        ),
        migrations.RunPython(to_json, migrations.RunPython.noop),
    ]
