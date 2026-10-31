from django.db import migrations, models


def fill_old_rows(apps, schema_editor):
    """Rows saved by the old 2-class model only had prob_pneumonia; derive prob_normal from it."""
    Prediction = apps.get_model("detector", "Prediction")
    for p in Prediction.objects.all():
        p.prob_normal = 1.0 - p.prob_pneumonia
        p.prob_covid19 = 0.0
        p.save(update_fields=["prob_normal", "prob_covid19"])


class Migration(migrations.Migration):

    dependencies = [
        ("detector", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="prediction",
            name="prob_covid19",
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name="prediction",
            name="prob_normal",
            field=models.FloatField(default=0.0),
        ),
        migrations.AlterField(
            model_name="prediction",
            name="prob_pneumonia",
            field=models.FloatField(default=0.0),
        ),
        migrations.AlterField(
            model_name="prediction",
            name="label",
            field=models.CharField(
                choices=[("COVID19", "COVID19"), ("NORMAL", "NORMAL"), ("PNEUMONIA", "PNEUMONIA")],
                max_length=10,
            ),
        ),
        migrations.RunPython(fill_old_rows, migrations.RunPython.noop),
    ]
