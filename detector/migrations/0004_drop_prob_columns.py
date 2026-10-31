from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("detector", "0003_scores_json"),
    ]

    operations = [
        migrations.RemoveField(model_name="prediction", name="prob_covid19"),
        migrations.RemoveField(model_name="prediction", name="prob_normal"),
        migrations.RemoveField(model_name="prediction", name="prob_pneumonia"),
        migrations.AlterField(
            model_name="prediction",
            name="label",
            field=models.CharField(max_length=20),
        ),
    ]
