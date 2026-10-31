from django.urls import path

from . import views

app_name = "detector"

urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload, name="upload"),
    path("result/<int:pk>/", views.result, name="result"),
    path("image/<int:pk>/", views.prediction_image, name="image"),
    path("history/", views.history, name="history"),
    path("calendar-data/", views.calendar_data, name="calendar_data"),
    path("delete/<int:pk>/", views.delete_prediction, name="delete"),
    path("delete-all/", views.delete_all, name="delete_all"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("about/", views.about, name="about"),
]
