from django.urls import path
from . import views
from .views import home, predict, predict_form, parse, predict_text, predict_text_form


urlpatterns = [
    path("", home, name="home"),
    path("predict", predict, name="predict"),
    path("predict_form", predict_form, name="predict_form"),
    path("parse", parse, name="parse"),
    path("predict_text", predict_text, name="predict_text"),
    path("predict_text_form", predict_text_form, name="predict_text_form"),
    path("predict_text_ui", views.predict_text_ui, name="predict_text_ui"),
    path("predict_form_ui", views.predict_form_ui, name="predict_form_ui"),
]
