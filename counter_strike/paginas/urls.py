from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predecir_kill/', views.predecir_kill, name='predecir_kill'),
]