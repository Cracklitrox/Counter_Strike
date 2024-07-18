from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predecir_kill/', views.predecir_kill, name='predecir_kill'),
    path('predecir_rondas_sobrevividas/', views.predecir_rondas_sobrevividas, name='predecir_rondas_sobrevividas'),
]