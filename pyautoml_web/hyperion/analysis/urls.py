from django.contrib import admin
from django.urls import path, re_path

import analysis.views as views

urlpatterns = [
    re_path('', views.load_analysis)
]
