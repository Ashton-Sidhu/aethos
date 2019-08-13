from django.contrib import admin
from django.urls import path, re_path

import analysis.views as views

urlpatterns = [
    re_path('analyze', views.load_analysis),
    re_path('upload', views.file_upload, name='upload')
]
