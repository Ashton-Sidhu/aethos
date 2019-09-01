import os

from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.csrf import csrf_protect

import hyperion.settings as settings


def load_analysis(request):

    return render(request, 'analysis.html')

def file_upload(request):

    if request.method == 'POST':
        for file in request.FILES:
            _handle_uploaded_file(request.FILES[file], request.POST.get('title'))

    return render(request, 'fileupload.html')

def _handle_uploaded_file(f, name):

    if not os.path.exists(settings.SOURCE_DATA + name):
        os.makedirs(settings.SOURCE_DATA + name)

    with open(f'{settings.SOURCE_DATA}{name}/{f.name}', 'w+') as destination:
        for chunk in list(f.chunks()):
            destination.write(chunk.decode("utf-8"))
