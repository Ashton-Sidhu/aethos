from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.csrf import csrf_protect

import hyperion.settings as settings

from .forms import UploadFilesForm


def load_analysis(request):

    return render(request, 'analysis.html')

def file_upload(request):

    #FILES IS A DICT
    files = request.FILES.getlist('file_field')
    if request.method == 'POST':
        form = UploadFilesForm(request.POST, request.FILES)
        if form.is_valid():
            for f in files:
                _handle_uploaded_file(f, "test")

            return render(request, 'analysis.html')


    return render(request, 'fileupload.html')

def _handle_uploaded_file(f, name):
    with open(f'{settings.SOURCE_DATA}/{name}/name.txt', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
