from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render


def load_analysis(request):

    return render(request, 'analysis.html')
