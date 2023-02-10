from django.shortcuts import render

def viabilityPage(request):
    return render(request, 'viability/viability.html', context={})