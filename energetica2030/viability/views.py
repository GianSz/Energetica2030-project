from django.shortcuts import render
from django.contrib.auth.decorators import login_required

@login_required(login_url='/logIn/')
#This function renders the viability page
def viabilityPage(request):
    return render(request, 'viability/viabilityPage.html', context={})