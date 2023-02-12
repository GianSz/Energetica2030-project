from django.shortcuts import render
from django.contrib.auth.decorators import login_required

@login_required(login_url='/logIn/')
#This function renders the operation page
def operationPage(request):
    return render(request, 'operation/operationPage.html', context={})