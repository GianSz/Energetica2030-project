from django.shortcuts import render

def operationPage(request):
    return render(request, 'operation/operation.html', context={})