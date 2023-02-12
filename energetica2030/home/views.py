from django.shortcuts import render
from django.contrib.auth.decorators import login_required

@login_required(login_url='/logIn/')
#This function renders the home page that will allow the user to see the information about the project itself, also, from this page the user is able to navigate between the different tabs (viability and operation)
def homePage(request):
    return render(request, 'home/homePage.html', context={})