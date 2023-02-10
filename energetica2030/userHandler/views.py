from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.contrib import messages

#This function renders the signUp page that will allow the user to create a new account.
def signUpPage(request, error=None):
    #Next conditional compares if the method received is POST, i.e, if user has sent the signUp form, and then if that's true, takes all the info given in the info in order to create a new User and store it in the database
    if(request.method=='POST'):

        name = request.POST["name"]
        surname = request.POST["surname"]
        username = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]
        confirmPassword = request.POST["confirmPassword"]

        if(User.objects.filter(username=username)):
            #Then the given username already exists, thus our user is redirected to signUp page again
            messages.error(request, '¡Este nombre de usuario ya existe!')
            return HttpResponseRedirect(reverse('userHandler:signUpPage'))
        elif(User.objects.filter(email=email)):
            #Then the given email has already been registered, thus our user is redirected to signUp page again
            messages.error(request, '¡Este correo ya ha sido registrado anteriormente!')
            return HttpResponseRedirect(reverse('userHandler:signUpPage'))
        elif(password == confirmPassword):
            #Then creates a user with the given data and then redirects the user to the logIn page
            User.objects.create_user(username=username, email=email, password=password, first_name=name, last_name=surname)
            messages.success(request, 'La cuenta se ha creado correctamente')
            return HttpResponseRedirect(reverse('userHandler:logInPage'))
        else:
            #Then the passwords do not match, thus our user is redirected to the signUp page again
            messages.error(request, '¡Las contraseñas no coinciden!')
            return HttpResponseRedirect(reverse('userHandler:signUpPage'))
        
    return render(request, template_name='signUp/signUp.html', context={'errorMessage':error})

#This function renders the logIn page that will allow the user to get in the main page.
def logInPage(request):
    #Next conditional compares if the method received is POST, i.e, if user has sent the logIn form, and then if that's true, verifies if the given username and password are correct in terms of matching
    if(request.method == 'POST'):

        user = authenticate(username=request.POST["username"], password=request.POST["password"])

        if user:
            #Then the given credentials are correct, thus our user is redirected to the main page
            return HttpResponseRedirect(reverse('main:mainPage'))
        else:
            #The given credentials weren't correct, thus our user is redirected to logIn page again.
            messages.error(request, '¡El usuario y la contraseña no coinciden!')
            return HttpResponseRedirect(reverse('userHandler:logInPage'))
        
    return render(request, 'logIn/logIn.html', context={})