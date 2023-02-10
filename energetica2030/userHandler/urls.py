from django.urls import path
from .views import signUpPage, logInPage

app_name='userHandler'
urlpatterns = [
    path('signUp/', signUpPage, name='signUpPage'),
    path('logIn/', logInPage, name='logInPage'),
]