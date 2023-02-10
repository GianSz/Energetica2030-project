from django.urls import path
from .views import mainPage

app_name='main'
urlpatterns = [
    path('main/', mainPage, name='mainPage')
]