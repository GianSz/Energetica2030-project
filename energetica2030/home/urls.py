from django.urls import path
from .views import homePage

app_name='main'
urlpatterns = [
    path('', homePage, name='homePage'),
    path('home/', homePage, name='homePage'),
]