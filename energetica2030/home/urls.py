from django.urls import path
from .views import homePage

app_name='main'
urlpatterns = [
    path('home/', homePage, name='homePage')
]