from django.urls import path
from .views import operationPage

app_name = 'operation'
urlpatterns = [
    path('operation/', operationPage, name='operationPage'),
]