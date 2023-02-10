from django.urls import path
from .views import viabilityPage

app_name = 'viability'
urlpatterns = [
    path('viability/', viabilityPage, name='viabilityPage'),
]