from django.urls import path
from . import views

urlpatterns = [
    path('track/<int:track_id>/split/', views.split_stems, name='split_stems'),
]
