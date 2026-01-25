from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.community_chat, name='community_chat'),
    path('battles/', views.battles_page, name='battles'),
    path('battle/create/<int:opponent_id>/', views.create_battle, name='create_battle'),
    path('battle/<int:battle_id>/vote/<int:voted_for_id>/', views.vote_battle, name='vote_battle'),
]
