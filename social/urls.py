from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.community_chat, name='community_chat'),
    path('battles/', views.battles_page, name='battles'),
    path('battle/create/<int:opponent_id>/', views.create_battle, name='create_battle'),
    path('battle/<int:battle_id>/vote/<int:voted_for_id>/', views.vote_battle, name='vote_battle'),
    path('battle/accept/<int:battle_id>/', views.accept_battle, name='accept_battle'),
    path('battle/detail/<int:battle_id>/', views.battle_detail, name='battle_detail'),
    path('battle/vote/<int:battle_id>/<int:voted_for_id>/', views.vote_battle, name='vote_battle'),
]
