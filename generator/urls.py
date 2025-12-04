from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path("", views.dashboard_view, name="dashboard"),
    path("track/<int:track_id>/master/", views.master_track, name="master_track"),
    path("track/<int:track_id>/", views.track_detail, name="track_detail"),
    path("rankings/", views.rankings_page, name="rankings"),
    path("api/rankings/", views.rankings_data, name="rankings_data"),

    # 🔥 add these:
    path("login/", auth_views.LoginView.as_view(template_name="login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),

    path('register/', views.register_view, name='register'),
    path("dashboard/", views.dashboard_view, name="dashboard"),
]