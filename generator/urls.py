from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from generator.views import logout_view, track_detail

urlpatterns = [
    path("", views.landing_page, name="home"),
    path("track/<int:track_id>/master/", views.master_track, name="master_track"),
    path("track/<int:track_id>/", views.track_detail, name="track_detail"),
    path("rankings/", views.rankings_page, name="rankings"),
    path("api/rankings/", views.rankings_data, name="rankings_data"),
    path("track/<int:track_id>/status/", views.track_status, name="track_status"),
    path("login/", auth_views.LoginView.as_view(template_name="login.html"), name="login"),
    path("logout/", logout_view, name="logout"),
    path('register/', views.register_view, name='register'),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path('track/<int:track_id>/download/', views.download_mastered_track, name='download_track'),
    path('track/<int:track_id>/split/', views.split_stems, name='split_stems'),
]

from django.conf import settings
from django.conf.urls.static import static


if settings.DEBUG:  # only serve media in dev
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
