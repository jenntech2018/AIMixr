from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from generator import views as gen_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("privacy/", gen_views.privacy_policy, name="privacy_policy"),
    path("terms/", gen_views.terms_of_service, name="terms_of_service"),
    path("accounts/", include("allauth.urls")),
    path("", include("generator.urls")),  # your app's URLs
    path("", include("accounts.urls")),  # accounts app URLs
    path('social/', include('social.urls')),  # social app URLs
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
