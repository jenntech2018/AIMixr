from django.urls import path
from . import views

urlpatterns = [
    path("upgrade/", views.upgrade, name="upgrade"),
    path("upgrade/checkout/", views.checkout, name="checkout"),
    path("upgrade/create-checkout-session/", views.create_checkout_session, name="create_checkout"),
    path("upgrade/success/", views.success, name="success"),
    path("upgrade/cancel/", views.cancel, name="cancel"),
      path("webhook/", views.stripe_webhook, name="stripe_webhook"),
]
