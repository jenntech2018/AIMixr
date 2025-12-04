import stripe
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json

stripe.api_key = settings.STRIPE_SECRET_KEY

# ------------------------------------
#  PLAN DEFINITIONS
# ------------------------------------
PLANS = {
    "basic": {
        "name": "Basic Plan",
        "price": "4.99",
        "stripe_price_id": settings.STRIPE_PRICE_BASIC,
        "features": ["10 uploads/mo", "Standard mastering"],
    },
    "premium": {
        "name": "Premium Plan",
        "price": "9.99",
        "stripe_price_id": settings.STRIPE_PRICE_PREMIUM,
        "features": ["Unlimited uploads", "Advanced mastering"],
    },
    "studio_pro": {
        "name": "Studio Pro Plan",
        "price": "19.99",
        "stripe_price_id": settings.STRIPE_PRICE_STUDIO_PRO,
        "features": ["Priority processing", "AI stems", "AI remixing"],
    },
}

# ------------------------------------
#  UPGRADE LANDING
# ------------------------------------
def upgrade(request):
    return render(request, "upgrade.html")

# ------------------------------------
#  CHECKOUT PAGE
# ------------------------------------
def checkout(request):
    plan_key = request.GET.get("plan", "premium")

    if plan_key == "pro":
        plan_key = "studio_pro"

    plan = PLANS.get(plan_key, PLANS["premium"])

    return render(request, "checkout.html", {
        "plan": plan,
        "plan_key": plan_key,
        "STRIPE_PUBLIC_KEY": settings.STRIPE_PUBLIC_KEY,
    })

# ------------------------------------
#  CREATE CHECKOUT SESSION (AJAX)
# ------------------------------------
def create_checkout_session(request):
    plan_key = request.GET.get("plan")

    if plan_key == "pro":
        plan_key = "studio_pro"

    if plan_key not in PLANS:
        return JsonResponse({"error": "Invalid plan"}, status=400)

    price_id = PLANS[plan_key]["stripe_price_id"]

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=request.build_absolute_uri("/upgrade/success/"),
        cancel_url=request.build_absolute_uri("/upgrade/cancel/"),
    )

    return JsonResponse({"id": session.id})

# ------------------------------------
#  SUCCESS / CANCEL PAGES
# ------------------------------------
def success(request):
    return render(request, "success.html")

def cancel(request):
    return render(request, "cancel.html")

# ------------------------------------
#  STRIPE WEBHOOK
# ------------------------------------
@csrf_exempt
def stripe_webhook(request):
    payload = request.body
    sig_header = request.META.get("HTTP_STRIPE_SIGNATURE")
    webhook_secret = settings.STRIPE_WEBHOOK_SECRET

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except Exception:
        return HttpResponse(status=400)

    if event["type"] == "customer.subscription.created":
        sub = event["data"]["object"]
        print("SUB CREATED:", sub["id"])

    if event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        print("SUB DELETED:", sub["id"])

    return HttpResponse(status=200)
