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
	"free_trial": False,
        "name": "Basic Plan",
        "price": "4.99",
        "stripe_price_id": settings.STRIPE_PRICE_BASIC,
        "features": ["10 uploads/mo", "Standard mastering"],
    },
    "premium": {
	"free_trial": False,
        "name": "Premium Plan",
        "price": "9.99",
        "stripe_price_id": settings.STRIPE_PRICE_PREMIUM,
        "features": ["Unlimited uploads", "Advanced mastering"],
    },
    "studio_pro": {
	"free_trial": False,
        "name": "Studio Pro Plan",
        "price": "19.99",
        "stripe_price_id": settings.STRIPE_PRICE_STUDIO_PRO,
        "features": ["Priority processing", "AI stems", "AI remixing"],
    },
}
from google.oauth2 import service_account
from googleapiclient.discovery import build
from django.conf import settings

# Path to your downloaded JSON key
SERVICE_ACCOUNT_FILE = '/opt/aimixr/service-account.json'
PACKAGE_NAME = 'com.aimixr.app' # Replace with your actual package name

@csrf_exempt
def verify_google_purchase(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            purchase_token = data.get('purchaseToken')
            product_id = data.get('productId') # e.g., "studio_pro_monthly"

            # 1. Authenticate with Google
            scopes = ['https://www.googleapis.com/auth/androidpublisher']
            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=scopes
            )
            service = build('androidpublisher', 'v3', credentials=creds)

            # 2. Call Google to verify the subscription
            # Use .subscriptions() for monthly plans, .inappproducts() for one-time buys
            result = service.purchases().subscriptions().get(
                packageName=PACKAGE_NAME,
                subscriptionId=product_id,
                token=purchase_token
            ).execute()

            # 3. Check if the purchase is valid (0 = Success/Active)
            # For subscriptions, check 'startTimeMillis' and absence of 'cancelReason'
            if 'startTimeMillis' in result:
                # 4. Update the User's Profile
                user_profile = request.user.userprofile
                
                # Map Google Product ID to your internal plan name
                plan_mapping = {
                    "basic_monthly": "Basic",
                    "premium_monthly": "Premium",
                    "studio_pro_monthly": "Studio Pro"
                }
                
                user_profile.plan_name = plan_mapping.get(product_id, "Free")
                user_profile.save()

                return JsonResponse({"status": "success", "message": f"Upgraded to {user_profile.plan_name}"})
            
            return JsonResponse({"status": "error", "message": "Invalid purchase token"}, status=400)

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"status": "error"}, status=400)

# from django.shortcuts import get_object_or_404, redirect
# from generator.models import Track  # your Track model is in generator
# from worker.tasks import split_stems_task
# from django.contrib.auth.decorators import login_required

# @login_required
# def split_stems(request, track_id):
#     track = get_object_or_404(Track, id=track_id)

#     # Only allow Studio Pro users
#     if request.user.userprofile.plan_name != "Studio Pro":
#         return redirect("track_detail", track_id=track.id)

#     if request.method == "POST":
#         # Mark track as processing
#         track.status = "processing"
#         track.save()

#         # Schedule Celery task
#         split_stems_task.delay(track.id)

#         # Redirect back to track detail page
#         return redirect("track_detail", track_id=track.id)

#     # If GET, just redirect
#     return redirect("track_detail", track_id=track.id)



# ------------------------------------
#  PROFILE VIEW
# ------------------------------------
from django.contrib.auth.decorators import login_required
from django.shortcuts import render

@login_required
def profile_view(request):
    user = request.user
    profile = user.userprofile  # you already have this model

    context = {
        "user": user,
        "profile": profile,
    }
    return render(request, "profile.html", context)


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
