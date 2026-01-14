from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="userprofile")
    is_premium = models.BooleanField(default=False)   # <-- PREMIUM FLAG
    usage_count = models.PositiveIntegerField(default=0)
    tracks = models.FileField(upload_to='tracks/', blank=True, null=True)
    subscription_active = models.BooleanField(default=False)
    subscription_expires = models.DateTimeField(null=True, blank=True)
    uploads_used = models.IntegerField(default=0)

    @property
    def plan_name(self):
        """
        Returns the current plan based on subscription status and premium flag.
        """
        if self.subscription_active:
            if self.is_premium:
                return "Studio Pro"  # Premium users are Studio Pro
            return "Premium"       # Non-premium but active subscription
        return "Free"

    def __str__(self):
        return self.user.username
