# models.py
from django.db import models
from django.contrib.auth.models import User


# accounts/models.py
from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone

# class UserProfile(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE)
#     uploads_used = models.IntegerField(default=0)
#     subscription_active = models.BooleanField(default=False)
#     subscription_expires = models.DateTimeField(null=True, blank=True)

#     @property
#     def has_active_subscription(self):
#         return self.subscription_active and (
#             self.subscription_expires is None or self.subscription_expires > timezone.now()
#         )

class Track(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    lyrics = models.TextField(null=True, blank=True)    # ← NEW
    # main audio input options
    audio_file = models.FileField(upload_to="uploads/", null=True, blank=True)
    master_file = models.FileField(upload_to="masters/", null=True, blank=True)
    link = models.URLField(null=True, blank=True)

    source_type = models.CharField(
        max_length=10,
        choices=[("upload", "Uploaded File"), ("link", "External Link")],
        default="upload",
    )

    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("processing", "Processing"),
            ("done", "Done"),
            ("error", "Error"),
        ],
        default="pending"
    )

    # analysis JSON blob (rms, centroid, waveform, feedback, etc)
    analysis = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        user = self.user.username if self.user else "Unknown"
        return f"Track {self.id} by {user}"

    # ---------- CRITICAL ----------
    # Celery uses this to locate the audio file
    def get_audio_path(self):
        """
        Returns the absolute path to the audio file if uploaded.
        """
        if not self.audio_file:
            return None
        return self.audio_file.path
