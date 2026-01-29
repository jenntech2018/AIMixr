from django.db import models
from django.contrib.auth.models import User

class Track(models.Model):
    private = models.BooleanField(default=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    lyrics = models.TextField(null=True, blank=True)
    source_type = models.CharField(max_length=20, default="upload")
    audio_file = models.FileField(upload_to="uploads/", null=True, blank=True)
    master_file = models.FileField(upload_to="masters/", null=True, blank=True)

    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("processing", "Processing"),
            ("analyzed", "Analyzed"),
            ("error", "Error"),
        ],
        default="pending",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Track {self.id}"

    def get_audio_path(self):
        return self.audio_file.path if self.audio_file else None


class TrackAnalysis(models.Model):
    track = models.OneToOneField(
        Track,
        on_delete=models.CASCADE,
        related_name="analysis_obj"
    )
    ai_feedback = models.TextField(null=True, blank=True)
    # core features
    waveform = models.JSONField(null=True, blank=True)
    rms = models.JSONField(null=True, blank=True)
    spectral_centroid = models.JSONField(null=True, blank=True)

    # lyrics / vocals
    lyrics_raw = models.TextField(null=True, blank=True)
    vocal_pitch = models.JSONField(null=True, blank=True)
    vocal_stats = models.JSONField(null=True, blank=True)

    # meta
    error = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Analysis for Track {self.track_id}"
