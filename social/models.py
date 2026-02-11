from django.db import models
from django.conf import settings
from generator.models import Track
from django.utils import timezone            # ADD THIS
from datetime import timedelta


class ChatMessage(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']


from django.db import models
from django.conf import settings
from django.utils import timezone            # ADD THIS
from datetime import timedelta               # ADD THIS
from generator.models import Track

# ... ChatMessage stays the same ...

class Battle(models.Model):
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('active', 'Active'),
        ('finished', 'Finished'),
    )
    challenger = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='battles_initiated', on_delete=models.CASCADE)
    opponent = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='battles_received', on_delete=models.CASCADE)
    
    challenger_track = models.ForeignKey('generator.Track', related_name='battles_as_challenger', on_delete=models.CASCADE)
    opponent_track = models.ForeignKey('generator.Track', related_name='battles_as_opponent', on_delete=models.CASCADE, null=True, blank=True)
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    start_time = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    duration_hours = models.IntegerField(default=24)
    winner = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='battles_won', on_delete=models.SET_NULL, null=True, blank=True)

    @property
    def challenger_votes(self):
        # We use 'votes' because your Vote model has related_name='votes'
        return self.votes.filter(voted_for=self.challenger).count()

    @property
    def opponent_votes(self):
        return self.votes.filter(voted_for=self.opponent).count()

    @property
    def time_remaining(self):
        if self.status == 'active' and self.start_time:
            expiry = self.start_time + timedelta(hours=self.duration_hours)
            remaining = expiry - timezone.now()
            return max(remaining, timedelta(0))
        return timedelta(0)


class Vote(models.Model):
    battle = models.ForeignKey(Battle, related_name='votes', on_delete=models.CASCADE)
    voter = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)  # Null for AI
    voted_for = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='votes_received', on_delete=models.CASCADE)

    is_ai = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('battle', 'voter')
