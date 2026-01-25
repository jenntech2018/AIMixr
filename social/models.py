from django.db import models
from django.conf import settings


class ChatMessage(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']


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
    ended_at = models.DateTimeField(null=True, blank=True)
    
    winner = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='battles_won', on_delete=models.SET_NULL, null=True, blank=True)


class Vote(models.Model):
    battle = models.ForeignKey(Battle, related_name='votes', on_delete=models.CASCADE)
    voter = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)  # Null for AI
    voted_for = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='votes_received', on_delete=models.CASCADE)
    
    is_ai = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('battle', 'voter')
