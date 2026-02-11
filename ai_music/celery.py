from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_music.settings")

# Force Redis
REDIS_URL = os.environ.get(
    "REDIS_URL",
    "redis://127.0.0.1:6379/0"
)

app = Celery(
    "ai_music",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Load Django settings
app.config_from_object("django.conf:settings", namespace="CELERY")

# Discover tasks
app.autodiscover_tasks()
