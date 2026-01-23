

# ai_music/celery.py
import os
from celery import Celery
 
# Tell Celery where Django settings live
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_music.settings")

# CREATE the Celery app
app = Celery("ai_music")

# Load settings from Django settings.py
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover tasks from all installed apps
app.autodiscover_tasks()
