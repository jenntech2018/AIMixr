#!/bin/bash

# Apply migrations
python manage.py migrate --noinput

# Collect static files
python manage.py collectstatic --noinput

# Start server
gunicorn ai_music.wsgi:application --bind 0.0.0.0:$PORT
