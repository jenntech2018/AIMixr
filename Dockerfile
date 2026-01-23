FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip to a version that supports the legacy resolver
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# FORCE THE INSTALL using the legacy resolver to bypass "resolution-too-deep"
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

COPY . .

# Collect static files (may need dummy env vars if your settings.py requires them)
RUN python manage.py collectstatic --noinput || true

CMD gunicorn ai_music.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0