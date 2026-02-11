"""
Django settings for ai_music project.
Production-ready for Google Cloud Run, but works locally for development.
"""

from pathlib import Path
import os
import sys
from dotenv import load_dotenv

#-------------------------------------------------------------------------------
# LOGGING 
#------------------------------------------------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{levelname}] {asctime} {name} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": "/var/log/aimixr/django-debug.log",
            "formatter": "verbose",
        },
    },
    "loggers": {
        "django": {
            "handlers": ["file"],
            "level": "DEBUG",
            "propagate": True,
        },
    },
}






# -------------------------------------------------------------------------------
# Base
# -------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Ensure /usr/bin is in PATH so pydub can find ffmpeg
# This fixes issues where systemd services have restricted PATHs
current_path = os.environ.get("PATH", "")
if "/usr/bin" not in current_path:
    os.environ["PATH"] = current_path + os.pathsep + "/usr/bin"

load_dotenv(BASE_DIR / ".env")
FFMPEG_BINARY=os.environ.get('FFMPEG_BINARY')
FFPROBE_BINARY=os.environ.get('FFPROBE_BINARY')

# -------------------------------------------------------------------------------
# Security
# -------------------------------------------------------------------------------

SECRET_KEY = os.environ.get("SECRET_KEY", "unsafe-dev-key")

# Detect if running in production
DEBUG = True
# Force Debug if running locally via runserver to prevent SSL redirects
if 'runserver' in sys.argv:
    DEBUG = True

if DEBUG:
    print("--- RUNNING IN DEBUG MODE (SSL OFF) ---")
else:
    print("--- RUNNING IN PRODUCTION MODE (SSL ON) ---")

if DEBUG:
    SECURE_SSL_REDIRECT = False
    SESSION_COOKIE_SECURE = False
    CSRF_COOKIE_SECURE = False
    SECURE_HSTS_SECONDS = 0
else:
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
# -------------------------------------------------------------------------------
# Hosts & CSRF
# -------------------------------------------------------------------------------

if DEBUG:
    ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1", 
    "www.aimixr.online",
    "aimixr.online",
]
    CSRF_TRUSTED_ORIGINS = ["https://aimixr.online"]
else:
    ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "35.197.84.92",  # your server IP
    "aimixr.online",

    "www.aimixr.online",
    ]
    CSRF_TRUSTED_ORIGINS = [
        "https://aimixr.online",
        "https://*.run.app",
    ]

CELERY_BROKER_URL = os.environ.get(
    "REDIS_URL",
    "redis://127.0.0.1:6379/0"
)

CELERY_RESULT_BACKEND = CELERY_BROKER_URL


# -------------------------------------------------------------------------------
# SSL / Cookies
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Applications
# -------------------------------------------------------------------------------

# Only for local testing in Docker
if os.environ.get("DJANGO_LOCAL") == "1":
    DEBUG = True
    SECURE_SSL_REDIRECT = False   # disable HTTPS redirect locally
    SESSION_COOKIE_SECURE = False
    CSRF_COOKIE_SECURE = False
    ALLOWED_HOSTS = ["*"]         # accept all hosts locally
    CSRF_TRUSTED_ORIGINS = []

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sites",  # Required by allauth

    # Third-party
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.google",

    # your apps
    "generator",
    "accounts",
    "worker",
    "social",
]

AUTH_USER_MODEL = "auth.User"


SOCIALACCOUNT_LOGIN_ON_GET = True
SOCIALACCOUNT_STORE_TOKENS = True
ACCOUNT_LOGOUT_ON_GET = True

# Force redirect-based OAuth (no embedded flows)
SOCIALACCOUNT_QUERY_EMAIL = True

ACCOUNT_DEFAULT_HTTP_PROTOCOL = "https"

SOCIALACCOUNT_ADAPTER = "allauth.socialaccount.adapter.DefaultSocialAccountAdapter"


# -------------------------------------------------------------------------------
# Middleware
# -------------------------------------------------------------------------------

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "allauth.account.middleware.AccountMiddleware",
]

ROOT_URLCONF = "ai_music.urls"

# -------------------------------------------------------------------------------
# Templates
# -------------------------------------------------------------------------------

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "ai_music.wsgi.application"

# -------------------------------------------------------------------------------
# Database
# -------------------------------------------------------------------------------

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# -------------------------------------------------------------------------------
# Password validation
# -------------------------------------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# -------------------------------------------------------------------------------
# Internationalization
# -------------------------------------------------------------------------------

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# -------------------------------------------------------------------------------
# Static & Media
# -------------------------------------------------------------------------------

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# settings.py
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Upload limits (50MB)
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024
# FILE_UPLOAD_TEMP_DIR = "/tmp"

# -------------------------------------------------------------------------------
# Auth redirects
# -------------------------------------------------------------------------------

LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/dashboard/"
LOGOUT_REDIRECT_URL = "/login/"

# -------------------------------------------------------------------------------
# Allauth / Google
# -------------------------------------------------------------------------------

AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
]

SITE_ID = 1

SOCIALACCOUNT_PROVIDERS = {
    'google': {
        'APP': {
            'client_id': os.environ.get('GOOGLE_CLIENT_ID'),
            'secret': os.environ.get('GOOGLE_CLIENT_SECRET'),
            'key': ''
        },
        'SCOPE': ['profile', 'email'],
        'AUTH_PARAMS': {'access_type': 'online'},
    }
}

ACCOUNT_EMAIL_VERIFICATION = 'none'  # or 'optional'
# -------------------------------------------------------------------------------
# Celery
# -------------------------------------------------------------------------------

CELERY_BROKER_URL = os.environ.get("REDIS_URL")
CELERY_RESULT_BACKEND = os.environ.get("REDIS_URL")

# -------------------------------------------------------------------------------
# Third-party keys
# -------------------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

STRIPE_PUBLIC_KEY = os.environ.get("STRIPE_PUBLIC_KEY")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_PRICE_BASIC = os.environ.get("STRIPE_PRICE_BASIC")
STRIPE_PRICE_PREMIUM = os.environ.get("STRIPE_PRICE_PREMIUM")
STRIPE_PRICE_STUDIO_PRO = os.environ.get("STRIPE_PRICE_STUDIO_PRO")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")

# -------------------------------------------------------------------------------
# Default
# -------------------------------------------------------------------------------

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
