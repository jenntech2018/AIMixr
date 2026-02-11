# generator/utils.py



def celery_available():
    """
    Checks whether Celery is running by trying to ping it.
    Returns True if Celery is available, otherwise False.
    """
    try:
        from celery import Celery
        app = Celery('generator')
        result = app.control.ping(timeout=0.5)

        return bool(result)
    except Exception:
        return False
    
# generator/utils.py
# generator/utils.py

# generator/utils.py

def user_reached_free_limit(user):
    """
    Returns True if the user has reached their plan's upload limit.
    Limits:
      - Free: 1 track
      - Basic: 10 tracks
      - Premium / Studio Pro: unlimited
    """
    try:
        profile = user.userprofile
    except Exception:
        # If for some reason the user has no profile, assume free limit reached
        return True

    plan = profile.plan_name or "Free"

    if plan in ["Premium", "Studio Pro"]:
        return False  # unlimited
    elif plan == "Basic":
        return profile.usage_count >= 10
    else:  # Free users
        return profile.usage_count >= 3


