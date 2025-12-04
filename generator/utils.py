# generator/utils.py

import profile


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
def user_reached_free_limit(user):
    profile = user.userprofile
    return (not profile.is_premium) and profile.usage_count >= 1



  
