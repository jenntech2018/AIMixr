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
# generator/utils.py

def user_reached_free_limit(user):
    profile = user.userprofile
    plan = profile.plan_name  # Assuming this matches your tier names
    
    if profile.is_premium:
        # Premium and Studio Pro have no limits
        if plan in ["Premium", "Studio Pro"]:
            return False
        
        # Basic has a 10-track limit
        if plan == "Basic":
            return profile.usage_count >= 10
            
    # Free users have a 1-track limit
    return profile.usage_count >= 1


  
