def activate_premium(user):
    profile = user.userprofile
    profile.is_premium = True
    profile.save()

def deactivate_premium(user):
    profile = user.userprofile
    profile.is_premium = False
    profile.save()
