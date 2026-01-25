from django.contrib import admin
from .models import ChatMessage, Battle, Vote

admin.site.register(ChatMessage)
admin.site.register(Battle)
admin.site.register(Vote)
