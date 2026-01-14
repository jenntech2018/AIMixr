# generator/serializers.py
from rest_framework import serializers
from .models import Track

class TrackSerializer(serializers.ModelSerializer):
    audio_file_url = serializers.SerializerMethodField()
    master_file_url = serializers.SerializerMethodField()

    class Meta:
        model = Track
        fields = [
            "id",
            "user",
            "audio_file_url",
            "master_file_url",
            "status",
            "analysis",
            "created_at",
        ]

    def get_audio_file_url(self, obj):
        try:
            return obj.audio_file.url if obj.audio_file else None
        except:
            return None

    def get_master_file_url(self, obj):
        try:
            return obj.master_file.url if obj.master_file else None
        except:
            return None
