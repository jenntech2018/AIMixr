# fix_tracks.py
import os
from generator.models import Track
from django.conf import settings

uploads_dir = os.path.join(settings.MEDIA_ROOT, "uploads")

for track in Track.objects.all():
    expected_name = track.audio_file.name.split("/")[-1]  # filename Django expects
    expected_path = os.path.join(uploads_dir, expected_name)

    if not os.path.exists(expected_path):
        # Try to find a close match (partial filename match)
        partial_name = expected_name.split("_")[0]  # before first underscore
        matches = [f for f in os.listdir(uploads_dir) if partial_name in f]
        if matches:
            real_file = matches[0]
            real_path = os.path.join(uploads_dir, real_file)
            print(f"Renaming {real_file} -> {expected_name}")
            os.rename(real_path, expected_path)
        else:
            print(f"No match found for {expected_name}")
