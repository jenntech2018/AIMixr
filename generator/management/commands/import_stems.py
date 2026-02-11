import os

from django.core.management.base import BaseCommand
from django.conf import settings

from generator.models import Track, Stem


class Command(BaseCommand):
    help = "Import Demucs stems into database"

    def handle(self, *args, **kwargs):

        base = os.path.join(settings.MEDIA_ROOT, "stems")

        if not os.path.exists(base):
            self.stdout.write("No stems folder found.")
            return

        for track_folder in os.listdir(base):

            track_path = os.path.join(base, track_folder)

            if not os.path.isdir(track_path):
                continue

            # Match folder name to Track ID
            try:
                track_id = int(track_folder)
                track = Track.objects.get(id=track_id)
            except (ValueError, Track.DoesNotExist):
                self.stdout.write(f"Skipping {track_folder} (no track match)")
                continue

            for f in os.listdir(track_path):

                # Only process .wav files
                if not f.lower().endswith(".wav"):
                    continue

                file_path = f"stems/{track_folder}/{f}"

                # Skip if already exists
                if Stem.objects.filter(track=track, file=file_path).exists():
                    continue

                # Create Stem record
                Stem.objects.create(
                    track=track,
                    name=f.replace(".wav", ""),
                    file=file_path
                )

                self.stdout.write(f"Imported {f} → {track.audio_file}")
