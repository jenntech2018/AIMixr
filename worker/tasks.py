# worker/tasks.py
# worker/tasks.py
from celery import shared_task
from generator.models import Track, TrackAnalysis
from pipeline.analyze_track import analyze_track
import logging

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def analyze_track_task(self, track_id):
    logger.info(f"[TASK] Starting analysis for track {track_id}")

    track = Track.objects.get(id=track_id)

    analysis, _ = TrackAnalysis.objects.get_or_create(track=track)
    try:
        # mark as running
        analysis.error = None
        analysis.save(update_fields=["error", "updated_at"])

        # IMPORTANT: pass FILE PATH, not Track object
        audio_path = track.audio_file.path

        results = analyze_track(audio_path)

        if "error" in results:
            raise RuntimeError(results["error"])

        # 🔑 MAP RESULTS → MODEL FIELDS (THIS WAS THE BUG)
        analysis.waveform = results.get("waveform")
        analysis.rms = results.get("rms_over_time")
        analysis.spectral_centroid = results.get("centroid_over_time")
        analysis.ai_feedback = results.get("ai_feedback", "")
        analysis.rms = results.get("rms")
        analysis.spectral_centroid = results.get("centroid_over_time")
        analysis.vocal_pitch = results.get("vocal_pitch")
        analysis.vocal_stats = results.get("vocal_stats")
        analysis.lyrics_raw = results.get("lyrics_raw")
        analysis.error = None

        analysis.save(update_fields=[
            "waveform",
            "rms",
            "spectral_centroid",
            "vocal_pitch",
            "vocal_stats",
            "lyrics_raw",
            "error",
            "ai_feedback",
            "updated_at",
        ])

        # optional: mark track done
        track.status = "analyzed"
        track.save(update_fields=["status"])

        logger.info(f"[TASK] Analysis saved for track {track_id}")

    except Exception as e:
        logger.exception(f"[TASK] Analysis failed for track {track_id}")
        analysis.error = str(e)
        analysis.save(update_fields=["error", "updated_at"])
        raise

# accounts/tasks.py
import os
import uuid
from celery import shared_task
from django.conf import settings
from generator.models import Track  # adjust if your Track model is elsewhere
from spleeter.separator import Separator

@shared_task
def split_stems_task(track_id):
    """
    Splits a track into stems using Spleeter.
    Saves stems to media/stems/<track_id>_<uuid>/
    Returns a list of dicts: [{'name': 'vocals', 'url': '/media/stems/...'}, ...]
    """
    try:
        track = Track.objects.get(id=track_id)
        if not track.audio_file:
            return []

        # Paths
        audio_path = track.audio_file.path
        output_dir = os.path.join(settings.MEDIA_ROOT, "stems", f"{track.id}_{uuid.uuid4().hex}")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize Spleeter: 4 stems (vocals, drums, bass, other)
        separator = Separator("spleeter:4stems")
        separator.separate_to_file(audio_path, output_dir)

        stems = []
        # Spleeter saves in a subfolder named after the input file
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".wav"):
                    stem_name = file.replace(".wav", "")
                    stem_path = os.path.join(root, file)
                    stem_url = os.path.relpath(stem_path, settings.MEDIA_ROOT)
                    stem_url = f"{settings.MEDIA_URL}{stem_url.replace(os.path.sep, '/')}"
                    stems.append({"name": stem_name, "url": stem_url})

        # Optional: save to track model if you have a JSONField
        # track.stems = stems
        # track.save()

        return stems

    except Track.DoesNotExist:
        return []
    except Exception as e:
        # log error if you have logging setup
        print(f"[split_stems_task] Error: {e}")
        return []
