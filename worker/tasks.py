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

        # Run your analysis pipeline
        results = analyze_track(audio_path)

        # If pipeline returned an error, surface it
        if "error" in results:
            raise RuntimeError(results["error"])

        # Map results → model fields
        analysis.waveform = results.get("waveform")
        analysis.rms = results.get("rms")
        analysis.spectral_centroid = results.get("centroid_over_time")
        analysis.vocal_pitch = results.get("vocal_pitch")
        analysis.vocal_stats = results.get("vocal_stats")
        analysis.lyrics_raw = results.get("lyrics_raw")
        analysis.ai_feedback = results.get("ai_feedback", "")
        analysis.error = None

        analysis.save(update_fields=[
            "waveform",
            "rms",
            "spectral_centroid",
            "vocal_pitch",
            "vocal_stats",
            "lyrics_raw",
            "ai_feedback",
            "error",
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
    logger.info(f"[TASK] Analysis completed for track {track_id}")


import os
import subprocess
from celery import shared_task
from generator.models import Track


@shared_task(bind=True)
def split_stems_task(self, track_id):
    try:
        track = Track.objects.get(id=track_id)

        input_file = track.audio_file.path
        output_dir = "media/stems"

        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "demucs",
            "-o", output_dir,
            input_file
        ]

        subprocess.run(cmd, check=True)

        # Optional: mark status
        track.status = "stems_ready"
        track.save()

        return "Done"

    except Exception as e:
        track.status = "error"
        track.save()
        raise e
