# workers/tasks.py
from celery import shared_task
from django.conf import settings
from django.core.files import File
from generator.models import Track
import os

# workers/tasks.py
import os
import traceback
from celery import shared_task
from django.core.files import File
from django.conf import settings

# Import Django model lazily (Django app loaded when worker runs)
from generator.models import Track

# pipeline utilities
from pipeline.lyrics.transcribe import transcribe_audio_file
from pipeline.lyrics.clean_and_analyze import clean_and_analyze
from pipeline.analyze_track import analyze_track  # your existing analyzer
from pipeline.master import master_track_audio      # optional if needed

@shared_task(bind=True)
def analyze_track_task(self, track_id):
    """
    Celery task that:
     1) loads Track
     2) runs audio analysis (existing function)
     3) runs whisper transcription (transcribe_audio_file)
     4) cleans lyrics and attaches lyrics_* data to analysis JSON
     5) saves results on the Track model
    """
    try:
        track = Track.objects.get(id=track_id)
    except Track.DoesNotExist:
        return {"error": "Track not found"}

    # mark queued -> processing
    track.status = "processing"
    track.save()

    analysis = {}

    # 1) run existing analysis
    try:
        analysis = analyze_track(track.audio_file.path) or {}
    except Exception as e:
        analysis = {"error": f"analyze_track failed: {e}"}
        # continue to attempt transcription/lyrics

    # 2) Transcription
    raw_lyrics = ""
    try:
        raw_lyrics = transcribe_audio_file(track.audio_file.path, model_name="small")
    except Exception as e:
        raw_lyrics = ""
        analysis.setdefault("transcription_error", str(e))
        # log
        print("Transcription error:", e, traceback.format_exc())

    # 3) Clean & analyze lyrics
    try:
        lyrics_info = clean_and_analyze(raw_lyrics, prefer_openai=True)
    except Exception as e:
        lyrics_info = {
            "lyrics_raw": raw_lyrics,
            "lyrics_clean": "",
            "lyrics_feedback": f"clean_and_analyze error: {e}",
            "lyrics_quality": 0.0
        }
        print("Lyrics cleaning error:", e, traceback.format_exc())

    # merge into analysis
    analysis["lyrics_raw"] = lyrics_info.get("lyrics_raw", "")
    analysis["lyrics_clean"] = lyrics_info.get("lyrics_clean", "")
    analysis["lyrics_feedback"] = lyrics_info.get("lyrics_feedback", "")
    analysis["lyrics_quality"] = lyrics_info.get("lyrics_quality", 0.0)
    analysis["rhyme_density"] = lyrics_info.get("rhyme_density", 0.0)
    analysis["profanity_score"] = lyrics_info.get("profanity_score", 0.0)
    analysis["sentiment_hint"] = lyrics_info.get("sentiment_hint", "")

    # 4) scoring & finalize
    try:
        # if you have a separate scoring function already used elsewhere, call it
        from pipeline.analyze_track import score_track
        score = score_track(analysis)
        analysis["score"] = float(score)
    except Exception:
        analysis.setdefault("score_error", "scoring failed")

    # save analysis in Track
    track.analysis = analysis
    track.status = "done"
    track.save()

    return {"ok": True, "track_id": track.id}

@shared_task(bind=True)
def analyze_track_task(self, track_id):
    """
    1) Load track
    2) Run analysis (pipeline.analyze_track.analyze_track)
    3) Save analysis to track.analysis and status
    4) Optionally run auto_master and attach master_file
    """
    try:
        track = Track.objects.get(id=track_id)
    except Track.DoesNotExist:
        return {"error": "Track not found"}

    # import heavy modules lazily
    from pipeline.analyze_track import analyze_track
    from pipeline import master as master_module  # pipeline.master.auto_master

    # ensure audio exists
    if not track.audio_file:
        track.status = "error"
        track.analysis = {"error": "No audio file attached"}
        track.save()
        return {"error": "No audio file"}

    track.status = "processing"
    track.save()

    try:
        # analyze
        result = analyze_track(track.audio_file.path)
        track.analysis = result
        track.status = "done"
        track.save()
    except Exception as e:
        track.analysis = {"error": str(e)}
        track.status = "error"
        track.save()
        return {"error": str(e)}

    # Attempt auto mastering (non-fatal)
    try:
        input_path = track.audio_file.path
        base, ext = os.path.splitext(os.path.basename(input_path))
        output_fname = f"{base}_mastered{ext}"
        output_path = os.path.join(settings.MEDIA_ROOT, "masters", output_fname)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # call pipeline.master.auto_master (may use pedalboard)
        master_module.auto_master(input_path, output_path, result)

        # attach as FileField (relative path -> media storage)
        # Save with open file handle
        with open(output_path, "rb") as f:
            django_file = File(f)
            # Save to master_file field
            track.master_file.save(f"masters/{output_fname}", django_file, save=True)

    except Exception as e:
        # log but do not overwrite analysis
        print("Auto mastering failed:", e)

    return {"ok": True}
