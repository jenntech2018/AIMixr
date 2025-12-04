# tracks/tasks.py

from celery import shared_task
from pipeline.analyze_track import analyze_track
from pipeline.master import auto_master
from generator.models import Track

@shared_task
def process_track_task(track_id):
    track = Track.objects.get(id=track_id)

    # --- ANALYSIS ---
    result = analyze_track(track.audio_file.path)
    track.analysis = result
    track.save()

    # --- AUTO MASTERING ---
    try:
        input_path = track.audio_file.path
        output_path = input_path.replace(".wav", "_mastered.wav").replace(".mp3", "_mastered.mp3")

        auto_master(input_path, output_path, result)

        track.master_file = output_path
        track.save()
    except Exception as e:
        print("Auto mastering failed:", e)

    return True
