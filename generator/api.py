# generator/api.py
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from generator.models import Track
from pipeline.analyze_track import score_track


def track_analysis(request, track_id):
    """
    Returns JSON for frontend charts:
      - waveform
      - rms over time
      - spectral centroid over time
      - overall score
      - feedback text
    """
    track = get_object_or_404(Track, id=track_id)

    analysis = track.analysis or {}

    return JsonResponse({
        "status": track.status,
        "score": score_track(analysis) if analysis else 0,
        "analysis": analysis,
        "waveform": analysis.get("waveform", []),
        "rms_over_time": analysis.get("rms_over_time", []),
        "centroid_over_time": analysis.get("centroid_over_time", []),
        "feedback": analysis.get("feedback", "Processing..."),
        "duration": analysis.get("duration", None),
        "error": analysis.get("error", None),
    })
