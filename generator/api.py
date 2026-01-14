# generator/api.py
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_GET

from generator.models import Track, TrackAnalysis


@require_GET
def track_analysis(request, track_id):
    """
    Returns JSON for frontend charts and analysis display.
    Safe for:
    - track not analyzed yet
    - analysis errors
    """

    track = get_object_or_404(Track, id=track_id)

    # Try to get analysis safely
    try:
        analysis = track.analysis_obj
    except TrackAnalysis.DoesNotExist:
        return JsonResponse({
            "status": track.status,
            "score": 0,
            "analysis": {},
            "waveform": [],
            "rms_over_time": [],
            "centroid_over_time": [],
            "feedback": "Processing…",
            "duration": None,
            "error": None,
        })

    return JsonResponse({
        "status": track.status,

        # If you later add a score field, wire it here
        "score": getattr(analysis, "score", 0),

        "analysis": {
            "lyrics_raw": analysis.lyrics_raw,
            "vocal_stats": analysis.vocal_stats,
        },

        "waveform": analysis.waveform or [],
        "rms_over_time": analysis.rms or [],
        "centroid_over_time": analysis.spectral_centroid or [],

        "feedback": getattr(analysis, "feedback", ""),
        "duration": getattr(analysis, "duration", None),
        "error": analysis.error,
    })
