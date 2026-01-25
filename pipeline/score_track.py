from .analyze_track import *


def score_track(analysis: dict) -> float:
    """
    Calculates a score (0-100) based on track analysis metrics.
    """
    if not analysis or "error" in analysis:
        return 0.0
    
    # Extract metrics (with defaults to prevent errors)
    tempo = analysis.get("tempo", 0)
    rms = analysis.get("rms", 0)
    zcr = analysis.get("zero_crossing_rate", 0)
    
    # Simple scoring algorithm
    # 1. Tempo score (preference for 90-140 BPM)
    tempo_score = 30 if 90 <= tempo <= 140 else 15
    
    # 2. Loudness/Presence (RMS) & Complexity (ZCR)
    presence_score = min(rms * 100, 30) + min(zcr * 100, 20)
    
    return min(tempo_score + presence_score + 20, 100)
