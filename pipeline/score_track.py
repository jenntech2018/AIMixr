from .analyze_track import score_track


def score_track(analysis: dict) -> float:
    # Example: score by tempo and clarity
    if not analysis or "features" not in analysis:
        return 0.0
    feats = analysis["features"]
    return feats.get("tempo_bpm", 0) + feats.get("clarity_centroid", 0)
