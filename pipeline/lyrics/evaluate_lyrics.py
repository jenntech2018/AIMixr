# pipeline/lyrics/evaluate_lyrics.py

def evaluate_lyrics(clean_lyrics: str):
    """
    Returns extended lyric metrics and an AI-style review summary.

    This version is safe, never crashes, and always returns the keys
    the template expects — even if lyrics are empty.
    """

    if not clean_lyrics or not clean_lyrics.strip():
        return {
            "metrics": {
                "rhyme_density": 0.0,
                "clarity": 0.0,
                "cohesion": 0.0,
                "storytelling": 0.0,
                "originality": 0.0,
            },
            "review": "No lyrics available yet for analysis."
        }

    # --- TODO: real NLP logic here later ---
    # For now we return placeholder values so the UI works.

    metrics = {
        "rhyme_density": 0.42,
        "clarity": 0.75,
        "cohesion": 0.63,
        "storytelling": 0.58,
        "originality": 0.71,
    }

    review = (
        "Your lyrics demonstrate solid structure with moderate rhyme usage. "
        "Clarity is good, and the overall cohesion is strong. "
        "Storytelling elements are present but could be expanded further. "
        "Originality is above average, with several unique phrasing choices."
    )

    return {
        "metrics": metrics,
        "review": review,
    }
