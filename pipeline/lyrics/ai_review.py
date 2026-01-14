import re
import statistics
from openai import OpenAI

client = OpenAI()

def evaluate_lyrics(clean_lyrics: str):
    """
    Analyze lyrics using OpenAI plus lightweight local metrics.
    Returns a dict with 'metrics' and 'review'.
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

    # --- Local heuristic metrics ---
    words = re.findall(r"\b\w+\b", clean_lyrics.lower())
    lines = [line.strip() for line in clean_lyrics.splitlines() if line.strip()]
    endings = [w[-3:] for w in [line.split()[-1] for line in lines if line.split()]]
    rhyme_pairs = sum(1 for i in range(len(endings)-1) if endings[i] == endings[i+1])
    rhyme_density = rhyme_pairs / max(1, len(endings))
    clarity = 1.0 - (statistics.mean(len(w) for w in words) / 10.0)
    cohesion = len(set(words)) / max(1, len(words))
    storytelling_words = {"i", "he", "she", "we", "they", "story", "life", "day", "night"}
    storytelling = sum(1 for w in words if w in storytelling_words) / max(1, len(words))
    originality = 1.0 - cohesion

    metrics = {
        "rhyme_density": round(rhyme_density, 2),
        "clarity": round(max(0.0, min(1.0, clarity)), 2),
        "cohesion": round(cohesion, 2),
        "storytelling": round(storytelling, 2),
        "originality": round(originality, 2),
    }

    # --- AI critique via OpenAI ---
    prompt = (
        "Analyze the following rap lyrics. "
        "Comment on rhyme density, clarity, cohesion, storytelling, and originality. "
        "Provide a short constructive critique:\n\n"
        f"{clean_lyrics}\n\nAnalysis:"
    )

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        review = response.output_text
    except Exception as e:
        review = f"AI review failed: {e}"

    return {
        "metrics": metrics,
        "review": review,
    }
