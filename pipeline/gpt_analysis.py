import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")
    return OpenAI(api_key=api_key)

def generate_ai_summary(payload: dict) -> str:
    try:
        client = get_client()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional music critic. "
                        "Analyze both audio metrics AND lyrics. "
                        "Comment on themes, meaning, hooks, emotional impact, and clarity."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
Audio Metrics:
- RMS: {payload.get("rms")}
- Tempo: {payload.get("tempo")}
- Energy: {payload.get("energy")}
- Zero Crossing Rate: {payload.get("zero_crossing_rate")}
- Duration: {payload.get("duration")}

Lyrics:
{payload.get("lyrics") or "No lyrics provided."}
"""
                },
            ],
            temperature=0.6,
        )

        text = response.choices[0].message.content.strip()
        logger.info("GPT analysis generated (%d chars)", len(text))
        return text

    except Exception:
        logger.exception("GPT analysis failed")
        return ""
