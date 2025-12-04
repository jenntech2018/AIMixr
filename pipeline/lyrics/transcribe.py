# pipeline/lyrics/transcribe.py
import os

def _import_whisper():
    try:
        import whisper
        return whisper
    except Exception:
        return None

def transcribe_audio_file(audio_path: str, model_name: str = "small") -> str:
    """
    Transcribe audio with the whisper python package.
    Returns the raw transcript text (may be unpunctuated).
    Raises RuntimeError on failure.
    """
    whisper = _import_whisper()
    if not whisper:
        raise RuntimeError(
            "whisper is not installed. Install with `pip install -U openai-whisper` and ensure ffmpeg is on PATH."
        )

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"No audio at: {audio_path}")

    # load model
    model = whisper.load_model(model_name)
    # transcribe (language autodetect)
    result = model.transcribe(audio_path, language=None, task="transcribe")
    text = (result.get("text") or "").strip()
    return text
