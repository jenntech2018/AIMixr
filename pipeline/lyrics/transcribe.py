# pipeline/lyrics/transcribe.py
import os
import time
from openai import OpenAI
from pydub import AudioSegment

# FFMPEG settings
os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"
os.environ["FFPROBE_BINARY"] = "/usr/bin/ffprobe"

AudioSegment.converter = "/usr/bin/ffmpeg"
AudioSegment.ffprobe   = "/usr/bin/ffprobe"

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def transcribe_audio_file(audio_path: str) -> str:
    """
    Transcribe audio using OpenAI Whisper API.
    Returns the raw transcript text (may be unpunctuated).
    Raises RuntimeError on failure.
    """

    print("🎤 [transcribe] function entered")
    print(f"🎤 [transcribe] audio_path={audio_path}")

    start_time = time.time()

    print("🎤 [transcribe] checking audio file exists…")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"No audio at: {audio_path}")
    print("🎤 [transcribe] audio file exists")

    print("🎤 [transcribe] starting transcription via OpenAI API…")
    transcribe_start = time.time()

    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            file=f,
            model="whisper-1"
        )

    text = (transcript.text or "").strip()

    print(f"🎤 [transcribe] transcription finished in {time.time() - transcribe_start:.2f}s")
    print(f"🎤 [transcribe] transcript length={len(text)} chars")
    print(f"🎤 [transcribe] total time={time.time() - start_time:.2f}s")

    return text
