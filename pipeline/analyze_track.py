# pipeline/analyze_track.py

import os
import logging
import warnings
import numpy as np
import torch
from pydub import AudioSegment
from pipeline.gpt_analysis import generate_ai_summary

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# -------------------------
# UTILS
# -------------------------

def downsample(arr, n=1000):
    if not arr:
        return []
    step = max(1, len(arr) // n)
    return arr[::step]

def load_audio(path):
    audio = AudioSegment.from_file(path)
    samples = np.array(audio.get_array_of_samples()).astype("float32")
    
    if audio.sample_width == 2:
        samples /= 32768.0
    elif audio.sample_width == 4:
        samples /= 2147483648.0
    
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)
    
    return samples, audio.frame_rate

def zero_crossing_rate(y):
    if y.numel() < 2:
        return 0.0
    return float(
        ((torch.sign(y[1:]) - torch.sign(y[:-1])).abs() > 0)
        .float()
        .mean()
    )

def estimate_tempo(y, sr):
    if y.numel() < 2048:
        return 0.0
    diff = torch.abs(y[1:] - y[:-1])
    peaks = torch.where(diff > diff.mean() * 2)[0]
    if len(peaks) < 2:
        return 0.0
    interval = torch.diff(peaks.float()).mean().item()
    bpm = 60.0 / (interval / sr)
    return float(bpm) if 40 < bpm < 240 else 0.0

# -------------------------
# WHISPER
# -------------------------

def preprocess_for_whisper(path):
    audio = AudioSegment.from_file(path)
    audio = audio.strip_silence(
        silence_len=700,
        silence_thresh=audio.dBFS - 16
    )
    tmp = "/tmp/whisper_trimmed.wav"
    audio.export(tmp, format="wav")
    return tmp

import requests
import os
import logging

logger = logging.getLogger(__name__)

WHISPER_API_URL = "https://api.openai.com/v1/audio/transcriptions"
WHISPER_API_KEY = os.environ.get("OPENAI_API_KEY")  # your Aimixr OpenAI key

def analyze_lyrics(path: str) -> str:
    if not os.path.exists(path):
        logger.error(f"Audio file not found: {path}")
        return ""
    
    headers = {
        "Authorization": f"Bearer {WHISPER_API_KEY}"
    }
    
    files = {
        "file": (os.path.basename(path), open(path, "rb"), "audio/mpeg")
    }
    
    data = {
        "model": "whisper-1",
        "language": "en"
    }
    
    try:
        response = requests.post(WHISPER_API_URL, headers=headers, files=files, data=data)
        response.raise_for_status()
        result = response.json()
        return result.get("text", "").strip()
    except Exception as e:
        logger.exception("Whisper API call failed")
        return ""


# -------------------------
# WAVEFORM
# -------------------------
def generate_waveform(y, target_points=2000):
    y = y.detach().cpu().numpy()
    if len(y) <= target_points:
        return y.tolist()
    step = len(y) // target_points
    return [float(np.max(np.abs(y[i:i + step]))) for i in range(0, len(y), step)][:target_points]



def analyze_track(path):
    try:
        # --- AUDIO ---
        y_np, sr = load_audio(path)
        if len(y_np) == 0:
            raise ValueError("Empty audio")
        y = torch.tensor(y_np)

        duration = len(y) / sr
        rms = float(torch.sqrt(torch.mean(y ** 2)))
        peak = float(torch.max(torch.abs(y)))
        energy = float(torch.sum(y ** 2))

        spec = torch.stft(y, n_fft=2048, hop_length=512, return_complex=True)
        mag = torch.abs(spec) + 1e-8
        freqs = torch.linspace(0, sr / 2, mag.size(0))
        centroid_frames = (freqs[:, None] * mag).sum(dim=0) / mag.sum(dim=0)
        centroid_frames = centroid_frames.nan_to_num(0)
        rms_frames = torch.sqrt(mag.pow(2).mean(dim=0))

        zcr = zero_crossing_rate(y)
        tempo = estimate_tempo(y, sr)

        # --- LYRICS ---
        lyrics_raw = analyze_lyrics(path)

        # --- GPT ---
        analysis = generate_ai_summary({
            "rms": rms,
            "tempo": tempo,
            "energy": energy,
            "zero_crossing_rate": zcr,
            "duration": duration,
            "lyrics": lyrics_raw,
        })

        return {
            "sample_rate": sr,
            "duration": float(duration),
            "rms": rms,
            "peak": peak,
            "energy": energy,
            "tempo": tempo,
            "zero_crossing_rate": zcr,
            "waveform": generate_waveform(y),
            "rms_over_time": downsample(rms_frames.tolist()),
            "centroid_over_time": downsample(centroid_frames.tolist()),
            "lyrics_raw": lyrics_raw,
            "ai_feedback": analysis,
            "vocal_pitch": [],
            "vocal_stats": {},
        }
    except Exception as e:
        logger.exception("Analyze track failed")
        return {"error": str(e)}

def extract_waveform_only(path):
    """Helper to get just the waveform list for the UI."""
    try:
        y_np, _ = load_audio(path)
        if len(y_np) == 0:
            return []
        y = torch.tensor(y_np)
        return generate_waveform(y)
    except Exception as e:
        logger.error(f"Waveform extraction failed: {e}")
        return []
