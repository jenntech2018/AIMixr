# pipeline/analyze_track.py
import math
import os
import warnings

import numpy as np

# prefer torch for heavy numeric ops where available
import torch

from pydub import AudioSegment

# librosa & whisper are optional — this module will handle missing packages gracefully
try:
    import librosa
except Exception:
    librosa = None

try:
    import whisper
except Exception:
    whisper = None


# -------------------------
# Robust audio loader
# -------------------------
def load_audio(path):
    """
    Load audio using pydub and return (samples (np.float32 mono), sample_rate).
    Normalizes 16-bit/32-bit PCM into [-1.0, 1.0].
    """
    audio = AudioSegment.from_file(path)
    samples = np.array(audio.get_array_of_samples()).astype("float32")

    # normalize depending on sample width
    if audio.sample_width == 2:
        samples = samples / 32768.0
    elif audio.sample_width == 4:
        samples = samples / 2147483648.0
    else:
        # best-effort generic normalization
        samples = samples / (2 ** (8 * audio.sample_width - 1))

    # mix to mono if necessary
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
        samples = samples.mean(axis=1)

    return samples, audio.frame_rate


# -------------------------
# Helper: compute ZCR (torch)
# -------------------------
def zero_crossing_rate(y):
    """
    y: torch tensor 1D
    returns float
    """
    try:
        if y.numel() < 2:
            return 0.0
        signs = torch.sign(y)
        zcr = ((signs[1:] - signs[:-1]).abs() > 1e-6).float().mean().item()
        return float(zcr)
    except Exception:
        return 0.0


# -------------------------
# Tempo estimator (robust)
# -------------------------
def estimate_tempo(y, sr):
    """
    Simple onset-based tempo estimator using torch.
    Returns estimated BPM or 0.0 if unreliable.
    """
    try:
        if len(y) < 1024:
            return 0.0

        # onset envelope (difference signal)
        oenv = torch.abs(y[1:] - y[:-1])

        window_size = min(1024, max(16, len(oenv) // 8))
        window = torch.ones(window_size, dtype=oenv.dtype) / float(window_size)
        oenv_pad = oenv.unsqueeze(0).unsqueeze(0)
        kernel = window.unsqueeze(0).unsqueeze(0)
        padding = window_size - 1
        oenv_smooth = torch.nn.functional.conv1d(oenv_pad, kernel, padding=padding).squeeze()
        if oenv_smooth.numel() == 0:
            return 0.0

        threshold = oenv_smooth.mean() + 2 * (oenv_smooth.std() if oenv_smooth.numel() > 1 else 0)
        peaks = torch.where(oenv_smooth > threshold)[0]

        if len(peaks) < 2:
            return 0.0

        intervals = torch.diff(peaks.float()) / float(sr)
        intervals = intervals[intervals > 0.05]  # at least 50ms
        if len(intervals) == 0:
            return 0.0

        bpm = 60.0 / intervals.mean().item()
        if bpm < 40 or bpm > 240:
            return 0.0

        return float(bpm)
    except Exception:
        return 0.0


# -------------------------
# OPTIONAL: Speech-to-text (lyrics) + vocal timing / pitch
# -------------------------
def analyze_lyrics_and_vocals(path, y_np=None, sr=None):
    """
    path: path to audio file (used for whisper.transcribe if available)
    y_np, sr: optional preloaded array & sample rate (used for pitch analysis)

    Returns:
        {
            "transcription": str or None,
            "segments": [{"start":float,"end":float,"text":str}, ...],
            "words": [{"word":str,"start":float,"end":float}, ...]  # best-effort
            "vocal_pitch": [f0,...] or [],
            "vocal_stats": {"avg_pitch":..., "pitch_std": ...}
        }
    If analysis not available, returns {"error": "..."}.
    """
    out = {
        "transcription": None,
        "segments": [],
        "words": [],
        "vocal_pitch": [],
        "vocal_stats": {},
    }

    # --- Whisper transcription (if available) ---
    if whisper is not None:
        try:
            # prefer transcribing by file path (simpler & memory-friendly)
            model = whisper.load_model("small")  # small is a good balance; change if you want
            res = model.transcribe(path)
            text = res.get("text", "").strip()
            segments = res.get("segments", [])
            out["transcription"] = text
            out["segments"] = [
                {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": s.get("text", "").strip()}
                for s in segments
            ]
            # Whisper segments' 'text' may contain multi-word chunks; we keep it as segments.
            # Word-level alignment would require a separate forced-aligner; we'll expose segments.
        except Exception as e:
            out["error"] = f"whisper error: {e}"
    else:
        out["error"] = "whisper not installed"

    # --- Vocal pitch estimation using librosa.pyin ---
    if librosa is not None and y_np is not None and sr is not None and len(y_np) > 0:
        try:
            # ensure y_np is float32 mono
            y_for_lib = y_np.astype("float32")
            # librosa.pyin expects samples, sr
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y_for_lib,
                fmin=80.0,
                fmax=1000.0,
                sr=sr,
                frame_length=2048,
                hop_length=256,
            )
            # f0 is an array with np.nan where unvoiced
            f0_list = np.nan_to_num(f0, nan=0.0).astype(float).tolist()
            out["vocal_pitch"] = f0_list
            # some basic stats
            if np.count_nonzero(~np.isnan(f0)) > 0:
                valid = f0[~np.isnan(f0)]
                avg = float(np.mean(valid))
                std = float(np.std(valid))
            else:
                avg = 0.0
                std = 0.0
            out["vocal_stats"] = {"avg_pitch": avg, "pitch_std": std}
        except Exception as e:
            # keep existing error messaging but don't fail whole analysis
            out.setdefault("errors", []).append(f"librosa pitch error: {e}")

    return out


# -------------------------
# Full analysis pipeline
# -------------------------
def analyze_track(path):
    """
    Analyze audio file at `path` and return a dict with metrics and optional analysis
    """
    try:
        # -------------------------
        # LOAD AUDIO
        # -------------------------
        y_np, sr = load_audio(path)
        if y_np is None or len(y_np) == 0:
            raise ValueError("Loaded audio is empty")

        # convert to torch tensor for spectral ops
        y = torch.tensor(y_np, dtype=torch.float32)

        duration = float(len(y) / sr)

        # -------------------------
        # BASIC METRICS
        # -------------------------
        rms = float(torch.sqrt(torch.mean(y ** 2)).item()) if y.numel() > 0 else 0.0
        peak = float(torch.max(torch.abs(y)).item()) if y.numel() > 0 else 0.0
        energy = float(torch.sum(y ** 2).item()) if y.numel() > 0 else 0.0

        # -------------------------
        # SPECTRAL FEATURES
        # -------------------------
        n_fft = 2048
        if len(y) < n_fft:
            # next power of two or at least 2
            n_fft = 1 << (len(y) - 1).bit_length() if len(y) > 1 else 2
            n_fft = max(2, min(n_fft, len(y)))

        hop_length = max(128, n_fft // 4)
        win_length = n_fft

        if len(y) < n_fft:
            pad_size = n_fft - len(y)
            y_stft = torch.nn.functional.pad(y, (0, pad_size))
        else:
            y_stft = y

        spec = torch.stft(
            y_stft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            return_complex=True,
            center=True,
        )

        mag = torch.abs(spec) + 1e-8  # freq_bins x frames
        freqs = torch.linspace(0, float(sr) / 2.0, mag.size(0))

        mag_sum_per_frame = mag.sum(dim=0)
        safe_mag_sum = mag_sum_per_frame.clone()
        safe_mag_sum[safe_mag_sum == 0] = 1e-8

        centroid_per_frame = ((freqs[:, None] * mag).sum(dim=0) / safe_mag_sum).cpu()
        centroid = float(centroid_per_frame.mean().item()) if centroid_per_frame.numel() > 0 else 0.0

        bandwidth_per_frame = (((freqs[:, None] - centroid_per_frame) ** 2 * mag).sum(dim=0) / safe_mag_sum).sqrt().cpu()
        bandwidth = float(bandwidth_per_frame.mean().item()) if bandwidth_per_frame.numel() > 0 else 0.0

        log_mag = torch.log(mag + 1e-12)
        geo_mean = torch.exp(log_mag.mean(dim=0))
        arith_mean = mag.mean(dim=0)
        flatness_per_frame = (geo_mean / (arith_mean + 1e-12)).cpu()
        flatness = float(flatness_per_frame.mean().item()) if flatness_per_frame.numel() > 0 else 0.0

        mag_freq_sum = mag.sum(dim=1)
        total_energy = float(mag.sum().item())
        if total_energy <= 0:
            rolloff = 0.0
        else:
            cumsum = torch.cumsum(mag_freq_sum, dim=0)
            threshold = 0.85 * total_energy
            idxs = torch.nonzero(cumsum >= threshold, as_tuple=False)
            if idxs.numel() == 0:
                rolloff = float(freqs[-1].item())
            else:
                rolloff_idx = int(idxs[0].item())
                rolloff = float(freqs[rolloff_idx].item())

        # RMS over time & centroid over time arrays
        rms_per_frame = torch.sqrt((mag.pow(2).sum(dim=0)) / float(mag.size(0) + 1e-12)).cpu()
        rms_over_time = [float(x.item()) for x in rms_per_frame] if rms_per_frame.numel() > 0 else []
        centroid_over_time = [float(x.item()) for x in centroid_per_frame] if centroid_per_frame.numel() > 0 else []

        # -------------------------
        # ZERO‑CROSSING
        # -------------------------
        zcr = zero_crossing_rate(y)

        # -------------------------
        # TEMPO
        # -------------------------
        tempo = estimate_tempo(y, sr)

        # -------------------------
        # ATTACK TIME
        # -------------------------
        abs_y = torch.abs(y)
        attack_threshold = 0.3 * peak
        idxs_attack = torch.where(abs_y >= attack_threshold)[0]
        attack = float(idxs_attack[0] / sr) if len(idxs_attack) > 0 else 0.0

        # -------------------------
        # CREST FACTOR
        # -------------------------
        crest_factor = float(peak / (rms + 1e-12))

        # -------------------------
        # LYRIC & VOCAL ANALYSIS (optional)
        # -------------------------
        try:
            lyric_vocal = analyze_lyrics_and_vocals(path, y_np=y_np, sr=sr)
        except Exception as e:
            lyric_vocal = {"error": str(e)}

        # -------------------------
        # RESULT OBJECT
        # -------------------------
        result = {
            "sample_rate": int(sr),
            "duration": float(duration),
            "rms": float(rms),
            "peak": float(peak),
            "energy": float(energy),
            "centroid": float(centroid),
            "bandwidth": float(bandwidth),
            "flatness": float(flatness),
            "rolloff": float(rolloff),
            "zero_crossing_rate": float(zcr),
            "tempo": float(tempo),
            "attack": float(attack),
            "crest_factor": float(crest_factor),
            "rms_over_time": rms_over_time,
            "centroid_over_time": centroid_over_time,
            # include lyric/vocal analysis fields (if produced)
            "lyrics": lyric_vocal.get("transcription") if isinstance(lyric_vocal, dict) else None,
            "lyric_segments": lyric_vocal.get("segments", []) if isinstance(lyric_vocal, dict) else [],
            "vocal_pitch": lyric_vocal.get("vocal_pitch", []) if isinstance(lyric_vocal, dict) else [],
            "vocal_stats": lyric_vocal.get("vocal_stats", {}) if isinstance(lyric_vocal, dict) else {},
        }

        # Add AI feedback summary if summarize_analysis available
        try:
            result["feedback"] = summarize_analysis(result)
        except Exception:
            result["feedback"] = "No feedback available."

        return result

    except Exception as e:
        # safe serializable error
        return {"error": str(e)}


# -------------------------
# Scoring & summarizer
# -------------------------
def score_track(a):
    if not a or "error" in a:
        return 0.0

    rms = a.get("rms", 0)
    bandwidth = a.get("bandwidth", 0)
    centroid = a.get("centroid", 0)
    energy = a.get("energy", 0)
    tempo = a.get("tempo", 0)
    peak = a.get("peak", 0)
    crest = a.get("crest_factor", 0)

    return float(
        rms * 4.0
        + bandwidth * 0.001
        + centroid * 0.001
        + energy * 0.00001
        + (tempo / 200.0) * 2.0
        - peak * 2.0
        - crest * 0.5
    )


def summarize_analysis(a):
    """
    Summarize the analysis into human-readable feedback.
    Replace or extend this with your project's original summarize_analysis implementation
    if you have a more elaborate version.
    """
    try:
        rms = a.get("rms", 0)
        crest = a.get("crest_factor", 0)
        centroid = a.get("centroid", 0)
        flatness = a.get("flatness", 0)
        tempo = a.get("tempo", 0)
        attack = a.get("attack", 0)

        summary = []
        # Loudness
        if rms < 0.05:
            loud = "quiet"
            advice_loud = "Consider raising the overall level or applying gentle compression to increase perceived loudness."
        elif rms < 0.12:
            loud = "balanced"
            advice_loud = "Your loudness is in a solid range. Minor compression or limiting could enhance punch."
        else:
            loud = "loud"
            advice_loud = "Track may be too loud or over-compressed. Check for distortion and reduce limiting."

        summary.append(f"• **Loudness:** Your track sounds **{loud}**. {advice_loud}")

        # Dynamics
        if crest > 10:
            dynamics = "very dynamic (large difference between soft and loud parts)"
            advice_dyn = "Use compression to reduce the dynamic range if you want a more modern, punchy sound."
        elif crest > 6:
            dynamics = "moderately dynamic"
            advice_dyn = "Dynamics are healthy. Light compression could tighten the mix."
        else:
            dynamics = "compressed or limited"
            advice_dyn = "Reduce compression/limiting to restore natural dynamics."

        summary.append(f"• **Dynamics:** Your track is **{dynamics}**. {advice_dyn}")

        # Tone
        if centroid < 1500:
            tone = "warm / bass‑heavy"
            advice_tone = "Consider adding some high‑end EQ for clarity if desired."
        elif centroid < 3000:
            tone = "balanced"
            advice_tone = "Your tonal balance looks natural."
        else:
            tone = "bright / treble‑forward"
            advice_tone = "You might soften the highs with EQ to avoid harshness."

        summary.append(f"• **Tone:** Your track sounds **{tone}**. {advice_tone}")

        # Texture
        if flatness < 0.15:
            texture = "clean and tonal"
        elif flatness < 0.3:
            texture = "mixed harmonic/noise content"
        else:
            texture = "noisy or very dense"

        summary.append(f"• **Texture:** The audio texture is **{texture}**.")

        # Tempo
        if tempo < 20 or tempo > 250:
            tempo_info = "Tempo could not be reliably estimated."
        else:
            tempo_info = f"Estimated tempo: **{tempo:.1f} BPM**."

        summary.append(f"• **Tempo:** {tempo_info}")

        summary.append(f"• **Attack:** Sound reaches 30% of peak after **{attack:.3f} seconds**.")

        # Vocal & lyrics hints
        if a.get("vocal_stats"):
            vp = a["vocal_stats"]
            avg_pitch = vp.get("avg_pitch", 0.0)
            pitch_std = vp.get("pitch_std", 0.0)
            summary.append(f"• **Vocal average pitch:** {avg_pitch:.1f} Hz (std {pitch_std:.1f}).")

        summary.append("\n### 🎧 Tips for improvement\nTry small EQ moves, gentle compression, and checking levels against reference tracks you like. Subtle tweaks make the biggest difference!")

        return "\n".join(summary)
    except Exception as e:
        return f"No summary available: {e}"
