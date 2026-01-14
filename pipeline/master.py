# pipeline/master.py
import os
import tempfile
from pydub import AudioSegment, effects
from pydub.silence import detect_nonsilent

def master_track_audio(input_path, export_format="mp3", target_lufs=None):
    """
    Simple mastering pipeline:
      - load via pydub (requires ffmpeg)
      - normalize (peak normalize)
      - basic dynamic compression / limiter using pydub.effects.compress_dynamic_range
      - gentle high-frequency softening (low-pass) if too bright (optional)
      - export mastered file, return path

    Returns:
        path to mastered file (string)
    """
    
    # load
    track = AudioSegment.from_file(input_path)

    # 1) Basic trim of long leading/trailing silence (safe)
    try:
        nonsilent = detect_nonsilent(track, min_silence_len=200, silence_thresh=-50)
        if nonsilent:
            start = max(0, nonsilent[0][0] - 50)
            end = min(len(track), nonsilent[-1][1] + 50)
            track = track[start:end]
    except Exception:
        # if silence detection fails, continue with full track
        pass

    # 2) Normalize peak (makes sure peaks hit near 0 dBFS without clipping)
    track = effects.normalize(track)

    # 3) Gentle dynamic compression / limiting
    try:
        # pydub has compress_dynamic_range which is a simple compressor
        # parameters can be tuned; this is conservative
        track = effects.compress_dynamic_range(
            track,
            threshold=-14.0,    # dBFS threshold
            ratio=3.0,          # compression ratio
            attack=5,           # ms
            release=200,        # ms
            knee=5.0,
            makeup_gain_db=2.0
        )
    except Exception:
        # if compress_dynamic_range not available/configs fail, ignore
        pass

    # 4) Simple ceiling limiter: reduce any samples louder than -0.5 dBFS
    peak_db = track.max_dBFS
    if peak_db > -0.5:
        reduce_db = peak_db + 0.5
        track = track.apply_gain(-reduce_db)

    # 5) Optional: gentle high-shelf reduction if track is very bright (heuristic)
    # (pydub doesn't have advanced eq; you could integrate ffmpeg filters here later)

    # 6) Export to temporary mastered file
    base, ext = os.path.splitext(input_path)
    out_name = f"{base}_master.{export_format}"
    # prefer to export as mp3 if input not wav; keep container consistent
    try:
        track.export(out_name, format=export_format, bitrate="192k")
    except Exception:
        # fallback: export wav if mp3 export fails
        out_name = f"{base}_master.wav"
        track.export(out_name, format="wav")
    return out_name
