# pipeline/analysis/audio_utils.py
import librosa
import numpy as np

def analyze_in_chunks(file_path, chunk_duration=30.0, sr=22050):
    total_duration = librosa.get_duration(filename=file_path)
    results = []
    for start in np.arange(0, total_duration, chunk_duration):
        y, _ = librosa.load(file_path, sr=sr, offset=start, duration=chunk_duration)
        if len(y) == 0:
            continue
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        results.append({
            "start": start,
            "end": min(start + chunk_duration, total_duration),
            "rms": float(rms),
            "zcr": float(zcr),
            "centroid": float(centroid),
        })
    return results
