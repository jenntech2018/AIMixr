# src/features.py
import librosa, numpy as np

def load_audio(path, sr=44100):
    y, _sr = librosa.load(path, sr=sr, mono=True)
    y = librosa.util.array_ops.fix_length(y, int(sr*min(len(y)/_sr, 600)))  # cap 10 min
    return y, sr

def basic_features(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_idx = np.argmax(chroma.mean(axis=1))
    rms = librosa.feature.rms(y=y).mean()
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    return {
        "tempo_bpm": float(tempo),
        "key_index": int(key_idx),  # map 0–11 to C, C#, ..., B
        "rms": float(rms),
        "clarity_centroid": float(spec_centroid),
        "clarity_rolloff": float(spec_rolloff),
    }

def mel_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)