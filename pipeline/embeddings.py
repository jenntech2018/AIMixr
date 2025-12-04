# src/embeddings.py
import numpy as np, openl3, librosa

def track_embedding(path, sr=48000):
    y, sr = librosa.load(path, sr=sr, mono=True)
    emb, ts = openl3.get_audio_embedding(y, sr, content_type="music", embedding_size=512)
    v = emb.mean(axis=0)  # average over time
    v = v / (np.linalg.norm(v) + 1e-8)
    return v.astype(np.float32)