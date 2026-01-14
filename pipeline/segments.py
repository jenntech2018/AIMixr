# src/segments.py
import librosa, numpy as np

def detect_sections(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    S = librosa.segment.recurrence_matrix(chroma, mode="affinity", metric="cosine")
    novelty = np.diff(S.sum(axis=0))
    peaks = np.where((novelty[1:-1] > novelty[:-2]) & (novelty[1:-1] > novelty[2:]))[0] + 1
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=512)
    return times.tolist()