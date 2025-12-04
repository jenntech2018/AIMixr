from pedalboard import Pedalboard, Compressor, Gain, HighShelfFilter, LowShelfFilter
from pedalboard.io import AudioFile
import numpy as np

def auto_master(path_in, path_out, analysis):
    board = Pedalboard()

    # 🔊 Loudness adjustments
    if analysis["rms"] < 0.05:
        board.append(Gain(gain_db=6))   # boost 6 dB
    elif analysis["rms"] > 0.12:
        board.append(Gain(gain_db=-3))  # reduce slightly

    # 🎚️ Dynamics control
    if analysis["crest_factor"] > 10:
        board.append(Compressor(threshold_db=-18, ratio=4.0))
    elif analysis["crest_factor"] < 6:
        board.append(Compressor(threshold_db=-8, ratio=1.5))

    # 🎧 Tone balance
    if analysis["centroid"] < 1500:
        # too warm → brighten
        board.append(HighShelfFilter(cutoff_frequency_hz=5000, gain_db=3))
    elif analysis["centroid"] > 3000:
        # too bright → soften
        board.append(HighShelfFilter(cutoff_frequency_hz=6000, gain_db=-3))

    # --------------------------------
    # PROCESS THE AUDIO
    # --------------------------------
    with AudioFile(path_in) as f:
        audio = f.read(f.frames)
        sr = f.samplerate

    processed = board(audio, sr)

    with AudioFile(path_out, 'w', sr, processed.shape[0]) as f:
        f.write(processed)
