from pipeline.analyze_track import analyze_track, score_track

# Path to a local audio file (WAV, MP3 supported by torchaudio)
audio_path = r"C:\Users\poped\src\tracks\like-it-is.mp3"

# Run analysis
analysis = analyze_track(audio_path)
print("Analysis result:", analysis)

# Compute score
score = score_track(analysis)

print("Track score:", score)

input("Press Enter to exit...")

