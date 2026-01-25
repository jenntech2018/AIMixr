from .models import Vote
from pipeline.analyze_track import analyze_track
from pipeline.score_track import score_track

def get_track_score(track_obj):
    """
    Helper to analyze and score a track object.
    Assumes track_obj has a 'file' attribute with a path.
    """
    try:
        # Adjust 'file.path' if your Track model uses a different field name for the audio file
        if hasattr(track_obj, 'file') and track_obj.file:
            analysis = analyze_track(track_obj.file.path)
            return score_track(analysis)
    except Exception as e:
        print(f"Error scoring track {track_obj.id}: {e}")
    return 0

def cast_ai_vote(battle):
    """
    Analyzes tracks in a battle and casts an AI vote based on the pipeline score.
    """
    if not battle.challenger_track or not battle.opponent_track:
        return

    score_challenger = get_track_score(battle.challenger_track)
    score_opponent = get_track_score(battle.opponent_track)

    winner = None
    if score_challenger > score_opponent:
        winner = battle.challenger
    elif score_opponent > score_challenger:
        winner = battle.opponent
    
    if winner:
        Vote.objects.create(
            battle=battle,
            voter=None,  # AI vote has no user
            voted_for=winner,
            is_ai=True
        )
