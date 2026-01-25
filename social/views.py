from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import timedelta
from django.db.models import Count
from .models import ChatMessage, Battle, Vote
from .utils import cast_ai_vote
from generator.models import Track

User = get_user_model()

@login_required
def community_chat(request):
    if request.method == 'POST':
        content = request.POST.get('content')
        if content:
            ChatMessage.objects.create(user=request.user, content=content)
            return redirect('community_chat')
    
    messages = ChatMessage.objects.all().order_by('-timestamp')[:50]
    
    # Calculate "Online" users (users who posted in last 10 mins)
    recent_time = timezone.now() - timedelta(minutes=10)
    online_count = ChatMessage.objects.filter(timestamp__gte=recent_time).values('user').distinct().count()
    online_count = max(online_count, 1)

    return render(request, 'social/chat.html', {'messages': messages, 'online_count': online_count})

@login_required
def battles_page(request):
    active_battles = Battle.objects.filter(status='active').order_by('-created_at')
    leaderboard = User.objects.annotate(wins=Count('battles_won')).order_by('-wins')[:10]
    return render(request, 'social/battles.html', {'battles': active_battles, 'leaderboard': leaderboard})

@login_required
def create_battle(request, opponent_id):
    opponent = get_object_or_404(User, id=opponent_id)
    if request.method == 'POST':
        track_id = request.POST.get('track_id')
        track = get_object_or_404(Track, id=track_id, user=request.user)
        
        Battle.objects.create(
            challenger=request.user,
            opponent=opponent,
            challenger_track=track,
            status='active'
        )
        return redirect('community_chat') # Or a battle list view
        
    my_tracks = Track.objects.filter(user=request.user)
    return render(request, 'social/create_battle.html', {'opponent': opponent, 'tracks': my_tracks})

@login_required
def vote_battle(request, battle_id, voted_for_id):
    battle = get_object_or_404(Battle, id=battle_id)
    voted_for = get_object_or_404(User, id=voted_for_id)
    
    if battle.status != 'active':
        return redirect('community_chat')

    # Check if user already voted
    if not Vote.objects.filter(battle=battle, voter=request.user).exists():
        Vote.objects.create(battle=battle, voter=request.user, voted_for=voted_for)
        
        # Trigger AI vote if not already done
        if not Vote.objects.filter(battle=battle, is_ai=True).exists():
            cast_ai_vote(battle)
            
    return redirect('community_chat')
