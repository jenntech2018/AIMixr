from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import timedelta
from django.db.models import Count
from .models import ChatMessage, Battle, Vote
from .utils import cast_ai_vote
from generator.models import Track
from django.contrib import messages
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


from django.db.models import Count, F  # Make sure F is imported here

@login_required
def battles_page(request):
    active_battles = Battle.objects.filter(status='active').order_by('-created_at')
    
    # We annotate the counts, then use F() to add them together
    leaderboard = User.objects.annotate(
        wins_count=Count('battles_won', distinct=True),
        initiated=Count('battles_initiated', distinct=True),
        received=Count('battles_received', distinct=True)
    ).annotate(
        total_battles=F('initiated') + F('received')
    ).order_by('-wins_count', '-total_battles')

    return render(request, 'social/battles.html', {
        'battles': active_battles, 
        'leaderboard': leaderboard
    })



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
            status='pending' 
        )
        messages.success(request, f"Challenge sent to {opponent.username}!")
        return redirect('dashboard') # Redirect to dashboard to see it's sent

    my_tracks = Track.objects.filter(user=request.user)
    
    # 2. Ensure context matches what the template expects
    return render(request, 'social/create_battle.html', {
        'opponent': opponent, 
        'tracks': my_tracks
    })
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


# social/views.py

@login_required
def accept_battle(request, battle_id):
    # Use 'opponent' instead of 'challenged_user'
    battle = get_object_or_404(Battle, id=battle_id, opponent=request.user)
    
    if request.method == "POST":
        track_id = request.POST.get("track_id")
        track = get_object_or_404(Track, id=track_id, user=request.user)
        
        # Use 'opponent_track' instead of 'challenged_track'
        battle.opponent_track = track
        battle.status = 'active'
        # Set the end time to 24 hours from now
        battle.ended_at = timezone.now() + timedelta(hours=24)
        battle.save()
        
        from django.contrib import messages
        messages.success(request, "Battle is LIVE! Voting starts now.")
        return redirect('battle_detail', battle_id=battle.id)

    my_tracks = Track.objects.filter(user=request.user)
    # Ensure context key 'user_tracks' matches what your template expects
    return render(request, "social/accept_battle.html", {
        "battle": battle, 
        "user_tracks": my_tracks
    })


from django.shortcuts import render, get_object_or_404
from .models import Battle
from django.utils import timezone

def battle_detail(request, battle_id):
    battle = get_object_or_404(Battle, id=battle_id)
    
    # Auto-close battle if time is up but status is still 'active'
    if battle.status == 'active' and battle.ended_at and timezone.now() >= battle.ended_at:
        battle.status = 'finished'
        # Optional: Add logic here to determine winner based on vote counts
        battle.save()

    return render(request, 'social/battle_detail.html', {
        'battle': battle,
        'now': timezone.now()
    })
