# generator/views.py
from datetime import timedelta
import os
from time import timezone
import traceback
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import login
from django.views.decorators.http import require_POST
from django.core.files import File

from .models import Track
from generator.utils import celery_available
from workers.tasks import analyze_track_task

# pipeline
from pipeline.master import master_track_audio
from pipeline.analyze_track import analyze_track, score_track
from pipeline.lyrics.transcribe import transcribe_audio_file
from pipeline.lyrics.clean_and_analyze import clean_and_analyze
from pipeline.lyrics.evaluate_lyrics import evaluate_lyrics
from generator.utils import user_reached_free_limit


# ---------------------------------------------------------
# USER REGISTRATION
# ---------------------------------------------------------

def register_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        if password1 != password2:
            messages.error(request, "Passwords do not match.")
            return redirect("register")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken.")
            return redirect("register")

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password1,
        )
        login(request, user)
        return redirect("dashboard")

    return render(request, "register.html")

# ---------------------------------------------------------
# MASTER TRACK
# ---------------------------------------------------------

@login_required
@require_POST
def master_track(request, track_id):
    if user_reached_free_limit(request.user) and track.status == "done":
        messages.error(request, "Free tier allows only 1 mastered track.")
        return redirect("track_detail", track_id=track.id)

    track = get_object_or_404(Track, id=track_id, user=request.user)

    if not track.audio_file:
        messages.error(request, "No audio file found.")
        return redirect("track_detail", track_id=track_id)

    try:
        mastered_path = master_track_audio(track.audio_file.path)

        with open(mastered_path, "rb") as f:
            track.master_file.save(
                os.path.basename(mastered_path),
                File(f),
                save=True,
            )

        messages.success(request, "Mastered version created!")

    except Exception as e:
        messages.error(request, f"Mastering error: {e}")

    return redirect("track_detail", track_id=track_id)

# ---------------------------------------------------------
# DASHBOARD + UPLOAD
# ---------------------------------------------------------

from generator.utils import user_reached_free_limit

@login_required
def dashboard_view(request):
    if request.method == "POST":

        # ✨ LIMIT CHECK ✨
        if user_reached_free_limit(request.user):
            messages.error(
                request,
                "Free accounts can upload only 1 track. Upgrade to upload more."
            )
            return redirect("dashboard")

        audio_file = request.FILES.get("audio_file")
        if not audio_file:
            messages.error(request, "Please upload an audio file.")
            return redirect("dashboard")

        # (your existing track create + processing code)

        track = Track.objects.create(
            user=request.user,
            audio_file=audio_file,
            source_type="upload",
            status="pending",
        )
        profile = request.user.userprofile
        profile.usage_count += 1
        profile.save()

        # -------- Celery path --------
        if celery_available():
            track.status = "queued"
            track.save()
            analyze_track_task.delay(track.id)

        # -------- Non-celery fallback --------
        else:
            track.status = "processing"
            track.save()

            try:
                analysis = analyze_track(track.audio_file.path)

                # TRANSCRIPTION
                try:
                    raw_lyrics = transcribe_audio_file(track.audio_file.path, model_name="small")
                except Exception as e:
                    raw_lyrics = ""

                # CLEAN + FEEDBACK
                lyrics_info = clean_and_analyze(raw_lyrics, prefer_openai=True)

                # EXTENDED METRICS
                extended = evaluate_lyrics(lyrics_info.get("lyrics_clean", ""))

                # Store fields safely
                analysis = analysis or {}

                analysis["lyrics_raw"] = raw_lyrics
                analysis["lyrics_clean"] = lyrics_info.get("lyrics_clean", "")
                analysis["lyrics_feedback"] = lyrics_info.get("lyrics_feedback", "")
                analysis["lyrics_quality"] = lyrics_info.get("lyrics_quality", 0.0)

                analysis["lyrics_ai_review"] = extended.get("review", "")
                analysis["lyrics_metrics"] = extended.get("metrics", {})

                track.analysis = analysis
                track.status = "done"

            except Exception as e:
                import traceback
                traceback.print_exc()
                track.analysis = {"error": str(e)}
                track.status = "error"
            track.save()

        return redirect("track_detail", track_id=track.id)

    tracks = Track.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "dashboard.html", {"tracks": tracks})

def activate_subscription(user):
    profile = user.userprofile
    profile.subscription_active = True
    profile.subscription_expires = timezone.now() + timedelta(days=30)
    profile.uploads_used = 0   # optional: give them full access/reset
    profile.save()

# ---------------------------------------------------------
# TRACK DETAIL
# ---------------------------------------------------------

@login_required
def track_detail(request, track_id):
    track = get_object_or_404(Track, id=track_id)
    analysis = track.analysis or {}

    score = score_track(analysis) if analysis else 0
    ai_feedback = analysis.get("feedback", "Analysis pending...")

    lyrics_metrics = analysis.get("lyrics_metrics", {})
    print("ANALYSIS KEYS:", analysis.keys())

    def pct(val):
        try:
            return float(val) * 100
        except:
            return 0
    profile = request.user.userprofile


    ctx = {
        "is_premium": profile.is_premium,
        "free_limit_reached": user_reached_free_limit(request.user),
        "track": track,
        "score": score,
        "ai_feedback": ai_feedback,

        "waveform": analysis.get("waveform", []),
        "rms": analysis.get("rms_over_time", []),
        "spectral_centroid": analysis.get("centroid_over_time", []),

        "lyrics_raw": analysis.get("lyrics_raw", ""),
        "lyrics_clean": analysis.get("lyrics_clean", ""),
        "lyrics_feedback": analysis.get("lyrics_feedback", ""),
        "lyrics_ai_review": analysis.get("lyrics_ai_review", ""),
        "lyrics_quality": analysis.get("lyrics_quality", 0.0),

        "rhyme_density_pct": pct(lyrics_metrics.get("rhyme_density", 0)),
        "clarity_pct": pct(lyrics_metrics.get("clarity", 0)),
        "cohesion_pct": pct(lyrics_metrics.get("cohesion", 0)),
        "storytelling_pct": pct(lyrics_metrics.get("storytelling", 0)),
        "originality_pct": pct(lyrics_metrics.get("originality", 0)),
    }

    return render(request, "track_detail.html", ctx)

# ---------------------------------------------------------
# RANKINGS
# ---------------------------------------------------------

def rankings_page(request):
    tracks = Track.objects.filter(status="done")
    ranked = sorted(tracks, key=lambda t: score_track(t.analysis or {}), reverse=True)
    return render(request, "rankings.html", {"tracks": ranked})

def rankings_data(request):
    tracks = Track.objects.all()
    ranked = sorted(tracks, key=lambda t: score_track(t.analysis or {}), reverse=True)

    data = [
        {
            "id": t.id,
            "name": t.audio_file.name if t.audio_file else "",
            "score": score_track(t.analysis or {}),
            "source": t.source_type,
            "created_at": t.created_at.isoformat(),
        }
        for t in ranked
    ]

    return JsonResponse({"tracks": data})
