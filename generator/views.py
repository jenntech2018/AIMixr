# generator/views.py

import os
import logging

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout
from django.contrib.auth.models import User
from django.views.decorators.http import require_POST
from django.http import JsonResponse, FileResponse, Http404
from django.core.files import File
from django.db.models import Q

from .models import Track, Stem, TrackAnalysis
from accounts.models import UserProfile
from generator.utils import user_reached_free_limit
from pipeline.master import master_track_audio
from pipeline.analyze_track import extract_waveform_only


# -------------------------------------------------------------------------------
# Logger
# -------------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# AUTH
# ---------------------------------------------------------

def logout_view(request):
    logout(request)
    return render(request, "logout.html")


def register_view(request):

    if request.method == "POST":

        username = request.POST.get("username")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        if not password1:
            messages.error(request, "Password is required.")
            return redirect("register")

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

        user.backend = "django.contrib.auth.backends.ModelBackend"

        UserProfile.objects.get_or_create(user=user)

        login(request, user)

        return redirect("dashboard")

    return render(request, "register.html")


# ----------------------------------------------------------
# DOWNLOAD
# ----------------------------------------------------------

def download_mastered_track(request, track_id):

    track = get_object_or_404(Track, id=track_id)

    if not track.master_file:
        raise Http404("Mastered file not found.")

    file_path = track.master_file.path

    response = FileResponse(open(file_path, "rb"), as_attachment=True)

    filename = os.path.basename(file_path)

    response["Content-Disposition"] = f'attachment; filename="{filename}"'

    return response


# ---------------------------------------------------------
# DASHBOARD + UPLOAD
# ---------------------------------------------------------

@login_required
def dashboard_view(request):

    user = request.user

    profile, _ = UserProfile.objects.get_or_create(user=user)

    from social.models import Battle # Ensure this import is correct for your structure
    pending_challenges = Battle.objects.filter(opponent=user, status='pending')
    free_limit_reached = user_reached_free_limit(user)

    # =============================
    # HANDLE UPLOAD
    # =============================

    if request.method == "POST":

        if free_limit_reached:
            messages.error(
                request,
                "Upload limit reached. Please upgrade your plan."
            )
            return redirect("dashboard")

        audio_file = request.FILES.get("audio_file")

        if not audio_file:
            messages.error(request, "Please upload an audio file.")
            return redirect("dashboard")

        if not audio_file.content_type.startswith("audio/"):
            messages.error(request, "Invalid audio file.")
            return redirect("dashboard")

        visibility = request.POST.get("visibility", "public")

        is_private = (visibility == "private")

        try:

            from django.core.files.base import ContentFile

            # Keep original filename
            original_name = os.path.basename(audio_file.name)

            # Create track (no file yet)
            track = Track(
                user=user,
                source_type="upload",
                status="processing",
                private=is_private,
            )

            # Save uploaded file
            track.audio_file.save(
                original_name,
                ContentFile(audio_file.read()),
                save=True,
            )

            # Update usage
            profile.usage_count += 1
            profile.save()

            # Run Celery task
            from worker.tasks import analyze_track_task

            analyze_track_task.delay(track.id) 
            from worker.tasks import split_stems_task

            split_stems_task.delay(track.id)
            return redirect("track_detail", track_id=track.id)

        except Exception as e:

            logger.exception(f"Upload failed for {user.username}")

            messages.error(request, f"Upload failed: {e}")

            return redirect("dashboard")

    # =============================
    # DASHBOARD VIEW (GET)
    # =============================

    tracks = Track.objects.filter(
        Q(private=False) | Q(user=user)
    ).order_by("-created_at")

    return render(
        request,
        "dashboard.html",
        {
            "tracks": tracks,
            "profile": profile,
            "pending_challenges": pending_challenges,
            "free_limit_reached": free_limit_reached,
        },
    )


# ---------------------------------------------------------
# TRACK STATUS
# ---------------------------------------------------------

@login_required
def track_status(request, track_id):

    track = get_object_or_404(Track, id=track_id)

    return JsonResponse({
        "status": track.status,
        "has_analysis": hasattr(track, "analysis_obj"),
    })


# ---------------------------------------------------------
# TRACK DETAIL
# ---------------------------------------------------------

def track_detail(request, track_id):

    track = get_object_or_404(Track, id=track_id)

    analysis_obj = getattr(track, "analysis_obj", None)

    # Get stems for this track
    stems = Stem.objects.filter(track=track)

    context = {
        "track": track,
        "analysis_obj": analysis_obj or {},
        "master_waveform": getattr(analysis_obj, "master_waveform", []),
        "ai_feedback": getattr(analysis_obj, "ai_feedback", ""),
        "rms": getattr(analysis_obj, "rms", "-14.2"),
        "lyrics_raw": getattr(analysis_obj, "lyrics_raw", ""),
        "score": getattr(analysis_obj, "score", 0),

        # NEW
        "stems": stems,
    }

    return render(request, "track_detail.html", context)

# ---------------------------------------------------------
# MASTER TRACK
# ---------------------------------------------------------

@login_required
@require_POST
def master_track(request, track_id):

    track = get_object_or_404(Track, id=track_id, user=request.user)

    try:

        mastered_path = master_track_audio(track.audio_file.path)

        with open(mastered_path, "rb") as f:

            track.master_file.save(
                os.path.basename(mastered_path),
                File(f),
                save=True,
            )

        analysis = getattr(track, "analysis_obj", None)

        if analysis:
            analysis.master_waveform = extract_waveform_only(
                track.master_file.path
            )
            analysis.save()

        messages.success(request, "Mastering finished!")

    except Exception as e:

        logger.exception(f"Mastering failed: {track.id}")

        messages.error(request, f"Mastering failed: {e}")

    return redirect("track_detail", track_id=track.id)




# generator/views.py

@login_required
def split_stems(request, track_id):
    # 1. Get the track
    track = get_object_or_404(Track, id=track_id, user=request.user)
    
    # 2. Check Plan
    if request.user.userprofile.plan_name != "Studio Pro":
        messages.error(request, "Studio Pro plan required for stems.")
        return redirect('track_detail', track_id=track.id)

    # 3. Update Status
    track.status = "splitting"
    track.save()

    # 4. Trigger the REAL Celery task (located in worker/tasks.py)
    from worker.tasks import split_stems_task
    split_stems_task.delay(track.id)
    
    messages.success(request, "Stem separation started!")
    return redirect('track_detail', track_id=track.id)
# ---------------------------------------------------------
# RANKINGS
# ---------------------------------------------------------

def rankings_page(request):

    tracks = Track.objects.all()

    ranked = sorted(
        tracks,
        key=lambda t: getattr(
            getattr(t, "analysis_obj", None),
            "score",
            0,
        ),
        reverse=True,
    )

    return render(request, "rankings.html", {"tracks": ranked})


def rankings_data(request):

    tracks = Track.objects.all()

    ranked = sorted(
        tracks,
        key=lambda t: getattr(
            getattr(t, "analysis_obj", None),
            "score",
            0,
        ),
        reverse=True,
    )

    return JsonResponse({
        "tracks": [
            {
                "index": t.index,
                "id": t.id,
                "name": os.path.basename(t.audio_file.name)
                if t.audio_file else "Unknown",
                "score": getattr(
                    getattr(t, "analysis_obj", None),
                    "score",
                    0,
                ),
                "source": t.source_type,
                "created_at": t.created_at.isoformat(),
            }
            for t in ranked
        ]
    })


# ---------------------------------------------------------
# LEGAL
# ---------------------------------------------------------

def privacy_policy(request):
    return render(request, "legal/privacy.html")


def terms_of_service(request):
    return render(request, "legal/terms.html")
