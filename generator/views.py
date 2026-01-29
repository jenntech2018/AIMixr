# generator/views.py
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout
from django.contrib.auth.models import User
from django.views.decorators.http import require_POST
from django.http import JsonResponse, FileResponse, Http404
from django.core.files import File

from .models import Track
from accounts.models import UserProfile
from generator.utils import user_reached_free_limit
from worker.tasks import analyze_track_task
from pipeline.master import master_track_audio
from pipeline.analyze_track import extract_waveform_only 

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
    response = FileResponse(open(file_path, 'rb'), as_attachment=True)
    filename = os.path.basename(file_path)
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response

# ---------------------------------------------------------
# DASHBOARD + UPLOAD (FIXED: BLOCK & INCREMENT)
# ---------------------------------------------------------

@login_required
def dashboard_view(request):
    if request.method == "POST":
        # 1. BLOCK: Check limit before processing the file
        if user_reached_free_limit(request.user):
            messages.error(request, "Upload limit reached. Please upgrade your plan to add more tracks.")
            return redirect("dashboard")

        audio_file = request.FILES.get("audio_file")
        if not audio_file:
            messages.error(request, "Please upload an audio file.")
            return redirect("dashboard")

        if not audio_file.content_type.startswith("audio/"):
            messages.error(request, "Invalid file format. Please upload an audio file.")
            return redirect("dashboard")

        # 2. CREATE TRACK
        track = Track.objects.create(
            user=request.user,
            audio_file=audio_file,
            source_type="upload",
            status="processing",
        )

        try:
            # 3. INCREMENT: User is "charged" as soon as the upload is successful
            profile, _ = UserProfile.objects.get_or_create(user=request.user)
            profile.usage_count += 1
            profile.save()

            # Trigger background analysis (extracts original waveform)
            analyze_track_task.delay(track.id)

            return redirect("track_detail", track_id=track.id)
        except Exception as e:
            print(f"Upload Error: {e}")
            messages.error(request, "An error occurred during upload processing.")

    tracks = Track.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "dashboard.html", {"tracks": tracks})

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

@login_required
def track_detail(request, track_id):
    track = get_object_or_404(Track, id=track_id)
    analysis = getattr(track, "analysis_obj", None)

    context = {
        "track": track,
        "analysis": analysis,
        "score": getattr(analysis, "score", 0) if analysis else 0,
    }
    return render(request, "track_detail.html", context)

# ---------------------------------------------------------
# MASTER TRACK (FIXED: MASTERED WAVEFORM)
# ---------------------------------------------------------

@login_required
@require_POST
def master_track(request, track_id):
    track = get_object_or_404(Track, id=track_id, user=request.user)

    try:
        # 1. Process Audio
        mastered_path = master_track_audio(track.audio_file.path)

        # 2. Save Master File
        with open(mastered_path, "rb") as f:
            track.master_file.save(
                os.path.basename(mastered_path),
                File(f),
                save=True,
            )

        # 3. EXTRACT MASTERED WAVEFORM
        # This makes the 2nd waveform show up in your UI
        analysis = getattr(track, "analysis_obj", None)
        if analysis:
            analysis.master_waveform = extract_waveform_only(track.master_file.path)
            analysis.save()

        messages.success(request, "Mastering finished successfully!")

    except Exception as e:
        print(f"Mastering Error: {e}")
        messages.error(request, f"Mastering failed: {e}")

    return redirect("track_detail", track_id=track.id)

# ---------------------------------------------------------
# RANKINGS
# ---------------------------------------------------------

def rankings_page(request):
    tracks = Track.objects.all()
    ranked = sorted(tracks, key=lambda t: getattr(getattr(t, "analysis_obj", None), "score", 0), reverse=True)
    return render(request, "rankings.html", {"tracks": ranked})

def rankings_data(request):
    tracks = Track.objects.all()
    ranked = sorted(tracks, key=lambda t: getattr(getattr(t, "analysis_obj", None), "score", 0), reverse=True)
    return JsonResponse({
        "tracks": [
            {
                "id": t.id,
                "name": os.path.basename(t.audio_file.name) if t.audio_file else "Unknown",
                "score": getattr(getattr(t, "analysis_obj", None), "score", 0),
                "source": t.source_type,
                "created_at": t.created_at.isoformat(),
            } for t in ranked
        ]
    })

# ---------------------------------------------------------
# LEGAL
# ---------------------------------------------------------

def privacy_policy(request):
    return render(request, "legal/privacy.html")

def terms_of_service(request):
    return render(request, "legal/terms.html")
