#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(pwd)"

echo "========================================"
echo "🔎 AIMIXR FULL PROJECT SCAN"
echo "========================================"

echo ""
echo "📁 Project root: $PROJECT_ROOT"
echo ""

# -------------------------------
# 1. Python syntax check
# -------------------------------
echo "🐍 Checking Python syntax..."
python -m compileall . || echo "❌ Python syntax errors found"

# -------------------------------
# 2. Django config check
# -------------------------------
echo ""
echo "🧠 Django system check..."
python manage.py check || echo "❌ Django check failed"

# -------------------------------
# 3. Import validation
# -------------------------------
echo ""
echo "📦 Import sanity check..."
python - <<'EOF'
import pkgutil, sys, traceback

failed = False
for mod in [
    "generator.models",
    "generator.views",
    "pipeline.analyze_track",
    "worker.tasks",
]:
    try:
        __import__(mod)
        print(f"✅ Imported {mod}")
    except Exception as e:
        failed = True
        print(f"❌ FAILED import {mod}")
        traceback.print_exc()

if failed:
    sys.exit(1)
EOF

# -------------------------------
# 4. Model integrity
# -------------------------------
echo ""
echo "🧬 Model integrity check..."
python - <<'EOF'
from generator.models import Track
from django.core.exceptions import ImproperlyConfigured

t = Track.objects.first()
print("Track object:", t)

# verify analysis relationship
if not hasattr(t, "analysis_obj"):
    raise Exception("Track missing analysis_obj relationship")

print("✅ Track.analysis_obj exists")
EOF

# -------------------------------
# 5. Pending migrations
# -------------------------------
echo ""
echo "🗄️ Migration status..."
python manage.py makemigrations --check --dry-run || echo "⚠️ Unapplied model changes detected"

# -------------------------------
# 6. Celery task imports
# -------------------------------
echo ""
echo "⚙️ Celery task scan..."
python - <<'EOF'
from worker import tasks
print("✅ worker.tasks imported")

try:
    from ai_music.tracks import tasks as tracks_tasks
    print("⚠️ Found SECOND tasks.py in ai_music.tracks (potential conflict)")
except ImportError:
    print("✅ No conflicting ai_music.tracks.tasks")
EOF

# -------------------------------
# 7. Dangerous patterns scan
# -------------------------------
echo ""
echo "🚨 Scanning for known crash patterns..."

grep -R "track.analysis =" . && echo "❌ Found invalid Track.analysis assignment"
grep -R "track.analysis\." . && echo "⚠️ Found Track.analysis attribute access"
grep -R "import whisper" . || echo "⚠️ whisper not referenced"
grep -R "librosa.load" . || echo "⚠️ librosa.load not referenced"

# -------------------------------
# 8. Redis / Celery sanity
# -------------------------------
echo ""
echo "🧯 Redis connectivity check..."
redis-cli ping || echo "❌ Redis not responding"

# -------------------------------
# 9. Gunicorn log hint
# -------------------------------
echo ""
echo "📜 Recent gunicorn errors:"
journalctl -u gunicorn --since "10 minutes ago" | tail -n 20

# -------------------------------
# DONE
# -------------------------------
echo ""
echo "========================================"
echo "✅ Scan complete"
echo "========================================"
