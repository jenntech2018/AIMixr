# pipeline/lyrics/clean_and_analyze.py
"""
Simple lyrics cleaning + lightweight analysis.

Provides:
- lyrics_clean: cleaned, human-readable lyric text
- lyrics_feedback: short human-readable notes for the UI
- lyrics_quality: 0.0-1.0 overall heuristic score (higher = better)
- profanity_score: 0.0-1.0 (higher = more profanity)
- rhyme_density: 0.0-1.0 (naive heuristic)
- word_count, line_count
"""

import re
import math

# small built-in list of explicit words for a profanity estimate (keep short; extend as needed)
_PROFANITY_SET = {
    "fuck", "shit", "damn", "bitch", "asshole", "bastard", "crap", "piss", "douche"
}

def _strip_timestamps(text: str) -> str:
    # remove common timestamp formats like [00:12], 00:12, 0:12.000
    text = re.sub(r"\[?\b\d{1,2}:\d{2}(?:\.\d+)?\]?", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?:\.\d+)?\b", " ", text)
    return text

def _strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _clean_line(line: str) -> str:
    # remove things like (chorus), <chorus>, {repeat}, and leading punctuation
    line = re.sub(r"\(.*?\)|\<.*?\>|\{.*?\}", " ", line)
    # allow contractions but remove stray non-word characters
    line = re.sub(r"[^\w'\s\-.,?!]", " ", line)
    line = line.strip()
    return line

def _compute_profanity(words):
    if not words:
        return 0.0
    matches = sum(1 for w in words if w.lower().strip(".,?!'\"") in _PROFANITY_SET)
    return min(1.0, matches / max(1, len(words) / 10))  # scaled heuristically

def _naive_rhyme_density(lines):
    """
    Naive rhyme density: compare last 2-3 letters of line-ending words.
    Returns fraction 0.0-1.0 of line pairs that rhyme (adjacent pairs).
    """
    endings = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        words = ln.split()
        last = words[-1].lower().strip(".,?!'\"")
        if len(last) >= 3:
            endings.append(last[-3:])
        elif last:
            endings.append(last)
    if len(endings) < 2:
        return 0.0
    matches = 0
    pairs = 0
    for i in range(len(endings)-1):
        a = endings[i]
        b = endings[i+1]
        pairs += 1
        # count as rhyme if suffixes equal or Levenshtein-ish small difference
        if a == b:
            matches += 1
        else:
            # allow one-char difference for short endings
            if len(a) <= 2 and len(b) <= 2 and a[0] == b[0]:
                matches += 1
    return matches / pairs if pairs > 0 else 0.0

def clean_and_analyze(raw_lyrics: str, prefer_openai: bool = False) -> dict:
    """
    Clean raw transcription and produce simple analysis.
    prefer_openai is accepted for API-aware pipelines (unused here).
    """
    result = {
        "lyrics_raw": raw_lyrics or "",
        "lyrics_clean": "",
        "lyrics_feedback": "",
        "lyrics_quality": 0.0,
        "profanity_score": 0.0,
        "rhyme_density": 0.0,
        "word_count": 0,
        "line_count": 0,
    }

    if not raw_lyrics or not raw_lyrics.strip():
        result["lyrics_clean"] = ""
        result["lyrics_feedback"] = "No vocal content detected."
        return result

    text = raw_lyrics

    # 1) remove timestamps & urls
    text = _strip_timestamps(text)
    text = _strip_urls(text)

    # 2) split into lines and clean lines
    lines = re.split(r"[\\n\r]+|(?<=\.)\s+", text)
    cleaned_lines = []
    words_total = []
    for ln in lines:
        cl = _clean_line(ln)
        if cl:
            cleaned_lines.append(cl)
            words_total.extend(cl.split())

    cleaned = "\n".join(cleaned_lines)
    cleaned = _normalize_whitespace(cleaned)

    result["lyrics_clean"] = cleaned
    result["word_count"] = len(words_total)
    result["line_count"] = len(cleaned_lines)

    # 3) profanity
    result["profanity_score"] = _compute_profanity(words_total)

    # 4) rhyme density (naive)
    result["rhyme_density"] = _naive_rhyme_density(cleaned_lines)

    # 5) heuristics for 'clarity' and 'quality'
    # - clarity: penalize lots of single-letter tokens or many numeric tokens
    if words_total:
        short_tokens = sum(1 for w in words_total if len(w) == 1)
        numeric_tokens = sum(1 for w in words_total if re.fullmatch(r"\d+", w))
        clarity = max(0.0, 1.0 - (short_tokens / max(1, len(words_total)))*2.0 - (numeric_tokens / max(1, len(words_total))))
        clarity = min(1.0, clarity)
    else:
        clarity = 0.0

    # 6) base quality: combine clarity and rhyme (weighted)
    quality = 0.5 * clarity + 0.4 * result["rhyme_density"] - 0.3 * result["profanity_score"]
    # normalize to 0..1
    quality = max(0.0, min(1.0, quality))

    result["lyrics_quality"] = quality

    # 7) short human feedback strings
    feedback_lines = []
    feedback_lines.append(f"Words: {result['word_count']}, Lines: {result['line_count']}.")
    feedback_lines.append(f"Rhyme density: {result['rhyme_density']:.2f} (naive).")
    feedback_lines.append(f"Profanity score: {result['profanity_score']:.2f}.")
    feedback_lines.append(f"Clarity: {clarity:.2f}, Overall lyric quality (0-1): {quality:.2f}.")

    # actionable suggestions
    suggestions = []
    if clarity < 0.6:
        suggestions.append("Consider rewording unclear lines and removing disfluencies.")
    if result["rhyme_density"] < 0.2:
        suggestions.append("Add more internal or end rhymes to increase musicality.")
    if result["profanity_score"] > 0.2:
        suggestions.append("Profanity is present; consider sanitizing if needed for audiences.")
    if result["word_count"] < 20:
        suggestions.append("Lyrics are short — consider adding more verses or hooks for structure.")

    if suggestions:
        feedback_lines.append("Suggestions: " + " ".join(suggestions))
    else:
        feedback_lines.append("Nice — no obvious quick fixes from the basic analysis.")

    result["lyrics_feedback"] = " ".join(feedback_lines)

    return result
