"""
src/query/time_parser.py — Natural language time range parsing.

Handles expressions like:
  "last hour", "last 30 minutes", "today", "yesterday",
  "this morning", "between 10:00 and 12:00", etc.

For recorded footage:
  "now" = latest indexed timestamp (not wall clock)
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

# For recorded footage, we use relative video seconds.
# Returns (start_seconds, end_seconds) relative to video start.
# None means "all time" (no filter).


def parse_time_range(text: str,
                     max_video_seconds: float = float("inf")) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse a time range expression from natural language.

    Returns:
        (start_seconds, end_seconds) from video start.
        Both None means "no time filter" (search all).
    """
    text = text.lower().strip()

    if not text or text in ("all", "all time", "any", "anytime", ""):
        return None, None

    # "now" or "latest" → last 5 minutes of footage
    if text in ("now", "latest", "recent"):
        start = max(0.0, max_video_seconds - 300)
        return start, max_video_seconds

    # "last X seconds/minutes/hours"
    m = re.search(r"last\s+(\d+(?:\.\d+)?)\s*(second|seconds|sec|secs|s\b)", text)
    if m:
        delta = float(m.group(1))
        start = max(0.0, max_video_seconds - delta)
        return start, max_video_seconds

    m = re.search(r"last\s+(\d+(?:\.\d+)?)\s*(minute|minutes|min|mins)", text)
    if m:
        delta = float(m.group(1)) * 60
        start = max(0.0, max_video_seconds - delta)
        return start, max_video_seconds

    m = re.search(r"last\s+(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs)", text)
    if m:
        delta = float(m.group(1)) * 3600
        start = max(0.0, max_video_seconds - delta)
        return start, max_video_seconds

    # "last minute"
    if re.search(r"last\s+minute\b", text):
        start = max(0.0, max_video_seconds - 60)
        return start, max_video_seconds

    # "last hour"
    if re.search(r"last\s+hour\b", text):
        start = max(0.0, max_video_seconds - 3600)
        return start, max_video_seconds

    # "first X minutes/seconds/hours"
    m = re.search(r"first\s+(\d+(?:\.\d+)?)\s*(second|seconds|sec|secs|s\b)", text)
    if m:
        return 0.0, float(m.group(1))

    m = re.search(r"first\s+(\d+(?:\.\d+)?)\s*(minute|minutes|min|mins)", text)
    if m:
        return 0.0, float(m.group(1)) * 60

    m = re.search(r"first\s+(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs)", text)
    if m:
        return 0.0, float(m.group(1)) * 3600

    # "between HH:MM and HH:MM" (treated as video timestamps)
    m = re.search(
        r"between\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s+and\s+(\d{1,2}):(\d{2})(?::(\d{2}))?",
        text
    )
    if m:
        h1, m1 = int(m.group(1)), int(m.group(2))
        s1 = int(m.group(3) or 0)
        h2, m2 = int(m.group(4)), int(m.group(5))
        s2 = int(m.group(6) or 0)
        start = h1 * 3600 + m1 * 60 + s1
        end = h2 * 3600 + m2 * 60 + s2
        return float(start), float(end)

    # "at HH:MM" → ±5 minutes window
    m = re.search(r"\bat\s+(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        s = int(m.group(3) or 0)
        t = h * 3600 + mn * 60 + s
        return max(0.0, float(t) - 300), float(t) + 300

    # "around HH:MM" → ±10 minutes window
    m = re.search(r"\baround\s+(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        s = int(m.group(3) or 0)
        t = h * 3600 + mn * 60 + s
        return max(0.0, float(t) - 600), float(t) + 600

    # "today" / "this morning" / "this afternoon" / "this evening" / "yesterday"
    # For recorded footage these are relative to video length
    if "today" in text or "this morning" in text or "this afternoon" in text or "this evening" in text:
        return None, None  # no filter — search all footage

    if "yesterday" in text:
        return None, None  # no filter — search all footage

    return None, None


def format_timestamp(seconds: float) -> str:
    """Format video seconds as HH:MM:SS."""
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def get_max_video_seconds() -> float:
    """Get the maximum timestamp from indexed footage."""
    try:
        from src import database as db
        cameras = db.get_all_cameras()
        if not cameras:
            return float("inf")
        return max(c["duration"] for c in cameras if c.get("indexed") and c.get("duration"))
    except Exception:
        return float("inf")
