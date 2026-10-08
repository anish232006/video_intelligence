"""
src/evidence/clip_generator.py — Lazy evidence clip generation using FFmpeg.

Design:
- Clips are NOT generated during indexing
- Generated on demand when user requests evidence
- Cached to disk after first generation
- Gracefully falls back to thumbnail if FFmpeg is unavailable
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from src.config import cfg

logger = logging.getLogger(__name__)

# Check FFmpeg availability once at module load
_FFMPEG_AVAILABLE: Optional[bool] = None


def _check_ffmpeg() -> bool:
    global _FFMPEG_AVAILABLE
    if _FFMPEG_AVAILABLE is None:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=5,
            )
            _FFMPEG_AVAILABLE = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            _FFMPEG_AVAILABLE = False
        if _FFMPEG_AVAILABLE:
            logger.info("FFmpeg detected — evidence clips enabled")
        else:
            logger.warning("FFmpeg not found — evidence clips disabled, thumbnails only")
    return _FFMPEG_AVAILABLE


def generate_evidence_clip(
    video_path: str,
    timestamp: float,
    track_db_id: int,
    camera_id: str,
    before_secs: float = None,
    after_secs: float = None,
) -> Optional[str]:
    """
    Generate a short evidence clip around the given timestamp.

    Clips are cached: same inputs → same output file (won't re-generate).

    Args:
        video_path: Path to the original camera video.
        timestamp: The event timestamp (seconds from video start).
        track_db_id: Database ID for naming the clip.
        camera_id: Camera identifier.
        before_secs: Seconds to include before timestamp.
        after_secs: Seconds to include after timestamp.

    Returns:
        Path to generated clip, or None if generation fails.
    """
    before = before_secs or cfg.evidence_clip_before
    after = after_secs or cfg.evidence_clip_after

    # Build a deterministic filename
    key = f"{camera_id}_{track_db_id}_{timestamp:.1f}"
    clip_id = hashlib.md5(key.encode()).hexdigest()[:12]
    clip_path = cfg.clips_dir / f"{clip_id}.mp4"

    # Return cached clip if it exists
    if clip_path.exists() and clip_path.stat().st_size > 0:
        logger.debug("Using cached clip: %s", clip_path)
        return str(clip_path)

    # Compute time window
    start = max(0.0, timestamp - before)
    duration = before + after

    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        logger.error("Video not found for clip generation: %s", video_path)
        return None

    cfg.clips_dir.mkdir(parents=True, exist_ok=True)

    if not _check_ffmpeg():
        # Fallback to OpenCV VideoWriter
        return _generate_clip_opencv(video_path_obj, start, duration, clip_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", str(video_path_obj),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(clip_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
        )
        if result.returncode == 0 and clip_path.exists() and clip_path.stat().st_size > 0:
            logger.info("Evidence clip generated: %s (%.1f-%.1fs)",
                        clip_path.name, start, start + duration)
            return str(clip_path)
        else:
            stderr = result.stderr.decode("utf-8", errors="ignore")[:500]
            logger.warning("FFmpeg failed (code %d): %s — trying OpenCV fallback", result.returncode, stderr)
            return _generate_clip_opencv(video_path_obj, start, duration, clip_path)
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timed out generating clip")
        if clip_path.exists():
            clip_path.unlink()
        return _generate_clip_opencv(video_path_obj, start, duration, clip_path)
    except Exception as e:
        logger.warning("FFmpeg error: %s — trying OpenCV fallback", e)
        return _generate_clip_opencv(video_path_obj, start, duration, clip_path)


def _generate_clip_opencv(video_path: Path, start: float, duration: float, output_path: Path) -> Optional[str]:
    """Fallback clip generator using OpenCV when FFmpeg is not available."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_frame = max(0, int(start * fps))
        end_frame = int((start + duration) * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            return None

        cur = start_frame
        while cur <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            cur += 1

        cap.release()
        out.release()
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info("Evidence clip generated via OpenCV: %s", output_path.name)
            return str(output_path)
    except Exception as e:
        logger.error("OpenCV clip generation failed: %s", e)
    return None


def ffmpeg_available() -> bool:
    """Check if FFmpeg is available on this system."""
    return _check_ffmpeg()
