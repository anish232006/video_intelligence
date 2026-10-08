"""
src/ingestion/metadata.py — Extract and store video metadata.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src import database as db
from src.ingestion.video_reader import VideoInfo

logger = logging.getLogger(__name__)


def register_video(
    camera_id: str,
    name: str,
    video_path: str,
    description: Optional[str] = None,
) -> Optional[VideoInfo]:
    """
    Read video metadata and register the camera in the database.

    Returns VideoInfo if valid, None if the video is unreadable.
    """
    info = VideoInfo(video_path)
    if not info.valid:
        logger.error("Cannot register invalid video: %s", video_path)
        return None

    db.upsert_camera(
        camera_id=camera_id,
        name=name,
        video_path=video_path,
        description=description,
        duration=info.duration,
        fps=info.fps,
        width=info.width,
        height=info.height,
        frame_count=info.frame_count,
    )
    logger.info(
        "Registered camera %s: %s  (%.1fs @ %.1f fps, %dx%d)",
        camera_id, info.path.name, info.duration, info.fps, info.width, info.height,
    )
    return info
