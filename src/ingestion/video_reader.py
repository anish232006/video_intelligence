"""
src/ingestion/video_reader.py — Video reading and frame sampling.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from src.config import cfg

logger = logging.getLogger(__name__)


class VideoInfo:
    """Metadata about a video file."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.frame_count: int = 0
        self.duration: float = 0.0
        self.valid: bool = False
        self._read_metadata()

    def _read_metadata(self) -> None:
        if not self.path.exists():
            logger.error("Video not found: %s", self.path)
            return
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            logger.error("Cannot open video: %s", self.path)
            cap.release()
            return
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.valid = True
        cap.release()

    def __repr__(self) -> str:
        return (f"VideoInfo({self.path.name}, {self.width}x{self.height}, "
                f"{self.fps:.1f}fps, {self.duration:.1f}s)")


def sample_frames(
    video_path: str,
    sample_fps: float = 2.0,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    Yield (frame_number, timestamp_seconds, frame_bgr) at the requested FPS.

    Args:
        video_path: Path to video file.
        sample_fps: Target sampling rate (frames per second).
        start_sec: Start time in seconds.
        end_sec: End time in seconds (None = until end of video).

    Yields:
        Tuple of (original_frame_number, timestamp_seconds, bgr_frame)
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / native_fps

        # Clamp end_sec
        if end_sec is None:
            end_sec = total_duration
        end_sec = min(end_sec, total_duration)

        # Frame skip interval
        frame_interval = max(1, int(round(native_fps / sample_fps)))

        # Seek to start
        if start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        last_yielded_time = -1.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_num / native_fps

            if timestamp > end_sec:
                break

            if timestamp < start_sec:
                frame_num += 1
                continue

            # Only yield if enough time has passed
            if timestamp - last_yielded_time >= (1.0 / sample_fps) - 0.001:
                yield frame_num, timestamp, frame
                last_yielded_time = timestamp

            # Skip frames efficiently
            frame_num += frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    finally:
        cap.release()


def extract_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 padding: int = 0) -> Optional[np.ndarray]:
    """Extract a padded crop from a frame."""
    h, w = frame.shape[:2]
    x1p = max(0, x1 - padding)
    y1p = max(0, y1 - padding)
    x2p = min(w, x2 + padding)
    y2p = min(h, y2 + padding)

    if x2p <= x1p or y2p <= y1p:
        return None

    crop = frame[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return None
    return crop


def compute_sharpness(image: np.ndarray) -> float:
    """Estimate image sharpness using Laplacian variance."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def save_crop(crop: np.ndarray, save_path: Path) -> bool:
    """Save a crop image to disk."""
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), crop)
        return True
    except Exception as e:
        logger.error("Failed to save crop %s: %s", save_path, e)
        return False
