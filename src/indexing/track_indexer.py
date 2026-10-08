"""
src/indexing/track_indexer.py — Core indexing pipeline.

For each camera video:
1. Sample frames
2. Detect objects (YOLO)
3. Track objects (ByteTrack)
4. Select best crops per track
5. Extract color attributes
6. Generate CLIP embeddings
7. Store in FAISS + SQLite

CRITICAL design rules:
- Never index every frame — use sampled frames only
- Select only best 3-5 crops per track
- Tracker state MUST be reset between cameras
- Unload YOLO before loading CLIP if VRAM is tight
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src import database as db
from src.config import cfg
from src.detection.detector import Detection, YOLODetector
from src.embeddings.clip_encoder import CLIPEncoder
from src.embeddings.color import estimate_dominant_color
from src.indexing.faiss_store import get_store
from src.ingestion.video_reader import (
    VideoInfo,
    compute_sharpness,
    extract_crop,
    sample_frames,
    save_crop,
)
from src.tracking.tracker import ByteTrackWrapper, TrackState

logger = logging.getLogger(__name__)


@dataclass
class CropCandidate:
    """A candidate crop for a track."""
    frame_number: int
    timestamp: float
    bbox: Tuple[int, int, int, int]
    crop: np.ndarray
    confidence: float
    sharpness: float
    area: int
    score: float = 0.0

    def compute_score(self, frame_width: int, frame_height: int) -> None:
        """Score crop based on quality metrics."""
        # Boundary penalty: crops near edges score lower
        x1, y1, x2, y2 = self.bbox
        margin_x = min(x1, frame_width - x2) / frame_width
        margin_y = min(y1, frame_height - y2) / frame_height
        margin_score = (margin_x + margin_y) / 2

        # Area score: larger is better (up to a point)
        area_score = min(self.area / (frame_width * frame_height * 0.25), 1.0)

        # Sharpness score (normalized)
        sharp_score = min(self.sharpness / 1000.0, 1.0)

        self.score = (
            0.40 * self.confidence +
            0.30 * sharp_score +
            0.20 * area_score +
            0.10 * margin_score
        )


@dataclass
class TrackAccumulator:
    """Accumulates crops for a track during indexing."""
    track_id: int
    camera_id: str
    class_name: str
    candidates: List[CropCandidate] = field(default_factory=list)
    observations: List[Dict] = field(default_factory=list)
    first_seen: float = float("inf")
    last_seen: float = 0.0
    max_confidence: float = 0.0
    frame_count: int = 0

    def add_candidate(self, candidate: CropCandidate) -> None:
        self.candidates.append(candidate)
        self.first_seen = min(self.first_seen, candidate.timestamp)
        self.last_seen = max(self.last_seen, candidate.timestamp)
        self.max_confidence = max(self.max_confidence, candidate.confidence)
        self.frame_count += 1

    def select_best_crops(self, max_crops: int = 5) -> List[CropCandidate]:
        """Select the best crops by score, spread across time."""
        if not self.candidates:
            return []

        scored = sorted(self.candidates, key=lambda c: c.score, reverse=True)
        if len(scored) <= max_crops:
            return scored

        # Select top crops with temporal diversity
        selected = [scored[0]]
        min_time_gap = (self.last_seen - self.first_seen) / max_crops

        for cand in scored[1:]:
            if len(selected) >= max_crops:
                break
            if all(abs(cand.timestamp - s.timestamp) >= min_time_gap
                   for s in selected):
                selected.append(cand)

        # Fill remaining if needed
        remaining = [c for c in scored if c not in selected]
        while len(selected) < max_crops and remaining:
            selected.append(remaining.pop(0))

        return selected[:max_crops]


class VideoIndexer:
    """
    Full pipeline indexer for a single camera video.

    Usage:
        indexer = VideoIndexer(camera_id, video_path, progress_callback)
        indexer.run()
    """

    def __init__(self,
                 camera_id: str,
                 video_path: str,
                 progress_callback: Optional[Callable[[float, str], None]] = None,
                 sample_fps: float = None,
                 stop_event=None):
        self.camera_id = camera_id
        self.video_path = video_path
        self.progress_callback = progress_callback
        self.sample_fps = sample_fps or cfg.sample_fps
        self.stop_event = stop_event

        # Will be set during run
        self._detector: Optional[YOLODetector] = None
        self._encoder: Optional[CLIPEncoder] = None
        self._tracker = ByteTrackWrapper()

        # Accumulators indexed by track_id
        self._track_accumulators: Dict[int, TrackAccumulator] = {}

        # Stats
        self.frames_processed = 0
        self.tracks_found = 0
        self.embeddings_added = 0

    def _report_progress(self, pct: float, msg: str) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(pct, msg)
            except Exception:
                pass

    def _is_stopped(self) -> bool:
        if self.stop_event and self.stop_event.is_set():
            return True
        return False

    def run(self) -> Dict:
        """Execute the full indexing pipeline."""
        logger.info("=== Indexing camera %s: %s ===", self.camera_id, self.video_path)

        video_info = VideoInfo(self.video_path)
        if not video_info.valid:
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        # Reset tracker for this camera
        self._tracker.reset()

        try:
            # Phase 1: Detection + Tracking
            self._report_progress(0.05, "Loading YOLO detector...")
            self._detector = YOLODetector()
            self._detector.load()

            self._report_progress(0.10, "Sampling frames and detecting objects...")
            self._phase_detect_and_track(video_info)

            # Unload YOLO to free VRAM for CLIP
            self._detector.unload()
            self._detector = None
            self._report_progress(0.60, "YOLO unloaded. Loading CLIP encoder...")

            # Phase 2: Crop selection + Embedding
            self._encoder = CLIPEncoder()
            self._encoder.load()

            self._report_progress(0.65, "Extracting embeddings for best crops...")
            self._phase_embed_and_store()

            self._encoder.unload()
            self._encoder = None

            self._report_progress(0.95, "Saving indexes to disk...")
            get_store().save()

            # Mark camera as indexed
            db.mark_camera_indexed(self.camera_id)

            self._report_progress(1.0, "Indexing complete!")

            return {
                "camera_id": self.camera_id,
                "frames_processed": self.frames_processed,
                "tracks_found": self.tracks_found,
                "embeddings_added": self.embeddings_added,
                "success": True,
            }

        except Exception as e:
            logger.error("Indexing failed for %s: %s", self.camera_id, e, exc_info=True)
            # Clean up
            if self._detector and self._detector.is_loaded:
                self._detector.unload()
            if self._encoder and self._encoder.is_loaded:
                self._encoder.unload()
            raise

    def _phase_detect_and_track(self, video_info: VideoInfo) -> None:
        """Run YOLO + tracker over sampled frames, accumulate crops."""
        frame_gen = sample_frames(self.video_path, sample_fps=self.sample_fps)
        total_duration = video_info.duration

        for frame_num, timestamp, frame in frame_gen:
            if self._is_stopped():
                break

            # Detect
            detections = self._detector.detect(frame)

            # Track
            h, w = frame.shape[:2]
            assigned = self._tracker.update(detections, frame_num, timestamp, (h, w))

            # Accumulate crops
            for track_id, det in assigned:
                crop = extract_crop(frame, det.x1, det.y1, det.x2, det.y2,
                                    padding=cfg.crop_padding)
                if crop is None:
                    continue

                # Skip tiny crops
                if crop.shape[0] < cfg.min_crop_size or crop.shape[1] < cfg.min_crop_size:
                    continue

                sharpness = compute_sharpness(crop)
                area = (det.x2 - det.x1) * (det.y2 - det.y1)

                cand = CropCandidate(
                    frame_number=frame_num,
                    timestamp=timestamp,
                    bbox=det.bbox,
                    crop=crop,
                    confidence=det.confidence,
                    sharpness=sharpness,
                    area=area,
                )
                cand.compute_score(w, h)

                if track_id not in self._track_accumulators:
                    self._track_accumulators[track_id] = TrackAccumulator(
                        track_id=track_id,
                        camera_id=self.camera_id,
                        class_name=det.class_name,
                    )
                self._track_accumulators[track_id].add_candidate(cand)

            self.frames_processed += 1

            # Progress update every 50 frames
            if self.frames_processed % 50 == 0:
                pct = min(timestamp / total_duration, 1.0) * 0.55 + 0.10
                self._report_progress(
                    pct,
                    f"Frame {self.frames_processed} | "
                    f"Time {timestamp:.1f}s / {total_duration:.1f}s | "
                    f"Tracks: {len(self._track_accumulators)}"
                )

        self.tracks_found = len(self._track_accumulators)
        logger.info("Detection/tracking complete: %d frames, %d tracks",
                    self.frames_processed, self.tracks_found)

    def _phase_embed_and_store(self) -> None:
        """Select best crops, generate embeddings, store in SQLite + FAISS."""
        store = get_store()
        n_tracks = len(self._track_accumulators)

        for i, (track_id, acc) in enumerate(self._track_accumulators.items()):
            if self._is_stopped():
                break

            best_crops = acc.select_best_crops(cfg.max_crops_per_track)
            if not best_crops:
                continue

            # Save crops to disk
            crop_dir = cfg.crops_dir / self.camera_id / f"track_{track_id:04d}"
            saved_paths = []
            best_path = None

            for j, cand in enumerate(best_crops):
                save_path = crop_dir / f"crop_{j+1:02d}.jpg"
                if save_crop(cand.crop, save_path):
                    saved_paths.append(str(save_path))
                    if best_path is None:
                        best_path = str(save_path)

            if not saved_paths:
                continue

            # Estimate color from best crop
            dominant_color = estimate_dominant_color(best_crops[0].crop)

            # Store track in SQLite
            track_db_id = db.upsert_track(
                camera_id=self.camera_id,
                track_id=track_id,
                class_name=acc.class_name,
                first_seen=acc.first_seen,
                last_seen=acc.last_seen,
                frame_count=acc.frame_count,
                confidence=acc.max_confidence,
                dominant_color=dominant_color,
                best_crop=best_path,
                crop_paths=saved_paths,
            )

            # Store observations (sample, not all)
            track_state = self._tracker.active_tracks.get(track_id)
            if track_state and track_state.observations:
                obs_sample = track_state.observations[::max(1, len(track_state.observations) // 10)]
                for obs in obs_sample:
                    x1, y1, x2, y2 = obs["bbox"]
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    db.insert_observation(
                        track_db_id=track_db_id,
                        camera_id=self.camera_id,
                        track_id=track_id,
                        timestamp=obs["timestamp"],
                        frame_number=obs["frame_number"],
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=obs["confidence"],
                        center_x=cx, center_y=cy,
                    )

            # Generate CLIP embeddings for crops
            try:
                embeddings = self._encoder.encode_images(
                    [c.crop for c in best_crops]
                )

                # Mean embedding as representative
                mean_emb = embeddings.mean(axis=0)
                norm = np.linalg.norm(mean_emb)
                if norm > 0:
                    mean_emb = mean_emb / norm

                # Add to FAISS (representative + individual crops)
                all_embs = np.vstack([mean_emb.reshape(1, -1), embeddings])
                meta_list = []

                # Representative embedding (first)
                meta_list.append({
                    "track_db_id": track_db_id,
                    "camera_id": self.camera_id,
                    "crop_path": best_path,
                    "is_representative": True,
                    "class_name": acc.class_name,
                    "dominant_color": dominant_color,
                    "first_seen": acc.first_seen,
                    "last_seen": acc.last_seen,
                })

                # Individual crop embeddings
                for j, cand in enumerate(best_crops):
                    path = saved_paths[j] if j < len(saved_paths) else best_path
                    meta_list.append({
                        "track_db_id": track_db_id,
                        "camera_id": self.camera_id,
                        "crop_path": path,
                        "is_representative": False,
                        "class_name": acc.class_name,
                        "dominant_color": dominant_color,
                        "first_seen": cand.timestamp,
                        "last_seen": cand.timestamp,
                    })

                faiss_indices = store.add(all_embs, meta_list)

                # Store embedding records in SQLite (only representative)
                db.insert_embedding(
                    track_db_id=track_db_id,
                    camera_id=self.camera_id,
                    crop_path=best_path,
                    faiss_idx=faiss_indices[0],
                )

                self.embeddings_added += 1

            except Exception as e:
                logger.error("Embedding error for track %d: %s", track_id, e)
                continue

            if (i + 1) % 10 == 0:
                pct = 0.65 + (i / n_tracks) * 0.30
                self._report_progress(pct, f"Embedded {i+1}/{n_tracks} tracks")

        logger.info("Embedding complete: %d tracks embedded", self.embeddings_added)
