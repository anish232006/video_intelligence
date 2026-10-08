"""
src/tracking/tracker.py — Multi-object tracker with ByteTrack support.

Uses Ultralytics ByteTrack integration. Falls back to simple IoU tracker
if ByteTrack is unavailable.

CRITICAL: Tracker state must NOT leak between cameras.
Create a new instance per camera.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detection.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """Live tracking state for one object."""
    track_id: int
    class_name: str
    class_id: int
    observations: List[Dict] = field(default_factory=list)
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    disappeared: int = 0
    frame_count: int = 0

    @property
    def first_seen(self) -> float:
        return self.observations[0]["timestamp"] if self.observations else 0.0

    @property
    def last_seen(self) -> float:
        return self.observations[-1]["timestamp"] if self.observations else 0.0


class SimpleIoUTracker:
    """
    Fallback IoU-based tracker when ByteTrack unavailable.
    Not as robust as ByteTrack but functional.
    """

    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.3):
        self.tracks: Dict[int, TrackState] = {}
        self.next_id = 1
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    @staticmethod
    def _iou(a: Tuple, b: Tuple) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: List[Detection],
               frame_number: int, timestamp: float) -> List[Tuple[int, Detection]]:
        """
        Match detections to existing tracks.
        Returns list of (track_id, detection) tuples.
        """
        # Mark all tracks as possibly disappeared
        for t in self.tracks.values():
            t.disappeared += 1

        if not detections:
            # Remove stale tracks
            stale = [tid for tid, t in self.tracks.items()
                     if t.disappeared > self.max_disappeared]
            for tid in stale:
                del self.tracks[tid]
            return []

        results = []
        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        # Try to match detections to existing tracks
        for det_idx in list(unmatched_dets):
            det = detections[det_idx]
            best_iou = self.iou_threshold
            best_tid = None

            for tid, track in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                if track.class_name != det.class_name:
                    continue
                if track.last_bbox is None:
                    continue
                iou = self._iou(track.last_bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                track = self.tracks[best_tid]
                track.last_bbox = det.bbox
                track.confidence = max(track.confidence, det.confidence)
                track.disappeared = 0
                track.frame_count += 1
                track.observations.append({
                    "timestamp": timestamp,
                    "frame_number": frame_number,
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                })
                results.append((best_tid, det))
                matched_track_ids.add(best_tid)
                unmatched_dets.remove(det_idx)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = TrackState(
                track_id=tid,
                class_name=det.class_name,
                class_id=det.class_id,
                last_bbox=det.bbox,
                confidence=det.confidence,
                frame_count=1,
                observations=[{
                    "timestamp": timestamp,
                    "frame_number": frame_number,
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                }],
            )
            results.append((tid, det))

        # Remove stale tracks
        stale = [tid for tid, t in self.tracks.items()
                 if t.disappeared > self.max_disappeared]
        for tid in stale:
            del self.tracks[tid]

        return results


class ByteTrackWrapper:
    """
    Wrapper around Ultralytics ByteTrack.
    Falls back to SimpleIoUTracker if ByteTrack is unavailable.
    """

    def __init__(self, max_disappeared: int = None):
        self.max_disappeared = max_disappeared or 30
        self._use_bytetrack = False
        self._bt_tracker = None
        self._iou_tracker = SimpleIoUTracker(max_disappeared=self.max_disappeared)
        self._try_init_bytetrack()

    def _try_init_bytetrack(self) -> None:
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker

            class _Args:
                track_high_thresh = 0.5
                track_low_thresh = 0.1
                new_track_thresh = 0.6
                track_buffer = 30
                match_thresh = 0.8
                mot20 = False
                frame_rate = 2

            try:
                self._bt_tracker = BYTETracker(_Args())
            except TypeError:
                self._bt_tracker = BYTETracker(_Args(), frame_rate=2)
            self._use_bytetrack = True
            logger.info("ByteTrack initialized")
        except Exception as e:
            logger.info("ByteTrack unavailable (%s), using IoU tracker", e)
            self._use_bytetrack = False

    def reset(self) -> None:
        """CRITICAL: Reset tracker state between cameras."""
        self._iou_tracker.reset()
        if self._use_bytetrack:
            try:
                from ultralytics.trackers.byte_tracker import BYTETracker

                class _Args:
                    track_high_thresh = 0.5
                    track_low_thresh = 0.1
                    new_track_thresh = 0.6
                    track_buffer = 30
                    match_thresh = 0.8
                    mot20 = False
                    frame_rate = 2

                try:
                    self._bt_tracker = BYTETracker(_Args())
                except TypeError:
                    self._bt_tracker = BYTETracker(_Args(), frame_rate=2)
            except Exception:
                self._use_bytetrack = False

    def update(self, detections: List[Detection],
               frame_number: int, timestamp: float,
               frame_shape: Tuple[int, int]) -> List[Tuple[int, Detection]]:
        """
        Update tracker with new detections.
        Returns (track_id, detection) pairs.
        """
        if self._use_bytetrack:
            return self._update_bytetrack(detections, frame_number, timestamp, frame_shape)
        else:
            return self._iou_tracker.update(detections, frame_number, timestamp)

    def _update_bytetrack(self, detections: List[Detection],
                          frame_number: int, timestamp: float,
                          frame_shape: Tuple[int, int]) -> List[Tuple[int, Detection]]:
        try:
            if not detections:
                return []

            h, w = frame_shape
            # Build detection array: [x1, y1, x2, y2, conf, cls]
            det_arr = np.array([
                [d.x1, d.y1, d.x2, d.y2, d.confidence, d.class_id]
                for d in detections
            ], dtype=np.float32)

            tracks = self._bt_tracker.update(det_arr, (h, w), (h, w))

            results = []
            for track in tracks:
                tid = int(track.track_id)
                x1, y1, x2, y2 = [int(v) for v in track.tlbr]
                cls_id = int(track.cls)
                conf = float(track.score)

                # Find matching original detection
                matched_det = None
                for d in detections:
                    if d.class_id == cls_id:
                        matched_det = d
                        break

                if matched_det is None:
                    matched_det = Detection(x1, y1, x2, y2, conf, cls_id,
                                           str(cls_id))

                # Update IoU tracker state for consistency
                if tid not in self._iou_tracker.tracks:
                    self._iou_tracker.tracks[tid] = TrackState(
                        track_id=tid,
                        class_name=matched_det.class_name,
                        class_id=matched_det.class_id,
                        confidence=conf,
                        frame_count=1,
                    )
                track_state = self._iou_tracker.tracks[tid]
                track_state.last_bbox = (x1, y1, x2, y2)
                track_state.frame_count += 1
                track_state.observations.append({
                    "timestamp": timestamp,
                    "frame_number": frame_number,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                })

                results.append((tid, matched_det))

            return results

        except Exception as e:
            logger.warning("ByteTrack update failed (%s), falling back to IoU", e)
            self._use_bytetrack = False
            return self._iou_tracker.update(detections, frame_number, timestamp)

    @property
    def active_tracks(self) -> Dict[int, TrackState]:
        return self._iou_tracker.tracks
