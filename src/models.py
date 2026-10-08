"""
src/models.py — Pydantic data models (schemas) for the entire system.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


# ─── Camera ─────────────────────────────────────────────────────────────────
class CameraInfo(BaseModel):
    camera_id: str
    name: str
    description: Optional[str] = None
    video_path: str
    duration: float = 0.0          # seconds
    fps: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    indexed: bool = False
    indexed_at: Optional[datetime] = None


# ─── Track ──────────────────────────────────────────────────────────────────
class TrackObservation(BaseModel):
    track_id: int
    camera_id: str
    timestamp: float               # seconds from video start
    frame_number: int
    x1: int
    y1: int
    x2: int
    y2: int
    crop_path: Optional[str] = None
    confidence: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0


class Track(BaseModel):
    track_id: int
    camera_id: str
    class_name: str
    first_seen: float              # seconds from video start
    last_seen: float
    frame_count: int = 0
    confidence: float = 0.0
    dominant_color: Optional[str] = None
    crop_paths: List[str] = Field(default_factory=list)
    best_crop: Optional[str] = None
    embedding_ids: List[int] = Field(default_factory=list)
    db_id: Optional[int] = None    # SQLite row id


# ─── Query ──────────────────────────────────────────────────────────────────
class ParsedQuery(BaseModel):
    """Structured representation of a natural-language query."""
    object_class: Optional[str] = None        # "car", "person", etc.
    attributes: List[str] = Field(default_factory=list)   # ["red", "backpack"]
    location: Optional[str] = None            # "main gate"
    cameras: List[str] = Field(default_factory=list)      # ["CAM_01"]
    time_range: Optional[str] = None          # "last 1 hour", "all", etc.
    time_start: Optional[float] = None        # resolved timestamp (epoch or video seconds)
    time_end: Optional[float] = None
    intent: str = "find_event"               # "find_event", "count", "timeline"
    raw_query: str = ""
    search_text: str = ""                    # text to embed (e.g. "red car")


# ─── Search Result ──────────────────────────────────────────────────────────
class SearchResult(BaseModel):
    track_db_id: int
    camera_id: str
    camera_name: str
    timestamp: float               # video seconds
    object_class: str
    dominant_color: Optional[str] = None
    confidence: float = 0.0
    semantic_score: float = 0.0
    rerank_score: float = 0.0
    crop_path: Optional[str] = None
    explanation: str = ""
    clip_path: Optional[str] = None
    clip_generated: bool = False


# ─── Knowledge / Memory ─────────────────────────────────────────────────────
class KnowledgeEntry(BaseModel):
    term: str
    entry_type: str = "location"    # "location", "entity"
    camera_id: Optional[str] = None
    region_json: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)


# ─── Evaluation ─────────────────────────────────────────────────────────────
class EvalQuery(BaseModel):
    query: str
    expected_camera: Optional[str] = None
    expected_time_start: Optional[float] = None
    expected_time_end: Optional[float] = None
    expected_class: Optional[str] = None


class EvalResult(BaseModel):
    query: str
    expected_camera: Optional[str]
    top1_camera: Optional[str]
    top1_time: Optional[float]
    camera_correct: bool = False
    time_error: Optional[float] = None
    latency_ms: float = 0.0
    found: bool = False
