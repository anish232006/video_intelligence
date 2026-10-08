"""
src/reid/cross_camera.py — Cross-camera re-identification (stretch feature).

Uses CLIP appearance embeddings + time gap + class filtering.
Does NOT claim identity certainty — says "Possible same entity".

This module is completely optional — disable with:
    enable_cross_camera_reid: false (in config.yaml)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.config import cfg
from src import database as db

logger = logging.getLogger(__name__)


@dataclass
class CrossCameraMatch:
    """A proposed cross-camera match between two tracks."""
    track_a_id: int
    camera_a: str
    time_a: float
    track_b_id: int
    camera_b: str
    time_b: float
    similarity: float
    time_gap: float

    @property
    def label(self) -> str:
        return "Possible same entity"

    @property
    def confidence_label(self) -> str:
        if self.similarity >= 0.90:
            return "High"
        elif self.similarity >= 0.80:
            return "Medium"
        return "Low"


def find_cross_camera_matches(
    track_db_id: int,
    max_results: int = 5,
) -> List[CrossCameraMatch]:
    """
    Find tracks on OTHER cameras that might be the same entity.

    Uses CLIP embedding similarity + class match + time gap.
    Returns candidates sorted by similarity (descending).
    """
    if not cfg.enable_cross_camera_reid:
        return []

    try:
        from src.indexing.faiss_store import get_store
        store = get_store()

        if store.is_empty():
            return []

        # Get the target track
        target_track = db.get_track_by_db_id(track_db_id)
        if not target_track:
            return []

        target_camera = target_track.get("camera_id", "")
        target_class = target_track.get("class_name", "")
        target_first = target_track.get("first_seen", 0)
        target_last = target_track.get("last_seen", 0)

        # Get target track's FAISS embedding
        target_embs = db.get_embeddings_for_track(track_db_id)
        if not target_embs:
            return []

        representative_emb = target_embs[0]
        faiss_idx = representative_emb.get("faiss_idx")
        if faiss_idx is None:
            return []

        # Get embedding vector
        import faiss as faiss_lib
        if store._index is None or store._index.ntotal == 0:
            return []

        vec = np.zeros((1, cfg.embedding_dim), dtype=np.float32)
        store._index.reconstruct(faiss_idx, vec[0])

        # Search similar tracks
        scores, indices = store.search(vec[0], top_k=50)

        matches = []
        seen_tracks = {track_db_id}

        for score, idx in zip(scores, indices):
            meta = store.get_meta(idx)
            if not meta:
                continue

            other_track_id = meta.get("track_db_id")
            if other_track_id in seen_tracks:
                continue
            if meta.get("camera_id") == target_camera:
                continue  # same camera → not cross-camera
            if meta.get("class_name") != target_class:
                continue

            seen_tracks.add(other_track_id)
            other_track = db.get_track_by_db_id(other_track_id)
            if not other_track:
                continue

            # Time gap check
            other_first = other_track.get("first_seen", 0)
            other_last = other_track.get("last_seen", 0)
            time_gap = abs(other_first - target_last)

            if time_gap > cfg.reid_max_time_gap_seconds:
                continue

            if score < cfg.reid_similarity_threshold:
                continue

            matches.append(CrossCameraMatch(
                track_a_id=track_db_id,
                camera_a=target_camera,
                time_a=target_first,
                track_b_id=other_track_id,
                camera_b=meta.get("camera_id", ""),
                time_b=other_first,
                similarity=score,
                time_gap=time_gap,
            ))

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:max_results]

    except Exception as e:
        logger.error("Cross-camera ReID error: %s", e)
        return []
