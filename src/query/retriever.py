"""
src/query/retriever.py — Full retrieval pipeline.

Pipeline:
1. Parse structured query
2. Resolve location → camera_id (from memory)
3. Apply SQL hard filters (camera, class, color, time)
4. Generate CLIP text embedding
5. Search FAISS
6. Rerank candidates
7. Merge temporal duplicates
8. Return grounded results

CRITICAL: Never re-run YOLO on query. Index once, search many times.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from src import database as db
from src.config import cfg
from src.embeddings.clip_encoder import CLIPEncoder
from src.embeddings.color import color_similarity_score
from src.indexing.faiss_store import get_store
from src.models import SearchResult
from src.query.memory import resolve_location_to_camera
from src.query.schema import StructuredQuery
from src.query.time_parser import format_timestamp, get_max_video_seconds, parse_time_range

logger = logging.getLogger(__name__)

# Global CLIP encoder (kept loaded between queries for speed)
_encoder: Optional[CLIPEncoder] = None


def _get_encoder() -> CLIPEncoder:
    global _encoder
    if _encoder is None or not _encoder.is_loaded:
        _encoder = CLIPEncoder()
        _encoder.load()
    return _encoder


class RetrievalPipeline:
    """
    Main retrieval pipeline. Query → Grounded results.
    """

    def __init__(self):
        self._last_query_time = 0.0

    def search(self, structured_query: StructuredQuery,
               top_k: int = None) -> Tuple[List[SearchResult], Optional[str]]:
        """
        Execute retrieval pipeline.

        Returns:
            (results, clarification_needed)
            clarification_needed is set when a location is unrecognized.
        """
        top_k = top_k or cfg.top_k
        t0 = time.time()

        # ── Step 1: Resolve location ──────────────────────────────────────
        resolved_cameras = list(structured_query.cameras)  # explicit cameras
        clarification_needed = None

        if structured_query.location and not resolved_cameras:
            cam_id = resolve_location_to_camera(structured_query.location)
            if cam_id:
                resolved_cameras = [cam_id]
                logger.info("Location '%s' → %s", structured_query.location, cam_id)
            else:
                # Unknown location — need clarification
                clarification_needed = structured_query.location
                logger.info("Unknown location: '%s' — need clarification",
                            structured_query.location)
                # Still search all cameras
                resolved_cameras = []

        # ── Step 2: Resolve time range ────────────────────────────────────
        max_secs = get_max_video_seconds()
        time_start, time_end = parse_time_range(
            structured_query.time_range or "all", max_secs
        )

        # ── Step 3: SQL filter (strict class-aware) ──────────────────────
        filtered_tracks = db.get_tracks_filtered(
            camera_ids=resolved_cameras if resolved_cameras else None,
            class_name=structured_query.object_class,
            color=None,
            time_start=time_start,
            time_end=time_end,
        )

        # If user specified time range and no tracks found, try relaxing time range first
        if not filtered_tracks and (time_start is not None or time_end is not None):
            filtered_tracks = db.get_tracks_filtered(
                camera_ids=resolved_cameras if resolved_cameras else None,
                class_name=structured_query.object_class,
            )

        # CRITICAL: If user explicitly searched for an object class (e.g. truck, dog)
        # and none exist in footage, DO NOT fall back to returning random cars/people!
        if not filtered_tracks and structured_query.object_class:
            logger.info("No tracks found matching requested class '%s'", structured_query.object_class)
            return [], clarification_needed

        # Fallback only when query didn't specify an explicit object class
        if not filtered_tracks:
            filtered_tracks = db.get_all_tracks()

        if not filtered_tracks:
            logger.info("No tracks available in database")
            return [], clarification_needed

        # Get valid track_db_ids
        valid_track_ids = {t["id"] for t in filtered_tracks}
        track_map = {t["id"]: t for t in filtered_tracks}

        # ── Step 4: CLIP text embedding ───────────────────────────────────
        search_text = structured_query.search_text
        logger.info("Searching for: '%s'", search_text)

        try:
            encoder = _get_encoder()
            text_emb = encoder.encode_single_text(search_text)
        except Exception as e:
            logger.error("CLIP encoding failed: %s", e)
            # Fallback: return best SQL matches without semantic search
            return self._fallback_results(filtered_tracks, top_k), clarification_needed

        # ── Step 5: FAISS search ──────────────────────────────────────────
        store = get_store()
        if store.is_empty():
            logger.warning("FAISS index is empty — falling back to database tracks")
            return self._fallback_results(filtered_tracks, top_k), clarification_needed

        # Search the entire index so tracks across all cameras are evaluated
        search_k = store.total_vectors
        faiss_scores, faiss_indices = store.search(text_emb, top_k=search_k)

        if not faiss_indices:
            logger.warning("FAISS returned no indices — falling back to database tracks")
            return self._fallback_results(filtered_tracks, top_k), clarification_needed

        # ── Step 6: Rerank candidates ─────────────────────────────────────
        candidates = []
        seen_track_ids = set()

        for score, idx in zip(faiss_scores, faiss_indices):
            meta = store.get_meta(idx)
            if meta is None:
                continue

            track_db_id = meta.get("track_db_id")
            if track_db_id not in valid_track_ids:
                continue

            # Avoid duplicate tracks (keep best scoring)
            if track_db_id in seen_track_ids:
                continue
            seen_track_ids.add(track_db_id)

            track = track_map.get(track_db_id, {})
            cam_info = db.get_camera(meta.get("camera_id", ""))

            rerank = self._compute_rerank_score(
                semantic_score=score,
                track=track,
                meta=meta,
                structured_query=structured_query,
                resolved_cameras=resolved_cameras,
                time_start=time_start,
                time_end=time_end,
            )

            # Get best observation for timestamp
            best_obs = db.get_best_observation(track_db_id)
            timestamp = track.get("first_seen", 0.0)
            if best_obs:
                timestamp = best_obs.get("timestamp", timestamp)

            explanation = self._build_explanation(
                track=track,
                meta=meta,
                semantic_score=score,
                rerank_score=rerank,
                structured_query=structured_query,
                resolved_cameras=resolved_cameras,
                timestamp=timestamp,
            )

            candidates.append(SearchResult(
                track_db_id=track_db_id,
                camera_id=meta.get("camera_id", ""),
                camera_name=(cam_info or {}).get("name", meta.get("camera_id", "")),
                timestamp=timestamp,
                object_class=track.get("class_name", meta.get("class_name", "")),
                dominant_color=track.get("dominant_color"),
                confidence=track.get("confidence", score),
                semantic_score=score,
                rerank_score=rerank,
                crop_path=track.get("best_crop") or meta.get("crop_path"),
                explanation=explanation,
            ))

        # Filter out candidates below minimum score threshold
        min_threshold = 0.28 if structured_query.color_hints else 0.20
        candidates = [c for c in candidates if c.rerank_score >= min_threshold]

        # Sort by rerank score
        candidates.sort(key=lambda r: r.rerank_score, reverse=True)

        # ── Step 7: Temporal merging ──────────────────────────────────────
        merged = self._merge_temporal_duplicates(candidates)

        # Take top_k
        results = merged[:top_k]

        latency_ms = (time.time() - t0) * 1000
        logger.info("Retrieval: %d results in %.1fms", len(results), latency_ms)

        # Log query
        try:
            import json
            db.log_query(
                query_text=structured_query.raw_query,
                parsed_json=json.dumps(structured_query.model_dump()),
                result_count=len(results),
                latency_ms=latency_ms,
            )
        except Exception:
            pass

        return results, clarification_needed

    def _compute_rerank_score(self,
                              semantic_score: float,
                              track: dict,
                              meta: dict,
                              structured_query: StructuredQuery,
                              resolved_cameras: List[str],
                              time_start: Optional[float],
                              time_end: Optional[float]) -> float:
        """
        Rerank score combining multiple signals.
        """
        score = semantic_score * 0.50  # CLIP similarity is primary signal

        # Color match bonus / mismatch penalty
        if structured_query.color_hints:
            color_score = color_similarity_score(
                track.get("dominant_color"),
                structured_query.color_hints,
            )
            if color_score > 0.0:
                score += color_score * 0.30
            else:
                # If user specifically asked for a color (e.g. green) and this object is another color, penalize
                if track.get("dominant_color"):
                    score -= 0.35

        # Class match bonus
        if structured_query.object_class:
            if track.get("class_name") == structured_query.object_class:
                score += 0.15
            else:
                score -= 0.10

        # Camera match bonus
        if resolved_cameras and track.get("camera_id") in resolved_cameras:
            score += 0.10

        # Time overlap score
        if time_start is not None and time_end is not None:
            t_first = track.get("first_seen", 0)
            t_last = track.get("last_seen", 0)
            overlap_start = max(t_first, time_start)
            overlap_end = min(t_last, time_end)
            if overlap_end > overlap_start:
                range_len = time_end - time_start + 1
                overlap = (overlap_end - overlap_start) / range_len
                score += min(overlap, 1.0) * 0.05

        # Confidence bonus
        score += track.get("confidence", 0) * 0.05

        return max(0.0, min(score, 1.0))

    def _merge_temporal_duplicates(self, results: List[SearchResult],
                                   time_gap: float = 3.0) -> List[SearchResult]:
        """
        Merge results from the same camera with timestamps close together
        (likely the same event observed via multiple crops).
        """
        if len(results) <= 1:
            return results

        merged = []
        used = set()

        for i, r in enumerate(results):
            if i in used:
                continue
            group = [r]
            for j, r2 in enumerate(results[i+1:], start=i+1):
                if j in used:
                    continue
                if (r2.camera_id == r.camera_id and
                        r2.object_class == r.object_class and
                        abs(r2.timestamp - r.timestamp) < time_gap):
                    group.append(r2)
                    used.add(j)

            # Keep the highest-scoring from each group
            best = max(group, key=lambda x: x.rerank_score)
            merged.append(best)
            used.add(i)

        return merged

    def _fallback_results(self, tracks: List[dict], top_k: int) -> List[SearchResult]:
        """Return top tracks by confidence when CLIP is unavailable."""
        sorted_tracks = sorted(tracks, key=lambda t: t.get("confidence", 0), reverse=True)
        results = []
        for t in sorted_tracks[:top_k]:
            cam_info = db.get_camera(t.get("camera_id", ""))
            results.append(SearchResult(
                track_db_id=t["id"],
                camera_id=t.get("camera_id", ""),
                camera_name=(cam_info or {}).get("name", t.get("camera_id", "")),
                timestamp=t.get("first_seen", 0.0),
                object_class=t.get("class_name", ""),
                dominant_color=t.get("dominant_color"),
                confidence=t.get("confidence", 0),
                semantic_score=0.0,
                rerank_score=t.get("confidence", 0),
                crop_path=t.get("best_crop"),
                explanation="Result based on SQL filter (CLIP unavailable).",
            ))
        return results

    def _build_explanation(self, track: dict, meta: dict,
                           semantic_score: float, rerank_score: float,
                           structured_query: StructuredQuery,
                           resolved_cameras: List[str],
                           timestamp: float) -> str:
        """Build a human-readable explanation for a result."""
        parts = []

        cam_id = track.get("camera_id", meta.get("camera_id", ""))
        class_name = track.get("class_name", "object")
        color = track.get("dominant_color", "")
        ts = format_timestamp(timestamp)

        desc = f"{color} {class_name}".strip() if color else class_name
        parts.append(f"Matched a {desc} track on {cam_id} at {ts}.")

        sim_pct = int(semantic_score * 100)
        parts.append(f"Visual similarity: {sim_pct}%.")

        if structured_query.color_hints and color:
            if color in structured_query.color_hints:
                parts.append(f"Color '{color}' matches query.")
            else:
                parts.append(f"Stored color '{color}' (query: {', '.join(structured_query.color_hints)}).")

        if resolved_cameras and cam_id in resolved_cameras:
            if structured_query.location:
                parts.append(
                    f"Camera mapping: '{structured_query.location}' → {cam_id}."
                )

        return " ".join(parts)
