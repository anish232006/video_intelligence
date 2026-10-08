"""
src/database.py — SQLite database layer.
All database operations are wrapped in transactions.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from src.config import cfg

logger = logging.getLogger(__name__)

DB_PATH = cfg.db_path


# ─── Connection helper ───────────────────────────────────────────────────────
@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─── Schema initialization ───────────────────────────────────────────────────
def init_db() -> None:
    """Create all tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS cameras (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id   TEXT    NOT NULL UNIQUE,
            name        TEXT    NOT NULL,
            description TEXT,
            video_path  TEXT    NOT NULL,
            duration    REAL    DEFAULT 0,
            fps         REAL    DEFAULT 0,
            width       INTEGER DEFAULT 0,
            height      INTEGER DEFAULT 0,
            frame_count INTEGER DEFAULT 0,
            indexed     INTEGER DEFAULT 0,
            indexed_at  TEXT,
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS tracks (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id        TEXT    NOT NULL,
            track_id         INTEGER NOT NULL,
            class_name       TEXT    NOT NULL,
            first_seen       REAL    NOT NULL,
            last_seen        REAL    NOT NULL,
            frame_count      INTEGER DEFAULT 0,
            confidence       REAL    DEFAULT 0,
            dominant_color   TEXT,
            best_crop        TEXT,
            crop_paths_json  TEXT,
            UNIQUE(camera_id, track_id)
        );

        CREATE TABLE IF NOT EXISTS track_observations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            track_db_id  INTEGER NOT NULL REFERENCES tracks(id),
            camera_id    TEXT    NOT NULL,
            track_id     INTEGER NOT NULL,
            timestamp    REAL    NOT NULL,
            frame_number INTEGER NOT NULL,
            x1           INTEGER,
            y1           INTEGER,
            x2           INTEGER,
            y2           INTEGER,
            crop_path    TEXT,
            confidence   REAL    DEFAULT 0,
            center_x     REAL,
            center_y     REAL
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            track_db_id INTEGER NOT NULL REFERENCES tracks(id),
            camera_id   TEXT    NOT NULL,
            crop_path   TEXT,
            faiss_idx   INTEGER,
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS knowledge (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            term        TEXT    NOT NULL UNIQUE,
            type        TEXT    DEFAULT 'location',
            camera_id   TEXT,
            region_json TEXT,
            created_at  TEXT    DEFAULT (datetime('now')),
            updated_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS aliases (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_term TEXT    NOT NULL,
            alias          TEXT    NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS queries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text  TEXT    NOT NULL,
            parsed_json TEXT,
            result_count INTEGER DEFAULT 0,
            created_at  TEXT    DEFAULT (datetime('now')),
            latency_ms  REAL
        );

        CREATE INDEX IF NOT EXISTS idx_tracks_camera     ON tracks(camera_id);
        CREATE INDEX IF NOT EXISTS idx_tracks_class      ON tracks(class_name);
        CREATE INDEX IF NOT EXISTS idx_tracks_color      ON tracks(dominant_color);
        CREATE INDEX IF NOT EXISTS idx_obs_camera_time   ON track_observations(camera_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_obs_track         ON track_observations(track_db_id);
        CREATE INDEX IF NOT EXISTS idx_emb_track         ON embeddings(track_db_id);
        CREATE INDEX IF NOT EXISTS idx_aliases_canonical ON aliases(canonical_term);
        """)
    logger.info("Database initialized at %s", DB_PATH)


# ─── Camera operations ────────────────────────────────────────────────────────
def upsert_camera(camera_id: str, name: str, video_path: str,
                  description: Optional[str] = None,
                  duration: float = 0, fps: float = 0,
                  width: int = 0, height: int = 0,
                  frame_count: int = 0) -> int:
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO cameras (camera_id, name, description, video_path,
                                 duration, fps, width, height, frame_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(camera_id) DO UPDATE SET
                name=excluded.name,
                description=excluded.description,
                video_path=excluded.video_path,
                duration=excluded.duration,
                fps=excluded.fps,
                width=excluded.width,
                height=excluded.height,
                frame_count=excluded.frame_count
        """, (camera_id, name, description, video_path,
              duration, fps, width, height, frame_count))
        row = conn.execute("SELECT id FROM cameras WHERE camera_id=?",
                           (camera_id,)).fetchone()
        return row["id"]


def mark_camera_indexed(camera_id: str) -> None:
    with get_conn() as conn:
        conn.execute("""
            UPDATE cameras SET indexed=1, indexed_at=? WHERE camera_id=?
        """, (datetime.now(timezone.utc).isoformat(), camera_id))


def get_all_cameras() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM cameras ORDER BY camera_id").fetchall()
        return [dict(r) for r in rows]


def get_camera(camera_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM cameras WHERE camera_id=?",
                           (camera_id,)).fetchone()
        return dict(row) if row else None


def delete_camera(camera_id: str) -> None:
    """Remove a camera and all its tracks/observations/embeddings."""
    with get_conn() as conn:
        conn.execute("DELETE FROM cameras WHERE camera_id=?", (camera_id,))
        track_ids = [r[0] for r in conn.execute(
            "SELECT id FROM tracks WHERE camera_id=?", (camera_id,)).fetchall()]
        for tid in track_ids:
            conn.execute("DELETE FROM track_observations WHERE track_db_id=?", (tid,))
            conn.execute("DELETE FROM embeddings WHERE track_db_id=?", (tid,))
        conn.execute("DELETE FROM tracks WHERE camera_id=?", (camera_id,))


# ─── Track operations ─────────────────────────────────────────────────────────
def upsert_track(camera_id: str, track_id: int, class_name: str,
                 first_seen: float, last_seen: float,
                 frame_count: int = 0, confidence: float = 0.0,
                 dominant_color: Optional[str] = None,
                 best_crop: Optional[str] = None,
                 crop_paths: Optional[List[str]] = None) -> int:
    crop_json = json.dumps(crop_paths or [])
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO tracks (camera_id, track_id, class_name, first_seen,
                                last_seen, frame_count, confidence, dominant_color,
                                best_crop, crop_paths_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(camera_id, track_id) DO UPDATE SET
                class_name=excluded.class_name,
                first_seen=MIN(tracks.first_seen, excluded.first_seen),
                last_seen=MAX(tracks.last_seen, excluded.last_seen),
                frame_count=excluded.frame_count,
                confidence=excluded.confidence,
                dominant_color=excluded.dominant_color,
                best_crop=excluded.best_crop,
                crop_paths_json=excluded.crop_paths_json
        """, (camera_id, track_id, class_name, first_seen, last_seen,
              frame_count, confidence, dominant_color, best_crop, crop_json))
        row = conn.execute(
            "SELECT id FROM tracks WHERE camera_id=? AND track_id=?",
            (camera_id, track_id)).fetchone()
        return row["id"]


def get_tracks_for_camera(camera_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tracks WHERE camera_id=? ORDER BY first_seen",
            (camera_id,)).fetchall()
        return [dict(r) for r in rows]


def get_all_tracks() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM tracks ORDER BY camera_id, first_seen").fetchall()
        return [dict(r) for r in rows]


def get_track_count() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]


def get_tracks_filtered(camera_ids: Optional[List[str]] = None,
                        class_name: Optional[str] = None,
                        color: Optional[str] = None,
                        time_start: Optional[float] = None,
                        time_end: Optional[float] = None) -> List[Dict[str, Any]]:
    """Hard-filter tracks by camera / class / color / time range."""
    query = "SELECT * FROM tracks WHERE 1=1"
    params: List[Any] = []

    if camera_ids:
        placeholders = ",".join("?" * len(camera_ids))
        query += f" AND camera_id IN ({placeholders})"
        params.extend(camera_ids)
    if class_name:
        query += " AND class_name=?"
        params.append(class_name)
    if color:
        query += " AND dominant_color=?"
        params.append(color)
    if time_start is not None:
        query += " AND last_seen>=?"
        params.append(time_start)
    if time_end is not None:
        query += " AND first_seen<=?"
        params.append(time_end)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


# ─── Observation operations ───────────────────────────────────────────────────
def insert_observation(track_db_id: int, camera_id: str, track_id: int,
                       timestamp: float, frame_number: int,
                       x1: int, y1: int, x2: int, y2: int,
                       crop_path: Optional[str] = None,
                       confidence: float = 0.0,
                       center_x: float = 0.0, center_y: float = 0.0) -> int:
    with get_conn() as conn:
        cursor = conn.execute("""
            INSERT INTO track_observations
            (track_db_id, camera_id, track_id, timestamp, frame_number,
             x1, y1, x2, y2, crop_path, confidence, center_x, center_y)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (track_db_id, camera_id, track_id, timestamp, frame_number,
              x1, y1, x2, y2, crop_path, confidence, center_x, center_y))
        return cursor.lastrowid


def get_best_observation(track_db_id: int) -> Optional[Dict[str, Any]]:
    """Get the observation with highest confidence for a track."""
    with get_conn() as conn:
        row = conn.execute("""
            SELECT * FROM track_observations
            WHERE track_db_id=?
            ORDER BY confidence DESC
            LIMIT 1
        """, (track_db_id,)).fetchone()
        return dict(row) if row else None


# ─── Embedding operations ─────────────────────────────────────────────────────
def insert_embedding(track_db_id: int, camera_id: str,
                     crop_path: Optional[str],
                     faiss_idx: int) -> int:
    with get_conn() as conn:
        cursor = conn.execute("""
            INSERT INTO embeddings (track_db_id, camera_id, crop_path, faiss_idx)
            VALUES (?, ?, ?, ?)
        """, (track_db_id, camera_id, crop_path, faiss_idx))
        return cursor.lastrowid


def get_embeddings_for_track(track_db_id: int) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM embeddings WHERE track_db_id=?", (track_db_id,)).fetchall()
        return [dict(r) for r in rows]


def get_track_by_db_id(track_db_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM tracks WHERE id=?", (track_db_id,)).fetchone()
        return dict(row) if row else None


def get_embeddings_by_faiss_indices(faiss_indices: List[int]) -> List[Dict[str, Any]]:
    if not faiss_indices:
        return []
    placeholders = ",".join("?" * len(faiss_indices))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM embeddings WHERE faiss_idx IN ({placeholders})",
            faiss_indices).fetchall()
        return [dict(r) for r in rows]


# ─── Knowledge / Memory operations ───────────────────────────────────────────
def save_knowledge(term: str, camera_id: str,
                   region_json: Optional[str] = None,
                   entry_type: str = "location") -> None:
    """Save or update a location knowledge entry."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO knowledge (term, type, camera_id, region_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(term) DO UPDATE SET
                camera_id=excluded.camera_id,
                region_json=excluded.region_json,
                updated_at=datetime('now')
        """, (term.lower().strip(), entry_type, camera_id, region_json))


def add_alias(canonical_term: str, alias: str) -> None:
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO aliases (canonical_term, alias)
            VALUES (?, ?)
        """, (canonical_term.lower().strip(), alias.lower().strip()))


def resolve_location(term: str) -> Optional[Dict[str, Any]]:
    """Resolve a location term to camera_id via aliases or direct lookup."""
    term_lower = term.lower().strip()
    with get_conn() as conn:
        # Direct lookup
        row = conn.execute(
            "SELECT * FROM knowledge WHERE term=?", (term_lower,)).fetchone()
        if row:
            return dict(row)
        # Alias lookup
        alias_row = conn.execute(
            "SELECT canonical_term FROM aliases WHERE alias=?",
            (term_lower,)).fetchone()
        if alias_row:
            canonical = alias_row["canonical_term"]
            row = conn.execute(
                "SELECT * FROM knowledge WHERE term=?", (canonical,)).fetchone()
            if row:
                return dict(row)
    return None


def get_all_knowledge() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT k.*, GROUP_CONCAT(a.alias, ', ') as aliases_str
            FROM knowledge k
            LEFT JOIN aliases a ON a.canonical_term = k.term
            GROUP BY k.id
            ORDER BY k.term
        """).fetchall()
        return [dict(r) for r in rows]


def delete_knowledge(term: str) -> None:
    term_lower = term.lower().strip()
    with get_conn() as conn:
        conn.execute("DELETE FROM knowledge WHERE term=?", (term_lower,))
        conn.execute("DELETE FROM aliases WHERE canonical_term=?", (term_lower,))


# ─── Query logging ────────────────────────────────────────────────────────────
def log_query(query_text: str, parsed_json: str,
              result_count: int, latency_ms: float) -> None:
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO queries (query_text, parsed_json, result_count, latency_ms)
            VALUES (?, ?, ?, ?)
        """, (query_text, parsed_json, result_count, latency_ms))


# ─── Stats ────────────────────────────────────────────────────────────────────
def get_stats() -> Dict[str, Any]:
    with get_conn() as conn:
        cameras = conn.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
        indexed = conn.execute("SELECT COUNT(*) FROM cameras WHERE indexed=1").fetchone()[0]
        tracks = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        embeddings = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        knowledge = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
        total_duration = conn.execute(
            "SELECT SUM(duration) FROM cameras WHERE indexed=1").fetchone()[0] or 0
    return {
        "cameras": cameras,
        "indexed_cameras": indexed,
        "tracks": tracks,
        "embeddings": embeddings,
        "knowledge_entries": knowledge,
        "total_indexed_duration_hours": round(total_duration / 3600, 2),
    }
