"""
src/indexing/faiss_store.py — FAISS vector index management.

Uses CPU FAISS with FlatIP (inner product) for cosine similarity
on L2-normalized embeddings.

Design:
- embeddings stored as float32 in FAISS
- metadata (track_db_id, camera_id, crop_path) stored in JSON sidecar
- index persisted to disk after each camera is indexed
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import cfg

logger = logging.getLogger(__name__)


class FAISSStore:
    """
    FAISS-based vector store with JSON metadata sidecar.

    Thread-safety: not thread-safe. Use a single instance.
    """

    def __init__(self,
                 index_path: Optional[Path] = None,
                 meta_path: Optional[Path] = None,
                 dim: int = None):
        self.index_path = index_path or cfg.faiss_index_path
        self.meta_path = meta_path or cfg.faiss_meta_path
        self.dim = dim or cfg.embedding_dim
        self._index = None
        self._meta: List[Dict] = []   # parallel list to FAISS vectors

    def _ensure_faiss(self):
        try:
            import faiss
            return faiss
        except ImportError:
            logger.error("faiss-cpu not installed. Run: pip install faiss-cpu")
            raise

    def load_or_create(self) -> None:
        """Load existing index from disk, or create a new empty one."""
        faiss = self._ensure_faiss()

        if self.index_path.exists() and self.meta_path.exists():
            try:
                self._index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)
                logger.info("FAISS index loaded: %d vectors from %s",
                            self._index.ntotal, self.index_path)
                return
            except Exception as e:
                logger.warning("Failed to load FAISS index (%s), creating new", e)

        self._index = faiss.IndexFlatIP(self.dim)
        self._meta = []
        logger.info("FAISS: Created new FlatIP index (dim=%d)", self.dim)

    def add(self, embeddings: np.ndarray, metadata: Optional[List[Dict]] = None) -> List[int]:
        """
        Add embeddings and associated metadata to the index.

        Args:
            embeddings: (N, dim) float32 array, L2-normalized
            metadata: Optional List of N dicts describing each embedding

        Returns:
            List of FAISS indices assigned to each vector.
        """
        if self._index is None:
            self.load_or_create()

        if len(embeddings) == 0:
            return []

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        start_idx = self._index.ntotal
        self._index.add(embeddings)
        if metadata is not None:
            self._meta.extend(metadata)
        else:
            self._meta.extend([{} for _ in range(len(embeddings))])

        indices = list(range(start_idx, self._index.ntotal))
        return indices

    def search(self, query: np.ndarray, top_k: int = 20) -> Tuple[List[float], List[int]]:
        """
        Search the index for top_k nearest neighbors.

        Args:
            query: (dim,) or (1, dim) float32 array, L2-normalized
            top_k: Number of results to return

        Returns:
            (scores, indices) — both sorted descending by score
        """
        if self._index is None:
            self.load_or_create()

        if self._index.ntotal == 0:
            return [], []

        if query.ndim == 1:
            query = query.reshape(1, -1)

        query = np.ascontiguousarray(query, dtype=np.float32)
        actual_k = min(top_k, self._index.ntotal)

        scores, indices = self._index.search(query, actual_k)
        scores = scores[0].tolist()
        indices = indices[0].tolist()

        # Filter invalid indices (-1 from FAISS)
        valid = [(s, i) for s, i in zip(scores, indices) if i >= 0]
        if not valid:
            return [], []

        scores, indices = zip(*valid)
        return list(scores), list(indices)

    def get_meta(self, idx: int) -> Optional[Dict]:
        """Get metadata for a FAISS index position."""
        if 0 <= idx < len(self._meta):
            return self._meta[idx]
        return None

    def get_meta_batch(self, indices: List[int]) -> List[Optional[Dict]]:
        return [self.get_meta(i) for i in indices]

    def save(self) -> None:
        """Persist index to disk."""
        if self._index is None:
            return
        try:
            import faiss
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f)
            logger.info("FAISS index saved: %d vectors → %s",
                        self._index.ntotal, self.index_path)
        except Exception as e:
            logger.error("Failed to save FAISS index: %s", e)

    def remove_camera(self, camera_id: str) -> None:
        """
        Remove all embeddings belonging to a camera.
        Rebuilds the index (FAISS FlatIP doesn't support deletion).
        """
        import faiss

        if self._index is None or self._index.ntotal == 0:
            return

        keep_mask = [m.get("camera_id") != camera_id for m in self._meta]
        if all(keep_mask):
            return  # nothing to remove

        keep_indices = [i for i, k in enumerate(keep_mask) if k]
        if not keep_indices:
            self._index = faiss.IndexFlatIP(self.dim)
            self._meta = []
            return

        # Reconstruct vectors to keep
        all_vecs = np.zeros((self._index.ntotal, self.dim), dtype=np.float32)
        self._index.reconstruct_n(0, self._index.ntotal, all_vecs)

        keep_vecs = all_vecs[keep_indices]
        keep_meta = [self._meta[i] for i in keep_indices]

        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(keep_vecs)
        self._meta = keep_meta
        logger.info("FAISS: Removed camera %s, %d vectors remain",
                    camera_id, self._index.ntotal)

    def _check_disk_reload(self) -> None:
        """Reload index if disk file exists and has newer modification time or if index is empty."""
        if not self.index_path.exists() or not self.meta_path.exists():
            return
        try:
            mtime = self.index_path.stat().st_mtime
            if not hasattr(self, "_last_mtime") or self._last_mtime != mtime or self._index is None or self._index.ntotal == 0:
                self.load_or_create()
                self._last_mtime = mtime
        except Exception as e:
            logger.debug("Error checking FAISS disk reload: %s", e)

    @property
    def total_vectors(self) -> int:
        self._check_disk_reload()
        return self._index.ntotal if self._index is not None else 0

    def is_empty(self) -> bool:
        return self.total_vectors == 0


# Global singleton
_store: Optional[FAISSStore] = None


def get_store(reload: bool = False) -> FAISSStore:
    global _store
    if _store is None or reload:
        _store = FAISSStore()
        _store.load_or_create()
    else:
        _store._check_disk_reload()
    return _store


# Alias for backward compatibility and convenience
FaissStore = FAISSStore
