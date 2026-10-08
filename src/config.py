"""
src/config.py — Centralized configuration management.
Loads from config.yaml, overridden by .env variables.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Project root ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


def _load_yaml() -> dict:
    """Load raw YAML config."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    logger.warning("config.yaml not found — using defaults")
    return {}


_raw = _load_yaml()


def _get(key: str, default=None):
    """Get value from YAML or env override."""
    env_key = key.upper()
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return env_val
    return _raw.get(key, default)


# ─── Device ─────────────────────────────────────────────────────────────────
def resolve_device() -> str:
    """Resolve device: prefer CUDA, fallback to CPU."""
    import torch
    cfg_device = _get("device", "cuda")
    if cfg_device.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
        else:
            logger.warning("CUDA requested but not available — falling back to CPU.")
            return "cpu"
    return "cpu"


# ─── Config class ────────────────────────────────────────────────────────────
class Config:
    # Device
    device: str = "cpu"                     # resolved at runtime
    use_fp16: bool = bool(_get("use_fp16", True))

    # Video sampling
    sample_fps: float = float(_get("sample_fps", 2))
    min_fps: float = float(_get("min_fps", 1))
    max_fps: float = float(_get("max_fps", 5))

    # Detection
    yolo_model: str = str(_get("yolo_model", "yolov8n.pt"))
    confidence_threshold: float = float(_get("confidence_threshold", 0.25))
    image_size: int = int(_get("image_size", 640))
    detection_batch_size: int = int(_get("detection_batch_size", 1))
    target_classes: List[str] = _raw.get(
        "target_classes",
        ["person", "car", "bicycle", "motorcycle", "bus", "truck", "backpack", "handbag", "suitcase"],
    )

    # Tracking
    tracker_type: str = str(_get("tracker_type", "bytetrack"))
    max_disappeared: int = int(_get("max_disappeared", 30))

    # Crops
    max_crops_per_track: int = int(_get("max_crops_per_track", 5))
    min_crop_size: int = int(_get("min_crop_size", 32))
    crop_padding: int = int(_get("crop_padding", 10))

    # CLIP
    clip_model: str = str(_get("clip_model", "ViT-B-32"))
    clip_pretrained: str = str(_get("clip_pretrained", "openai"))
    clip_batch_size: int = int(_get("clip_batch_size", 32))
    embedding_dim: int = int(_get("embedding_dim", 512))

    # FAISS
    faiss_index_type: str = str(_get("faiss_index_type", "FlatIP"))
    top_k: int = int(_get("top_k", 20))
    rerank_top_k: int = int(_get("rerank_top_k", 10))

    # Query
    llm_provider: str = str(_get("llm_provider", "gemini"))
    query_timeout_seconds: int = int(_get("query_timeout_seconds", 30))

    # VLM
    use_vlm_verification: bool = str(_get("use_vlm_verification", False)).lower() in ("true", "1", "yes")
    vqa_top_k: int = int(_get("vqa_top_k", 3))

    # Paths
    data_dir: Path = ROOT_DIR / str(_get("data_dir", "data"))
    videos_dir: Path = ROOT_DIR / str(_get("videos_dir", "data/videos"))
    crops_dir: Path = ROOT_DIR / str(_get("crops_dir", "data/crops"))
    evidence_dir: Path = ROOT_DIR / str(_get("evidence_dir", "data/evidence"))
    clips_dir: Path = ROOT_DIR / str(_get("clips_dir", "data/evidence/clips"))
    index_dir: Path = ROOT_DIR / str(_get("index_dir", "data/index"))
    db_path: Path = ROOT_DIR / str(_get("db_path", "data/app.db"))
    faiss_index_path: Path = ROOT_DIR / str(_get("faiss_index_path", "data/index/faiss.index"))
    faiss_meta_path: Path = ROOT_DIR / str(_get("faiss_meta_path", "data/index/faiss_meta.json"))

    # Evidence clips
    evidence_clip_duration: int = int(_get("evidence_clip_duration", 10))
    evidence_clip_before: int = int(_get("evidence_clip_before", 5))
    evidence_clip_after: int = int(_get("evidence_clip_after", 5))

    # Re-ID
    enable_cross_camera_reid: bool = str(_get("enable_cross_camera_reid", False)).lower() in ("true", "1", "yes")
    reid_similarity_threshold: float = float(_get("reid_similarity_threshold", 0.75))
    reid_max_time_gap_seconds: int = int(_get("reid_max_time_gap_seconds", 300))

    # Evaluation
    evaluation_queries_path: Path = ROOT_DIR / str(_get("evaluation_queries_path", "evaluation/queries.json"))

    # API Keys
    groq_api_key: Optional[str] = os.environ.get("GROQ_API_KEY")
    groq_model: str = str(_get("groq_model", "qwen/qwen3.8-27b"))
    gemini_api_key: Optional[str] = os.environ.get("GEMINI_API_KEY")
    openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")

    def __init__(self):
        """Resolve device at init time."""
        try:
            self.device = resolve_device()
        except ImportError:
            self.device = "cpu"

        # Ensure directories exist
        for d in [self.data_dir, self.videos_dir, self.crops_dir,
                  self.evidence_dir, self.clips_dir, self.index_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Singleton
cfg = Config()


def get_settings() -> Config:
    """Return global Config singleton."""
    return cfg
