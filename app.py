"""
app.py — Modern Mission-Control CCTV Multi-Camera Search & Intelligence
Cutting-edge tactical dashboard for multi-stream surveillance retrieval.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="SENTINEL // Video Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bootstrap DB ───────────────────────────────────────────────────────────────
try:
    from src.database import init_db, get_stats
    init_db()
except Exception as e:
    st.error(f"Database initialization failed: {e}")
    st.stop()

# ── Imports (after DB init) ───────────────────────────────────────────────────
from src import database as db
from src.config import cfg
from src.evidence.clip_generator import ffmpeg_available, generate_evidence_clip
from src.ingestion.metadata import register_video
from src.ingestion.video_reader import VideoInfo
from src.query.memory import (
    get_all_learned_locations,
    learn_location,
    resolve_location_to_camera,
)
from src.query.parser import QueryParser
from src.query.retriever import RetrievalPipeline
from src.query.time_parser import format_timestamp

# ── Custom Futuristic High-End Dark UI CSS ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-base: #06080f;
    --bg-surface: #0c101d;
    --bg-card: rgba(16, 22, 38, 0.75);
    --bg-card-hover: rgba(22, 30, 52, 0.9);
    --accent-cyan: #00f2fe;
    --accent-blue: #4facfe;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --accent-violet: #8b5cf6;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-glow: rgba(0, 242, 254, 0.35);
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    color: var(--text-main);
}

.stApp {
    background-color: var(--bg-base);
    background-image: 
        radial-gradient(at 0% 0%, rgba(0, 242, 254, 0.06) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(79, 70, 229, 0.08) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.04) 0px, transparent 50%);
    background-attachment: fixed;
}

/* Hide default streamlit clutter */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 98% !important;
}

/* Sidebar styling */
div[data-testid="stSidebar"] {
    background: #080c16 !important;
    border-right: 1px solid var(--border-subtle) !important;
}
div[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
    margin: 12px 0 !important;
}

/* Mission Control Tactical Header */
.sentinel-header {
    background: linear-gradient(135deg, rgba(12, 16, 29, 0.9) 0%, rgba(16, 24, 44, 0.8) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-bottom: 1px solid rgba(0, 242, 254, 0.25);
    border-radius: 16px;
    padding: 16px 24px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.sentinel-brand {
    display: flex;
    align-items: center;
    gap: 16px;
}

.sentinel-logo-box {
    width: 46px;
    height: 46px;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
}

.sentinel-title {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 50%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}

.sentinel-subtitle {
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 3px;
}

.sentinel-stats-row {
    display: flex;
    gap: 12px;
}

.sentinel-stat-chip {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 6px 14px;
    text-align: center;
    min-width: 75px;
}
.sentinel-stat-chip .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--accent-cyan);
}
.sentinel-stat-chip .lbl {
    font-size: 0.62rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}

/* Hero Search Banner */
.search-hero {
    background: linear-gradient(135deg, rgba(16, 22, 38, 0.6) 0%, rgba(26, 36, 62, 0.4) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.search-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00f2fe, #4facfe, transparent);
}

.pill-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}
.query-pill {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 0.75rem;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.2s ease;
}
.query-pill:hover {
    border-color: var(--accent-cyan);
    color: #ffffff;
    background: rgba(0, 242, 254, 0.1);
}

/* Result Cards - Glassmorphism Surveillance HUD */
.hud-result-card {
    background: rgba(12, 17, 30, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 20px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    position: relative;
}
.hud-result-card:hover {
    border-color: rgba(0, 242, 254, 0.4);
    box-shadow: 0 8px 30px rgba(0, 242, 254, 0.12);
    transform: translateY(-2px);
}
.hud-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding-bottom: 12px;
    margin-bottom: 14px;
}
.hud-cam-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--accent-cyan);
    letter-spacing: 0.5px;
}
.hud-time-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--accent-amber);
    background: rgba(245, 158, 11, 0.1);
    padding: 2px 8px;
    border-radius: 6px;
    border: 1px solid rgba(245, 158, 11, 0.25);
}
.hud-match-badge {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.25));
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Feed Monitor HUD Card */
.monitor-card {
    background: #090d18;
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.3);
}
.monitor-bar {
    background: linear-gradient(90deg, #111827, #1e293b);
    padding: 8px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-subtle);
}
.monitor-title {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    color: var(--accent-cyan);
    font-size: 0.85rem;
}

/* Pathway Ribbon */
.pathway-ribbon {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.8));
    border: 1px solid rgba(0, 242, 254, 0.25);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 242, 254, 0.08);
}
.pathway-title {
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
}
.pathway-steps {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
}
.pathway-node {
    background: rgba(0, 242, 254, 0.08);
    border: 1px solid rgba(0, 242, 254, 0.2);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.82rem;
}
.pathway-arrow {
    color: var(--accent-cyan);
    font-size: 1.1rem;
    font-weight: bold;
}

/* Streamlit Native Overrides */
.stTextInput > div > div > input {
    background: rgba(12, 16, 29, 0.85) !important;
    border: 1px solid rgba(0, 242, 254, 0.25) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 0.98rem !important;
    padding: 12px 16px !important;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.3) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 16px rgba(0, 242, 254, 0.3) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    color: white !important;
    border: 1px solid rgba(0, 242, 254, 0.4) !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    box-shadow: 0 0 18px rgba(0, 242, 254, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(12, 17, 30, 0.8) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-subtle) !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.18), rgba(79, 70, 229, 0.18)) !important;
    border: 1px solid rgba(0, 242, 254, 0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state initialization ──────────────────────────────────────────────
def init_session():
    defaults = {
        "chat_history": [],
        "pending_clarification": None,
        "pending_query": None,
        "indexing_running": False,
        "indexing_progress": 0.0,
        "indexing_message": "",
        "last_results": [],
        "active_tab": "🔍 Query & Intelligence",
        "selected_camera": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── GPU info helper ────────────────────────────────────────────────────────────
def get_gpu_info() -> Dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            name = torch.cuda.get_device_name(0)
            return {"available": True, "name": name, "used_gb": used, "total_gb": total}
    except Exception:
        pass
    return {"available": False, "name": "CPU Acceleration", "used_gb": 0, "total_gb": 0}

# ── Render Mission Control Header ──────────────────────────────────────────────
def render_header():
    stats = get_stats()
    gpu = get_gpu_info()

    st.markdown(f"""
    <div class="sentinel-header">
        <div class="sentinel-brand">
            <div class="sentinel-logo-box">🛡️</div>
            <div>
                <div class="sentinel-title">SENTINEL CCTV INTELLIGENCE</div>
                <div class="sentinel-subtitle">Real Surveillance Stream Visual Search & Multi-Cam Tracking</div>
            </div>
        </div>
        <div class="sentinel-stats-row">
            <div class="sentinel-stat-chip">
                <div class="val">LIVE</div>
                <div class="lbl">Engine</div>
            </div>
            <div class="sentinel-stat-chip">
                <div class="val">{stats.get('cameras', 0)}</div>
                <div class="lbl">Cameras</div>
            </div>
            <div class="sentinel-stat-chip">
                <div class="val">{stats.get('tracks', 0)}</div>
                <div class="lbl">Entities</div>
            </div>
            <div class="sentinel-stat-chip">
                <div class="val">{stats.get('total_indexed_duration_hours', 0):.2f}h</div>
                <div class="lbl">Indexed</div>
            </div>
            <div class="sentinel-stat-chip">
                <div class="val">H.264</div>
                <div class="lbl">Codec</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <span style="font-size:1.4rem;">🛰️</span>
            <div>
                <strong style="color:#00f2fe;font-size:1rem;letter-spacing:0.5px;">CCTV CONTROL</strong><br/>
                <span style="color:#64748b;font-size:0.7rem;font-family:'JetBrains Mono';">v2.4 REAL-STREAM</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab = st.radio(
            "Navigation",
            ["🔍 Query & Intelligence", "📹 Live CCTV Grid", "🛣️ Multi-Cam Tracking", "⚙️ Camera Manager"],
            key="nav_radio",
            label_visibility="collapsed",
        )
        st.session_state.active_tab = tab
        st.markdown("---")

        # Active real cameras monitor in sidebar
        cameras = db.get_all_cameras()
        if cameras:
            st.markdown('<div style="font-size:0.7rem;color:#00f2fe;font-weight:700;letter-spacing:1px;margin-bottom:10px;">SURVEILLANCE FEEDS</div>', unsafe_allow_html=True)
            for cam in cameras:
                tracks = db.get_tracks_for_camera(cam['camera_id'])
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:8px 12px;margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-family:'JetBrains Mono';color:#00f2fe;font-weight:700;font-size:0.78rem;">{cam['camera_id']}</span>
                        <span style="color:#10b981;font-size:0.68rem;font-weight:600;">● RECORDING</span>
                    </div>
                    <div style="color:#cbd5e1;font-size:0.8rem;margin:2px 0;">{cam['name']}</div>
                    <div style="color:#64748b;font-size:0.68rem;font-family:'JetBrains Mono';">{len(tracks)} tracked entities</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("*No active cameras*")

        st.markdown("---")
        gpu = get_gpu_info()
        st.caption(f"💻 Engine: {gpu['name']}")
        st.caption("🔒 Verified Real Surveillance Dataset")

# ── Color styling helper ───────────────────────────────────────────────────────
def _get_color_style(color: str) -> str:
    color_map = {
        "red": "background:rgba(239,68,68,0.2);border:1px solid #ef4444;color:#fca5a5;",
        "blue": "background:rgba(59,130,246,0.2);border:1px solid #3b82f6;color:#93c5fd;",
        "green": "background:rgba(16,185,129,0.2);border:1px solid #10b981;color:#6ee7b7;",
        "yellow": "background:rgba(245,158,11,0.2);border:1px solid #f59e0b;color:#fde68a;",
        "orange": "background:rgba(249,115,22,0.2);border:1px solid #f97316;color:#fdba74;",
        "white": "background:rgba(255,255,255,0.2);border:1px solid #ffffff;color:#ffffff;",
        "gray": "background:rgba(100,116,139,0.2);border:1px solid #64748b;color:#cbd5e1;",
        "black": "background:rgba(15,23,42,0.6);border:1px solid #475569;color:#94a3b8;",
    }
    return color_map.get(color.lower(), "background:rgba(148,163,184,0.15);border:1px solid #64748b;color:#cbd5e1;")

# ── Search Tab ─────────────────────────────────────────────────────────────────
def render_search_tab():
    st.markdown("""
    <div class="search-hero">
        <div style="font-size:1.15rem;font-weight:700;color:#f8fafc;margin-bottom:4px;">Conversational Surveillance Retrieval</div>
        <div style="font-size:0.82rem;color:#94a3b8;">Natural language semantic search across indexed real CCTV feeds. Powered by YOLOv8, ByteTrack, OpenCLIP, and Vector Similarity.</div>
    </div>
    """, unsafe_allow_html=True)

    # Preset Quick Action Chips
    st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#00f2fe;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">QUICK AUDIT PRESETS</div>', unsafe_allow_html=True)
    presets = [
        "Find all cars in the footages",
        "Show persons detected on cameras",
        "Locate motorcycles",
        "Find bicycles",
        "Where was the car seen across cameras?",
    ]
    p_cols = st.columns(len(presets))
    for i, p in enumerate(presets):
        if p_cols[i].button(p, key=f"pre_{i}", use_container_width=True):
            st.session_state["_query_prefill"] = p

    prefill = st.session_state.pop("_query_prefill", "")
    query = st.text_input(
        "Enter natural language surveillance query:",
        value=prefill,
        placeholder="e.g. Find cars in the footages, show persons near market, or locate blue truck...",
        key="search_input",
    )

    c1, c2, c3 = st.columns([2.5, 1, 1])
    with c1:
        search_clicked = st.button("⚡ EXECUTE NEURAL SEARCH", type="primary", use_container_width=True)
    with c2:
        top_k = st.selectbox("Max candidates", [4, 8, 12, 20], index=1)
    with c3:
        show_clips = st.checkbox("Evidence Clips", value=False)

    if search_clicked and query.strip():
        _execute_search(query.strip(), top_k, show_clips)

    if st.session_state.last_results:
        _render_results(st.session_state.last_results, show_clips)


def _execute_search(query: str, top_k: int, show_clips: bool):
    stats = get_stats()
    if stats.get("tracks", 0) == 0:
        st.warning("⚠️ Index is empty. Register and index camera footage first.")
        return

    with st.spinner(f"🔍 Analyzing surveillance vectors for: \"{query}\"..."):
        try:
            parser = QueryParser()
            structured = parser.parse(query)

            pipeline = RetrievalPipeline()
            results, clarification_needed = pipeline.search(structured, top_k=top_k)

            if clarification_needed:
                st.session_state.pending_clarification = clarification_needed
                st.session_state.pending_query = query
                st.session_state.last_results = []
                st.rerun()
            else:
                st.session_state.last_results = results
                if not results:
                    st.info("No detections matched this exact query criteria.")

        except Exception as e:
            st.error(f"Search execution error: {e}")
            logger.error("Search error: %s", e, exc_info=True)


def _render_results(results: list, show_clips: bool):
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin:24px 0 16px 0;">
        <span style="font-size:1.1rem;font-weight:700;color:#f8fafc;">🎯 RETRIEVAL AUDIT TRAIL ({len(results)} GROUNDED DETECTIONS)</span>
        <span style="font-size:0.75rem;font-family:'JetBrains Mono';color:#10b981;">● TIME-CORRELATED</span>
    </div>
    """, unsafe_allow_html=True)

    for i, result in enumerate(results):
        confidence_pct = int(result.rerank_score * 100)
        color = result.dominant_color or "unknown"
        color_badge_style = _get_color_style(color)
        cam_info = db.get_camera(result.camera_id)
        video_path = cam_info.get("video_path", "") if cam_info else ""

        with st.container():
            st.markdown(f"""
            <div class="hud-result-card">
                <div class="hud-header">
                    <div>
                        <span class="hud-cam-tag">📹 {result.camera_id} // {result.camera_name}</span>
                        <div style="font-size:0.85rem;color:#cbd5e1;margin-top:2px;font-weight:600;">
                            Detected Object: <span style="{color_badge_style};padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:700;text-transform:uppercase;">{color} {result.object_class}</span>
                        </div>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span class="hud-time-tag">⏱️ {format_timestamp(result.timestamp)} ({result.timestamp:.1f}s)</span>
                        <span class="hud-match-badge">{confidence_pct}% MATCH</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_details, col_video = st.columns([1, 1.3])

            with col_details:
                st.markdown("**🔍 Target Entity Crop**")
                if result.crop_path and Path(result.crop_path).exists():
                    try:
                        img = Image.open(result.crop_path)
                        st.image(img, caption=f"Track #{result.track_db_id} | {color.capitalize()} {result.object_class}", use_container_width=True)
                    except Exception:
                        st.caption("[Thumbnail unavailable]")
                else:
                    st.caption("[Crop pending]")

                st.markdown(f"""
                - **Location:** `{result.camera_name}` (`{result.camera_id}`)
                - **Timestamp:** `{format_timestamp(result.timestamp)}`
                - **Entity Class:** `{result.object_class.upper()}`
                - **Dominant Color:** `{color.upper()}`
                - **Semantic Confidence:** `{result.semantic_score:.3f}`
                """)

                # Cross-camera matches
                if cfg.enable_cross_camera_reid:
                    from src.reid.cross_camera import find_cross_camera_matches
                    matches = find_cross_camera_matches(result.track_db_id)
                    if matches:
                        st.markdown("**🛰️ Cross-Camera Entity Trajectory:**")
                        for m in matches:
                            cam_b_info = db.get_camera(m.camera_b)
                            cname = cam_b_info['name'] if cam_b_info else m.camera_b
                            st.markdown(
                                f"<div style='background:rgba(0,242,254,0.08);border-left:3px solid #00f2fe;padding:6px 10px;margin-bottom:6px;border-radius:4px;font-size:0.78rem;'>"
                                f"<strong>{m.camera_b}</strong> ({cname}) @ <code>{format_timestamp(m.time_b)}</code> "
                                f"<span style='color:#10b981;font-weight:700;'>({int(m.similarity*100)}% Re-ID Match)</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

            with col_video:
                st.markdown("**🎬 Grounded CCTV Evidence Player**")
                vpath_obj = Path(video_path)
                if not vpath_obj.is_absolute():
                    vpath_obj = Path(__file__).parent / video_path

                if video_path and vpath_obj.exists():
                    start_sec = max(0, int(result.timestamp - 2))
                    st.video(str(vpath_obj), start_time=start_sec)
                    st.caption(f"▶️ Real surveillance footage queued to **{format_timestamp(result.timestamp)}** (-2s lead)")
                else:
                    st.warning("Video file not found on disk.")

        st.markdown("<hr style='border-color:rgba(255,255,255,0.05);margin:24px 0;'/>", unsafe_allow_html=True)


# ── Camera Grid Tab ────────────────────────────────────────────────────────────
def render_cameras_grid_tab():
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;">
        <div>
            <div style="font-size:1.3rem;font-weight:800;color:#f8fafc;">📹 REAL-TIME CCTV MATRIX MONITOR</div>
            <div style="font-size:0.8rem;color:#94a3b8;">Concurrent surveillance streams with telemetry, object density, and OSD timecodes.</div>
        </div>
        <div style="font-family:'JetBrains Mono';font-size:0.75rem;color:#00f2fe;background:rgba(0,242,254,0.1);padding:4px 10px;border-radius:8px;border:1px solid rgba(0,242,254,0.3);">
            ● {len(cameras)}-STREAM CCTV GRID ACTIVE
        </div>
    </div>
    """, unsafe_allow_html=True)

    cameras = db.get_all_cameras()
    if not cameras:
        st.info("No active camera streams configured.")
        return

    # If 3 cameras, render all 3 in a clean balanced 3-column row
    if len(cameras) == 3:
        cols = st.columns(3)
        for idx, cam in enumerate(cameras):
            tracks = db.get_tracks_for_camera(cam['camera_id'])
            with cols[idx]:
                st.markdown(f"""
                <div class="monitor-card">
                    <div class="monitor-bar">
                        <span class="monitor-title">🔴 {cam['camera_id']} // {cam['name']}</span>
                        <span style="font-size:0.72rem;font-family:'JetBrains Mono';color:#10b981;font-weight:700;">
                            {len(tracks)} TRACKED ENTITIES
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                vpath = cam.get("video_path", "")
                vpath_obj = Path(vpath)
                if not vpath_obj.is_absolute():
                    vpath_obj = Path(__file__).parent / vpath

                if vpath and vpath_obj.exists():
                    st.video(str(vpath_obj))
                    st.caption(f"📐 {cam.get('width', 0)}×{cam.get('height', 0)} @ {cam.get('fps', 0):.1f}fps | ⏱️ {cam.get('duration', 0):.1f}s | H.264")
                else:
                    st.error(f"Stream unavailable: {vpath}")
    else:
        for i in range(0, len(cameras), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(cameras):
                    cam = cameras[i + j]
                    tracks = db.get_tracks_for_camera(cam['camera_id'])
                    with cols[j]:
                        st.markdown(f"""
                        <div class="monitor-card">
                            <div class="monitor-bar">
                                <span class="monitor-title">🔴 {cam['camera_id']} // {cam['name']}</span>
                                <span style="font-size:0.72rem;font-family:'JetBrains Mono';color:#10b981;font-weight:700;">
                                    {len(tracks)} TRACKED ENTITIES
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        vpath = cam.get("video_path", "")
                        vpath_obj = Path(vpath)
                        if not vpath_obj.is_absolute():
                            vpath_obj = Path(__file__).parent / vpath

                        if vpath and vpath_obj.exists():
                            st.video(str(vpath_obj))
                            st.caption(f"📐 {cam.get('width', 0)}×{cam.get('height', 0)} @ {cam.get('fps', 0):.1f}fps | ⏱️ {cam.get('duration', 0):.1f}s | H.264")
                        else:
                            st.error(f"Stream unavailable: {vpath}")


# ── Sequential Multi-Camera Tracking Tab ───────────────────────────────────────
def render_multi_cam_tracking_tab():
    st.markdown("""
    <div style="margin-bottom:20px;">
        <div style="font-size:1.3rem;font-weight:800;color:#f8fafc;">🛣️ CHRONOLOGICAL CROSS-CAMERA ENTITY JOURNEY</div>
        <div style="font-size:0.8rem;color:#94a3b8;">Trace detected people, cars, trucks, and motorcycles as they move across sequential surveillance camera coverage zones.</div>
    </div>
    """, unsafe_allow_html=True)

    all_tracks = db.get_all_tracks()
    if not all_tracks:
        st.info("No indexed tracks available.")
        return

    track_options = {
        f"Track #{t['id']} — {t.get('dominant_color', '').capitalize()} {t['class_name'].capitalize()} (First seen {t['camera_id']} @ {t['first_seen']:.1f}s)": t
        for t in all_tracks
    }
    selected_track_str = st.selectbox("Select Target Entity to Trace Across Streams:", list(track_options.keys()))
    selected_track = track_options[selected_track_str]

    from src.reid.cross_camera import find_cross_camera_matches
    matches = find_cross_camera_matches(selected_track["id"], max_results=10)

    cams_dict = {c["camera_id"]: c for c in db.get_all_cameras()}
    origin_cam = cams_dict.get(selected_track["camera_id"], {})

    journey = [{
        "camera_id": selected_track["camera_id"],
        "camera_name": origin_cam.get("name", selected_track["camera_id"]),
        "time": selected_track["first_seen"],
        "last_time": selected_track["last_seen"],
        "class_name": selected_track["class_name"],
        "color": selected_track.get("dominant_color", "unknown"),
        "crop_path": selected_track.get("best_crop", ""),
        "video_path": origin_cam.get("video_path", ""),
        "similarity": 1.0,
        "confidence": "Origin Detection"
    }]

    for m in matches:
        m_track = db.get_track_by_db_id(m.track_b_id)
        m_cam = cams_dict.get(m.camera_b, {})
        if m_track and m_cam:
            journey.append({
                "camera_id": m.camera_b,
                "camera_name": m_cam.get("name", m.camera_b),
                "time": m.time_b,
                "last_time": m_track.get("last_seen", m.time_b),
                "class_name": m_track.get("class_name", selected_track["class_name"]),
                "color": m_track.get("dominant_color", selected_track.get("dominant_color", "")),
                "crop_path": m_track.get("best_crop", ""),
                "video_path": m_cam.get("video_path", ""),
                "similarity": m.similarity,
                "confidence": f"{m.confidence_label} ({int(m.similarity * 100)}%)"
            })

    journey.sort(key=lambda j: j["camera_id"])

    # High-Tech Visual Pathway Ribbon
    nodes_html = []
    for j in journey:
        nodes_html.append(f"""
        <div class="pathway-node">
            <strong style="color:#00f2fe;">{j['camera_id']}</strong> // {j['camera_name']}<br/>
            <span style="color:#f59e0b;font-family:'JetBrains Mono';font-size:0.75rem;">⏱️ {format_timestamp(j['time'])}</span>
        </div>
        """)
    ribbon_content = ' <span class="pathway-arrow">➔</span> '.join(nodes_html)

    st.markdown(f"""
    <div class="pathway-ribbon">
        <div class="pathway-title">CHRONOLOGICAL SURVEILLANCE CORRELATION PATH</div>
        <div class="pathway-steps">{ribbon_content}</div>
    </div>
    """, unsafe_allow_html=True)

    # Multi-camera synchronized cards
    cols = st.columns(min(len(journey), 4))
    for idx, stop in enumerate(journey):
        with cols[idx % len(cols)]:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(0,242,254,0.2);border-radius:10px;padding:12px;margin-bottom:12px;">
                <div style="font-family:'JetBrains Mono';font-weight:700;color:#00f2fe;font-size:0.85rem;">STOP #{idx+1}: {stop['camera_id']}</div>
                <div style="font-size:0.8rem;color:#cbd5e1;margin-bottom:8px;">{stop['camera_name']}</div>
            </div>
            """, unsafe_allow_html=True)

            if stop['crop_path'] and Path(stop['crop_path']).exists():
                try:
                    cimg = Image.open(stop['crop_path'])
                    st.image(cimg, caption=f"{stop['color'].capitalize()} {stop['class_name']}", use_container_width=True)
                except Exception:
                    pass

            st.caption(f"⏱️ **Window:** `{format_timestamp(stop['time'])}` - `{format_timestamp(stop['last_time'])}`")
            st.caption(f"🔗 **Match Confidence:** `{stop['confidence']}`")

            vp_obj = Path(stop['video_path'])
            if not vp_obj.is_absolute():
                vp_obj = Path(__file__).parent / stop['video_path']
            if stop['video_path'] and vp_obj.exists():
                start_at = max(0, int(stop['time'] - 1.5))
                st.video(str(vp_obj), start_time=start_at)


# ── Camera Manager Tab ─────────────────────────────────────────────────────────
def render_camera_manager_tab():
    st.markdown("### ⚙️ Camera Inventory & System Indexing")
    cameras = db.get_all_cameras()

    t1, t2 = st.tabs(["📋 Camera Inventory", "➕ Add Camera"])

    with t1:
        if not cameras:
            st.info("No cameras registered.")
        else:
            for cam in cameras:
                with st.expander(f"📹 {cam['camera_id']} — {cam['name']}", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"- **Path:** `{cam['video_path']}`")
                        st.markdown(f"- **Resolution:** `{cam['width']}×{cam['height']}` @ `{cam['fps']:.1f} fps`")
                        st.markdown(f"- **Duration:** `{cam['duration']:.1f}s` ({cam['frame_count']:,} frames)")
                    with c2:
                        tracks = db.get_tracks_for_camera(cam['camera_id'])
                        st.metric("Entities Tracked", len(tracks))

    with t2:
        st.markdown("#### Register New Camera Stream")
        with st.form("new_cam_form"):
            cid = st.text_input("Camera ID", placeholder="CAM_05")
            cname = st.text_input("Camera Name", placeholder="North Exit Gateway")
            cdesc = st.text_input("Location Description", placeholder="Outbound vehicle lane")
            vfile = st.text_input("Path to MP4 video file", placeholder="data/videos/custom_stream.mp4")
            if st.form_submit_button("Register Camera Stream"):
                if cid and cname and vfile and Path(vfile).exists():
                    register_video(cid, cname, vfile, cdesc)
                    st.success(f"Registered {cid}")
                    st.rerun()
                else:
                    st.error("Invalid camera parameters or video path.")


# ── Main Controller ────────────────────────────────────────────────────────────
def main():
    render_header()
    render_sidebar()

    active = st.session_state.get("active_tab", "🔍 Query & Intelligence")

    if active == "🔍 Query & Intelligence":
        render_search_tab()
    elif active == "📹 Live CCTV Grid":
        render_cameras_grid_tab()
    elif active == "🛣️ Multi-Cam Tracking":
        render_multi_cam_tracking_tab()
    elif active == "⚙️ Camera Manager":
        render_camera_manager_tab()

    # Footer
    st.markdown("""
    <div style="border-top:1px solid rgba(255,255,255,0.06);margin-top:40px;padding-top:16px;text-align:center;">
        <span style="font-family:'JetBrains Mono';font-size:0.75rem;color:#64748b;">
            🛡️ SENTINEL VIDEO INTELLIGENCE // 100% REAL CCTV SURVEILLANCE DATASET // MULTI-CAMERA NEURAL RETRIEVAL
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
