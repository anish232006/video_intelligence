"""
app.py — Multi-Stream Video Intelligence Dashboard
Streamlit UI for the CCTV conversational query system.
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
    page_title="Video Intelligence — Multi-Camera CCTV Search",
    page_icon="🎥",
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

# ── Custom CSS (dark futuristic theme) ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: #1a2035;
    --bg-card-hover: #1e2540;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --accent-purple: #8b5cf6;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: #1e3a5f;
    --border-active: #3b82f6;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 50%, #0a0e1a 100%);
}

/* Header */
.vi-header {
    background: linear-gradient(90deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border-bottom: 1px solid var(--border-active);
    padding: 20px 30px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.vi-title {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin: 0;
}

.vi-subtitle {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-top: 2px;
}

.vi-status-grid {
    display: flex;
    gap: 16px;
}

.vi-status-item {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 8px;
    padding: 8px 14px;
    text-align: center;
    min-width: 90px;
}

.vi-status-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent-cyan);
}

.vi-status-label {
    font-size: 0.65rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Camera cards in sidebar */
.cam-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.cam-card:hover {
    border-color: var(--accent-blue);
    background: var(--bg-card-hover);
}
.cam-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent-cyan);
    font-weight: 500;
}
.cam-name {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-primary);
}
.cam-indexed {
    font-size: 0.7rem;
    color: var(--success);
}
.cam-not-indexed {
    font-size: 0.7rem;
    color: var(--text-secondary);
}

/* Result cards */
.result-card {
    background: linear-gradient(135deg, #1a2035, #1e2540);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
.result-card:hover {
    border-color: #3b82f6;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 12px;
}

.result-camera {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent-cyan);
    font-weight: 500;
}
.result-timestamp {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-blue);
}
.result-object {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-primary);
    text-transform: capitalize;
}
.result-confidence {
    background: linear-gradient(135deg, #065f46, #047857);
    border: 1px solid #10b981;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #6ee7b7;
}
.result-explanation {
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-style: italic;
    margin-top: 8px;
    padding: 8px;
    background: rgba(255,255,255,0.03);
    border-radius: 6px;
    border-left: 2px solid var(--accent-blue);
}
.color-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: capitalize;
    margin-right: 4px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* Chat area */
.chat-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

/* Sidebar sections */
.sidebar-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 14px;
}
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* Knowledge badges */
.knowledge-badge {
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.4);
    border-radius: 8px;
    padding: 6px 10px;
    margin-bottom: 6px;
    font-size: 0.75rem;
}

.knowledge-term {
    color: var(--accent-purple);
    font-family: 'JetBrains Mono', monospace;
}

.knowledge-arrow {
    color: var(--text-secondary);
    margin: 0 6px;
}

.knowledge-cam {
    color: var(--accent-cyan);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

/* Query box */
.stTextInput > div > div > input {
    background: #0f172a !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-size: 1rem !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    color: white !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    transform: translateY(-1px) !important;
}

/* Alert box for clarification */
.clarify-box {
    background: linear-gradient(135deg, #1c1917, #292524);
    border: 1px solid #f59e0b;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
}
.clarify-title {
    color: #f59e0b;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 8px;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #06b6d4) !important;
}

/* Metrics */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Hide default streamlit header */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; }

div[data-testid="stSidebar"] {
    background: #0d1526 !important;
    border-right: 1px solid #1e3a5f !important;
}

.stSelectbox > div > div {
    background: #0f172a !important;
    border-color: #1e3a5f !important;
    color: #f1f5f9 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-blue) !important;
    background: rgba(59, 130, 246, 0.15) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state initialization ──────────────────────────────────────────────
def init_session():
    defaults = {
        "chat_history": [],
        "pending_clarification": None,     # Location term needing clarification
        "pending_query": None,             # Original query awaiting clarification
        "indexing_running": False,
        "indexing_progress": 0.0,
        "indexing_message": "",
        "last_results": [],
        "active_tab": "search",
        "selected_camera": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── GPU info helper ────────────────────────────────────────────────────────────
def get_gpu_info() -> Dict[str, Any]:
    """Get GPU memory stats. Never crashes."""
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            name = torch.cuda.get_device_name(0)
            return {"available": True, "name": name, "used_gb": used, "total_gb": total}
    except Exception:
        pass
    return {"available": False, "name": "CPU", "used_gb": 0, "total_gb": 0}


# ── Render header ─────────────────────────────────────────────────────────────
def render_header():
    stats = get_stats()
    gpu = get_gpu_info()

    st.markdown(f"""
    <div class="vi-header">
        <div>
            <div class="vi-title">🎥 VIDEO INTELLIGENCE</div>
            <div class="vi-subtitle">Multi-Camera Conversational CCTV Search</div>
        </div>
        <div class="vi-status-grid">
            <div class="vi-status-item">
                <div class="vi-status-value">{'🟢' if gpu['available'] else '🔵'}</div>
                <div class="vi-status-label">{'GPU' if gpu['available'] else 'CPU'}</div>
            </div>
            <div class="vi-status-item">
                <div class="vi-status-value">{stats.get('cameras', 0)}</div>
                <div class="vi-status-label">Cameras</div>
            </div>
            <div class="vi-status-item">
                <div class="vi-status-value">{stats.get('tracks', 0)}</div>
                <div class="vi-status-label">Tracks</div>
            </div>
            <div class="vi-status-item">
                <div class="vi-status-value">{stats.get('total_indexed_duration_hours', 0):.1f}h</div>
                <div class="vi-status-label">Indexed</div>
            </div>
            <div class="vi-status-item">
                <div class="vi-status-value">{stats.get('knowledge_entries', 0)}</div>
                <div class="vi-status-label">Locations</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🎥 CCTV Intelligence")
        st.markdown("---")

        # Navigation
        tab = st.radio(
            "Navigation",
            ["🔍 Search", "📹 Footage & Cameras", "⚙️ Index", "📊 Evaluate"],
            key="nav_radio",
            label_visibility="collapsed",
        )
        st.session_state.active_tab = tab
        st.markdown("---")

        # Camera list
        cameras = db.get_all_cameras()
        if cameras:
            st.markdown('<div class="sidebar-section-title">CAMERAS</div>', unsafe_allow_html=True)
            for cam in cameras:
                indexed = cam.get("indexed", 0)
                status = "🟢 Indexed" if indexed else "⭕ Not indexed"
                status_class = "cam-indexed" if indexed else "cam-not-indexed"
                st.markdown(f"""
                <div class="cam-card">
                    <div class="cam-id">{cam['camera_id']}</div>
                    <div class="cam-name">{cam['name']}</div>
                    <div class="{status_class}">{status}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("*No cameras registered yet*")

        st.markdown("---")

        # Learned locations
        locations = get_all_learned_locations()
        if locations:
            st.markdown('<div class="sidebar-section-title">LEARNED LOCATIONS</div>', unsafe_allow_html=True)
            for loc in locations:
                st.markdown(f"""
                <div class="knowledge-badge">
                    <span class="knowledge-term">{loc['term']}</span>
                    <span class="knowledge-arrow">→</span>
                    <span class="knowledge-cam">{loc.get('camera_id', '?')}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # GPU stats
        gpu = get_gpu_info()
        if gpu["available"]:
            st.markdown(f"""
            **🖥️ {gpu['name']}**
            VRAM: {gpu['used_gb']:.1f} / {gpu['total_gb']:.1f} GB
            """)
        else:
            st.markdown("🔵 **Running on CPU**")


# ── Search tab ────────────────────────────────────────────────────────────────
def render_search_tab():
    st.markdown("### 🔍 Conversational Video Search")

    # Example queries
    with st.expander("💡 Example queries", expanded=False):
        examples = [
            "Did a red car pass through the main gate in the last hour?",
            "Show me people near the entrance.",
            "Find the white car on camera 2.",
            "Did someone carrying a backpack enter?",
            "Where was the red shirt person seen?",
            "Show all vehicles detected today.",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i % 2].button(f"📌 {ex[:50]}...", key=f"ex_{i}"):
                st.session_state["_query_prefill"] = ex

    # Query input
    prefill = st.session_state.pop("_query_prefill", "")
    query = st.text_input(
        "Ask anything about your cameras...",
        value=prefill,
        placeholder="e.g. Did a red car pass through the main gate?",
        key="search_input",
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        top_k = st.selectbox("Results", [5, 10, 20], key="top_k_select")
    with col3:
        show_clips = st.checkbox("Auto-generate clips", value=False)

    # ── Handle clarification response ─────────────────────────────────────────
    if st.session_state.pending_clarification:
        term = st.session_state.pending_clarification
        original_query = st.session_state.pending_query

        st.markdown(f"""
        <div class="clarify-box">
            <div class="clarify-title">⚠️ Location Clarification Needed</div>
            I don't know which camera represents <strong>"{term}"</strong>.
            Please select the camera for this location.
        </div>
        """, unsafe_allow_html=True)

        cameras = db.get_all_cameras()
        if cameras:
            cam_options = {f"{c['camera_id']} — {c['name']}": c["camera_id"] for c in cameras}
            selected = st.selectbox(
                f"Which camera is '{term}'?",
                options=list(cam_options.keys()),
                key="clarification_camera",
            )
            alias_input = st.text_input(
                "Add aliases (comma-separated, optional):",
                placeholder="front gate, entrance gate",
                key="clarification_aliases",
            )

            col_learn, col_skip = st.columns(2)
            with col_learn:
                if st.button("✅ Learn & Search", type="primary"):
                    cam_id = cam_options[selected]
                    aliases = [a.strip() for a in alias_input.split(",") if a.strip()]
                    learn_location(term, cam_id, aliases=aliases)
                    st.session_state.pending_clarification = None

                    # Re-run original query
                    st.success(f"✅ Learned: '{term}' → {cam_id}")
                    if original_query:
                        _execute_search(original_query, top_k, show_clips)
                    st.session_state.pending_query = None
                    st.rerun()

            with col_skip:
                if st.button("⏭️ Search all cameras"):
                    st.session_state.pending_clarification = None
                    st.session_state.pending_query = None
                    _execute_search(query or original_query, top_k, show_clips)
        else:
            st.warning("No cameras registered. Add cameras first.")
            if st.button("Clear"):
                st.session_state.pending_clarification = None
                st.session_state.pending_query = None

    # ── Execute search ────────────────────────────────────────────────────────
    elif search_clicked and query.strip():
        _execute_search(query.strip(), top_k, show_clips)

    # ── Show results ──────────────────────────────────────────────────────────
    if st.session_state.last_results:
        _render_results(st.session_state.last_results, show_clips)


def _execute_search(query: str, top_k: int, show_clips: bool):
    """Run the retrieval pipeline and update session state."""
    stats = get_stats()
    if stats.get("tracks", 0) == 0:
        st.warning("⚠️ Index is empty. Add and index a video first.")
        return

    with st.spinner(f"🔍 Searching: *{query}*"):
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
                    st.info("No matches found. Try a different query or check that videos are indexed.")

        except Exception as e:
            st.error(f"Search error: {e}")
            logger.error("Search error: %s", e, exc_info=True)


def _render_results(results: list, show_clips: bool):
    st.markdown(f"### 📋 Results — {len(results)} match{'es' if len(results) != 1 else ''} found")

    for i, result in enumerate(results):
        confidence_pct = int(result.rerank_score * 100)
        color = result.dominant_color or "unknown"
        color_rgb = _get_color_style(color)
        cam_info = db.get_camera(result.camera_id)
        video_path = cam_info.get("video_path", "") if cam_info else ""

        with st.container():
            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">
                    <div>
                        <div class="result-camera">📹 {result.camera_id} — {result.camera_name}</div>
                        <div class="result-timestamp">⏱️ {format_timestamp(result.timestamp)}</div>
                        <div class="result-object">
                            <span class="color-badge" style="{color_rgb}">{color}</span>
                            {result.object_class}
                        </div>
                    </div>
                    <div class="result-confidence">{confidence_pct}% match</div>
                </div>
                <div class="result-explanation">💡 {result.explanation}</div>
            </div>
            """, unsafe_allow_html=True)

            # Two-column Evidence View: Crop & Meta on Left, Footage Player on Right
            col_crop, col_footage = st.columns([1, 1.4])

            with col_crop:
                st.markdown("**🔍 Target Detection**")
                if result.crop_path and Path(result.crop_path).exists():
                    try:
                        img = Image.open(result.crop_path)
                        st.image(img, caption=f"Crop: Track #{result.track_db_id} ({color} {result.object_class})", use_container_width=True)
                    except Exception:
                        st.markdown("*[Thumbnail unavailable]*")
                else:
                    st.markdown("*[No thumbnail]*")

                st.markdown(f"- **Camera:** `{result.camera_id}` ({result.camera_name})")
                st.markdown(f"- **Event Time:** `{format_timestamp(result.timestamp)}` ({result.timestamp:.1f}s)")
                st.markdown(f"- **Object Class:** `{result.object_class.capitalize()}`")
                st.markdown(f"- **Dominant Color:** `{color.capitalize()}`")
                st.markdown(f"- **CLIP Semantic Match:** `{result.semantic_score:.3f}`")
                st.markdown(f"- **Ranked Score:** `{result.rerank_score:.3f}`")

                # Cross-camera ReID
                if cfg.enable_cross_camera_reid:
                    from src.reid.cross_camera import find_cross_camera_matches
                    matches = find_cross_camera_matches(result.track_db_id)
                    if matches:
                        st.markdown("**🛰️ Re-Identified Across Other Cameras:**")
                        for m in matches:
                            cam_b_info = db.get_camera(m.camera_b)
                            cname = cam_b_info['name'] if cam_b_info else m.camera_b
                            st.markdown(
                                f"<div style='background:rgba(59,130,246,0.12);border-left:3px solid #3b82f6;padding:6px 10px;margin-bottom:6px;border-radius:4px;font-size:0.8rem;'>"
                                f"<strong>{m.camera_b}</strong> — {cname}<br/>"
                                f"⏱️ <code>{format_timestamp(m.time_b)}</code> ({m.time_b:.1f}s) &nbsp;|&nbsp; "
                                f"<span style='color:#10b981;font-weight:600;'>{m.confidence_label} Match ({int(m.similarity*100)}%)</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                    else:
                        st.caption("ℹ️ No cross-camera matches found for this track.")

            with col_footage:
                st.markdown("**🎬 Evidence Footage**")
                if video_path and Path(video_path).exists():
                    start_sec = max(0, int(result.timestamp - 2))
                    st.video(video_path, start_time=start_sec)
                    st.caption(f"▶️ CCTV footage queued to **{format_timestamp(result.timestamp)}** (playhead -2s offset)")

                    # Isolated 10s Clip Option
                    with st.expander("✂️ Isolated 10s Clip Generator", expanded=False):
                        if st.button(f"Generate ±5s Evidence Clip", key=f"clip_btn_{i}_{result.track_db_id}"):
                            _generate_and_show_clip(result)
                else:
                    st.warning("Original video file not found on disk.")

        st.markdown("---")


def _generate_and_show_clip(result):
    """Generate and display an isolated evidence clip."""
    cam_info = db.get_camera(result.camera_id)
    if not cam_info:
        st.error("Camera not found in database.")
        return

    video_path = cam_info.get("video_path", "")
    if not Path(video_path).exists():
        st.error(f"Video file not found: {video_path}")
        return

    with st.spinner("Generating 10s evidence clip..."):
        clip_path = generate_evidence_clip(
            video_path=video_path,
            timestamp=result.timestamp,
            track_db_id=result.track_db_id,
            camera_id=result.camera_id,
        )

    if clip_path and Path(clip_path).exists():
        st.video(clip_path)
        st.success(f"Isolated clip ready: ±{cfg.evidence_clip_before}s around {format_timestamp(result.timestamp)}")
    else:
        st.info("Isolated clip generation not supported on this host without FFmpeg. Playing source video above.")


def _get_color_style(color: str) -> str:
    """Return CSS style string for a color badge."""
    color_styles = {
        "red": "background:rgba(220,50,50,0.2);border-color:rgba(220,50,50,0.6);color:#fca5a5;",
        "blue": "background:rgba(59,130,246,0.2);border-color:rgba(59,130,246,0.6);color:#93c5fd;",
        "green": "background:rgba(34,197,94,0.2);border-color:rgba(34,197,94,0.6);color:#86efac;",
        "white": "background:rgba(255,255,255,0.15);border-color:rgba(255,255,255,0.4);color:#f8fafc;",
        "black": "background:rgba(30,30,30,0.4);border-color:rgba(100,100,100,0.5);color:#cbd5e1;",
        "gray": "background:rgba(100,116,139,0.2);border-color:rgba(100,116,139,0.5);color:#cbd5e1;",
        "yellow": "background:rgba(234,179,8,0.2);border-color:rgba(234,179,8,0.6);color:#fde047;",
        "orange": "background:rgba(249,115,22,0.2);border-color:rgba(249,115,22,0.6);color:#fdba74;",
        "brown": "background:rgba(120,53,15,0.3);border-color:rgba(120,53,15,0.6);color:#d97706;",
        "purple": "background:rgba(139,92,246,0.2);border-color:rgba(139,92,246,0.6);color:#c4b5fd;",
    }
    return color_styles.get(color.lower(), "background:rgba(100,116,139,0.15);color:#94a3b8;")


# ── Camera management tab ─────────────────────────────────────────────────────
# ── Camera management tab ─────────────────────────────────────────────────────
def _render_camera_feed_card(cam: dict, expanded: bool = False):
    vpath = cam.get("video_path", "")
    indexed = cam.get("indexed", 0)
    tracks = db.get_tracks_for_camera(cam["camera_id"])
    track_count = len(tracks)

    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px 10px 0 0;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;">
        <div>
            <span style="font-family:'JetBrains Mono';font-weight:700;color:#06b6d4;">📹 {cam['camera_id']}</span>
            <span style="color:#f1f5f9;margin-left:8px;font-weight:600;">{cam['name']}</span>
        </div>
        <div>
            <span style="background:{'#065f46' if indexed else '#374151'};color:{'#6ee7b7' if indexed else '#9ca3af'};padding:3px 8px;border-radius:12px;font-size:0.75rem;font-weight:600;">
                {'🟢 Indexed' if indexed else '⭕ Not indexed'} ({track_count} tracks)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if vpath and Path(vpath).exists():
        st.video(vpath)
        st.caption(f"📁 `{Path(vpath).name}` | 📐 {cam.get('width', 0)}×{cam.get('height', 0)} @ {cam.get('fps', 0):.1f}fps | ⏱️ {cam.get('duration', 0):.1f}s ({cam.get('frame_count', 0):,} frames)")
    else:
        st.error(f"Video file missing: {vpath}")


def render_cameras_tab():
    st.markdown("### 📹 CCTV Footage & Camera Feeds")

    tab_watch, tab_sequence, tab_add, tab_manage = st.tabs([
        "📺 Live / Uploaded Footage",
        "🛣️ Sequential Multi-Cam Tracking",
        "➕ Upload & Add Camera",
        "📋 Camera Inventory & Stats"
    ])

    with tab_watch:
        cameras = db.get_all_cameras()
        if not cameras:
            st.info("No cameras registered yet. Upload or register videos in the 'Upload & Add Camera' tab.")
        else:
            col_ctrl1, col_ctrl2 = st.columns([2, 1])
            with col_ctrl1:
                st.markdown(f"#### 🛰️ Surveillance Feeds Monitor ({len(cameras)} active camera{'s' if len(cameras) != 1 else ''})")
            with col_ctrl2:
                layout_mode = st.radio("Display Layout", ["Grid View (Multi-Cam Wall)", "Single Camera Focus"], horizontal=True, label_visibility="collapsed")

            if layout_mode == "Grid View (Multi-Cam Wall)":
                num_cols = 2
                for i in range(0, len(cameras), num_cols):
                    cols = st.columns(num_cols)
                    for j in range(num_cols):
                        if i + j < len(cameras):
                            cam = cameras[i + j]
                            with cols[j]:
                                _render_camera_feed_card(cam)
            else:
                cam_choices = {f"{c['camera_id']} — {c['name']}": c for c in cameras}
                selected_cam_key = st.selectbox("Select Camera Stream to View:", list(cam_choices.keys()))
                if selected_cam_key:
                    _render_camera_feed_card(cam_choices[selected_cam_key], expanded=True)

    with tab_sequence:
        st.markdown("#### 🛣️ Sequential Multi-Camera Vehicle Tracking Journey")
        st.caption("Traces a target entity chronologically as it traverses through successive camera zones (Cam A ➔ Cam B ➔ Cam C ➔ Cam D).")

        all_tracks = db.get_all_tracks()
        if not all_tracks:
            st.info("No indexed tracks available. Index the cameras first.")
        else:
            track_options = {
                f"Track #{t['id']} — {t.get('dominant_color', '').capitalize()} {t['class_name'].capitalize()} (Origin: {t['camera_id']} @ {t['first_seen']:.1f}s)": t
                for t in all_tracks
            }
            selected_track_str = st.selectbox("Select Target Entity to Track Across Streams:", list(track_options.keys()))
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

            # Sort journey chronologically by camera_id
            journey.sort(key=lambda j: j["camera_id"])

            # Render visual pathway banner
            timeline_items = []
            for j in journey:
                timeline_items.append(
                    f"<span style='color:#60a5fa;font-weight:700;'>{j['camera_id']}</span> "
                    f"<span style='color:#94a3b8;'>({j['camera_name']})</span> "
                    f"<code>{format_timestamp(j['time'])}</code>"
                )
            timeline_html = " <span style='color:#10b981;font-size:1.1rem;margin:0 8px;'>➔</span> ".join(timeline_items)
            st.markdown(
                f"<div style='background:rgba(15,23,42,0.85);border:1px solid #1e3a5f;border-radius:10px;padding:14px 18px;margin-bottom:20px;'>"
                f"<div style='font-size:0.75rem;color:#06b6d4;font-weight:600;margin-bottom:6px;text-transform:uppercase;'>Chained Cross-Camera Pathway</div>"
                f"<div style='font-size:0.95rem;display:flex;flex-wrap:wrap;align-items:center;'>{timeline_html}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Render 4 camera cards with synchronized video players
            cols = st.columns(min(len(journey), 4))
            for idx, stop in enumerate(journey):
                col = cols[idx % len(cols)]
                with col:
                    st.markdown(f"**Step {idx+1}: {stop['camera_id']}**")
                    st.caption(f"{stop['camera_name']}")
                    if stop['crop_path'] and Path(stop['crop_path']).exists():
                        try:
                            cimg = Image.open(stop['crop_path'])
                            st.image(cimg, caption=f"Tracked: {stop['color'].capitalize()} {stop['class_name']}", use_container_width=True)
                        except Exception:
                            pass
                    st.markdown(f"⏱️ **Passage:** `{format_timestamp(stop['time'])}` - `{format_timestamp(stop['last_time'])}`")
                    st.markdown(f"🔗 **Match:** `{stop['confidence']}`")
                    if stop['video_path'] and Path(stop['video_path']).exists():
                        start_at = max(0, int(stop['time'] - 1.5))
                        st.video(stop['video_path'], start_time=start_at)
                        st.caption(f"▶️ Queued @ {format_timestamp(stop['time'])}")

    with tab_add:
        st.markdown("#### ➕ Register a new camera video")
        with st.form("add_camera_form"):
            col1, col2 = st.columns(2)
            with col1:
                cam_id = st.text_input("Camera ID", placeholder="CAM_04", key="new_cam_id")
                cam_name = st.text_input("Camera Name", placeholder="North Entrance", key="new_cam_name")
            with col2:
                cam_desc = st.text_input("Location Description", placeholder="Main vehicle driveway", key="new_cam_desc")
                uploaded = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"], key="cam_video_upload")
                video_path = st.text_input("OR enter video path", placeholder="/path/to/video.mp4", key="new_cam_path")

            submitted = st.form_submit_button("➕ Register Camera", type="primary")

            if submitted:
                if not cam_id or not cam_name:
                    st.error("Camera ID and Name are required.")
                else:
                    final_path = ""
                    if uploaded:
                        save_dir = cfg.videos_dir
                        save_dir.mkdir(parents=True, exist_ok=True)
                        final_path = str(save_dir / uploaded.name)
                        with open(final_path, "wb") as f:
                            f.write(uploaded.read())
                        st.success(f"Video saved to: {final_path}")
                    elif video_path:
                        final_path = video_path

                    if not final_path:
                        st.error("Please provide a video file.")
                    elif not Path(final_path).exists():
                        st.error(f"Video file not found: {final_path}")
                    else:
                        info = register_video(cam_id, cam_name, final_path, cam_desc)
                        if info:
                            st.success(f"✅ Camera {cam_id} registered: {info}")
                            st.rerun()
                        else:
                            st.error("Failed to read video metadata. Is the file valid?")

        # Auto-detect videos in data/videos/
        st.markdown("#### 📂 Auto-detect videos in `data/videos/`")
        video_files = list(cfg.videos_dir.glob("*.mp4")) + \
                      list(cfg.videos_dir.glob("*.avi")) + \
                      list(cfg.videos_dir.glob("*.mov"))

        if video_files:
            st.info(f"Found {len(video_files)} video file(s) in `{cfg.videos_dir}`")
            for vf in video_files:
                stem = vf.stem
                cam_id_guess = stem.upper().replace("-", "_").replace(" ", "_")
                if not cam_id_guess.startswith("CAM"):
                    cam_id_guess = f"CAM_{cam_id_guess}"

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"📹 `{vf.name}` → Suggestion ID: `{cam_id_guess}`")
                with col2:
                    if st.button(f"Register", key=f"reg_{vf.name}"):
                        info = register_video(cam_id_guess, cam_id_guess, str(vf))
                        if info:
                            st.success(f"Registered {cam_id_guess}")
                            st.rerun()
        else:
            st.markdown("""
            > 📁 Place your MP4 files in `data/videos/` and they will appear here.
            """)

    with tab_manage:
        cameras = db.get_all_cameras()
        if not cameras:
            st.info("No cameras registered yet.")
            return

        for cam in cameras:
            with st.expander(f"📹 {cam['camera_id']} — {cam['name']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Path:** `{cam['video_path']}`")
                    st.markdown(f"**Resolution:** {cam['width']}×{cam['height']}")
                    st.markdown(f"**FPS:** {cam['fps']:.1f}")
                    st.markdown(f"**Duration:** {cam['duration']:.1f}s ({cam['duration']/60:.1f} min)")
                    st.markdown(f"**Frames:** {cam['frame_count']:,}")
                    st.markdown(f"**Indexed:** {'✅ Yes' if cam['indexed'] else '❌ No'}")

                with col2:
                    tracks = db.get_tracks_for_camera(cam['camera_id'])
                    st.metric("Tracks indexed", len(tracks))

                    if st.button(f"🗑️ Remove Camera", key=f"del_{cam['camera_id']}"):
                        db.delete_camera(cam['camera_id'])
                        st.success(f"Removed {cam['camera_id']}")
                        st.rerun()

                # Show track summary
                if tracks:
                    st.markdown("**Detected object classes:**")
                    class_counts = {}
                    for t in tracks:
                        cls = t.get("class_name", "unknown")
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                        st.markdown(f"- **{cls.capitalize()}**: {count} tracks")


# ── Indexing tab ──────────────────────────────────────────────────────────────
def render_indexing_tab():
    st.markdown("### ⚙️ Video Indexing")

    cameras = db.get_all_cameras()
    if not cameras:
        st.warning("No cameras registered. Add cameras first.")
        return

    # Indexing settings
    with st.expander("⚙️ Indexing Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            sample_fps = st.slider("Sample FPS", 1.0, 5.0, cfg.sample_fps, 0.5)
        with col2:
            confidence = st.slider("Detection confidence", 0.1, 0.9, cfg.confidence_threshold, 0.05)
        with col3:
            image_size = st.selectbox("Image size", [416, 640, 1280], index=1)

    st.markdown("#### Select cameras to index:")
    cameras_to_index = []
    for cam in cameras:
        indexed = cam.get("indexed", 0)
        label = f"{cam['camera_id']} — {cam['name']} {'✅' if indexed else '⭕'}"
        if st.checkbox(label, key=f"idx_{cam['camera_id']}", value=not indexed):
            cameras_to_index.append(cam)

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button(
            "🚀 Start Indexing",
            type="primary",
            disabled=st.session_state.indexing_running or len(cameras_to_index) == 0,
            use_container_width=True,
        )
    with col2:
        if st.button("🔄 Rebuild All Indexes", use_container_width=True):
            # Clear FAISS and re-index
            from src.indexing.faiss_store import get_store
            store = get_store()
            import faiss
            store._index = faiss.IndexFlatIP(cfg.embedding_dim)
            store._meta = []
            store.save()
            for cam in cameras:
                db.mark_camera_indexed.__wrapped__ if hasattr(db.mark_camera_indexed, '__wrapped__') else None
                with db.get_conn() as conn:
                    conn.execute("UPDATE cameras SET indexed=0 WHERE camera_id=?", (cam['camera_id'],))
            st.success("Indexes cleared. Re-index cameras to rebuild.")
            st.rerun()

    # Run indexing
    if start_btn and cameras_to_index and not st.session_state.indexing_running:
        st.session_state.indexing_running = True
        _run_indexing(cameras_to_index, sample_fps)

    # Show progress
    if st.session_state.indexing_running:
        st.progress(st.session_state.indexing_progress)
        st.info(st.session_state.indexing_message)


def _run_indexing(cameras: list, sample_fps: float):
    """Run indexing pipeline with progress updates."""
    from src.indexing.track_indexer import VideoIndexer

    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    total = len(cameras)

    for i, cam in enumerate(cameras):
        base_progress = i / total
        cam_progress_share = 1 / total

        def progress_cb(pct: float, msg: str):
            overall = base_progress + pct * cam_progress_share
            st.session_state.indexing_progress = overall
            st.session_state.indexing_message = f"[{cam['camera_id']}] {msg}"
            progress_placeholder.progress(overall)
            status_placeholder.info(f"**{cam['camera_id']}** — {msg}")

        try:
            indexer = VideoIndexer(
                camera_id=cam["camera_id"],
                video_path=cam["video_path"],
                progress_callback=progress_cb,
                sample_fps=sample_fps,
            )
            indexer.run()
            status_placeholder.success(
                f"✅ {cam['camera_id']} indexed: "
                f"{indexer.frames_processed} frames, "
                f"{indexer.tracks_found} tracks, "
                f"{indexer.embeddings_added} embeddings"
            )
        except Exception as e:
            status_placeholder.error(f"❌ Error indexing {cam['camera_id']}: {e}")
            logger.error("Indexing error: %s", e, exc_info=True)

    st.session_state.indexing_running = False
    st.session_state.indexing_progress = 1.0
    st.session_state.indexing_message = "Indexing complete!"
    st.success("🎉 All cameras indexed successfully!")
    st.rerun()


# ── Evaluation tab ────────────────────────────────────────────────────────────
def render_evaluation_tab():
    st.markdown("### 📊 System Evaluation")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Add Evaluation Query")
        with st.form("eval_query_form"):
            eval_query = st.text_input("Query", placeholder="Find the red car at the main gate")
            exp_camera = st.text_input("Expected Camera", placeholder="CAM_01")
            exp_time_start = st.number_input("Expected Time Start (seconds)", value=0.0)
            exp_time_end = st.number_input("Expected Time End (seconds)", value=0.0)

            if st.form_submit_button("Add Query"):
                _add_eval_query(eval_query, exp_camera, exp_time_start, exp_time_end)

    with col2:
        st.markdown("#### Run Evaluation")
        eval_queries_path = cfg.evaluation_queries_path
        if eval_queries_path.exists():
            import json
            with open(eval_queries_path) as f:
                queries = json.load(f)
            st.info(f"Loaded {len(queries)} evaluation queries from `{eval_queries_path}`")

            if st.button("▶️ Run Evaluation", type="primary"):
                _run_evaluation()
        else:
            st.info("No evaluation queries file found. Add queries first.")
            st.markdown(f"File will be created at: `{eval_queries_path}`")


def _add_eval_query(query: str, camera: str, time_start: float, time_end: float):
    import json
    path = cfg.evaluation_queries_path
    path.parent.mkdir(parents=True, exist_ok=True)

    queries = []
    if path.exists():
        with open(path) as f:
            queries = json.load(f)

    queries.append({
        "query": query,
        "expected_camera": camera or None,
        "expected_time_start": time_start or None,
        "expected_time_end": time_end or None,
    })

    with open(path, "w") as f:
        json.dump(queries, f, indent=2)

    st.success(f"Query added. Total: {len(queries)}")


def _run_evaluation():
    from src.evaluation.evaluator import load_eval_queries, run_evaluation

    with st.spinner("Running evaluation..."):
        queries = load_eval_queries()
        if not queries:
            st.error("No evaluation queries found.")
            return
        metrics = run_evaluation(queries)

    st.markdown("#### 📊 Evaluation Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Top-1 Accuracy", f"{metrics.get('top1_accuracy', 0):.1%}")
    col2.metric("Camera Accuracy", f"{metrics.get('camera_accuracy', 0):.1%}")
    col3.metric("Mean Latency", f"{metrics.get('mean_latency_ms', 0):.0f}ms")
    err = metrics.get("mean_localization_error_seconds")
    col4.metric("Time Error", f"{err:.1f}s" if err else "N/A")

    # Per-query results
    if "per_query" in metrics:
        import pandas as pd
        df = pd.DataFrame(metrics["per_query"])
        st.dataframe(df, use_container_width=True)


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    render_header()
    render_sidebar()

    tab = st.session_state.get("active_tab", "🔍 Search")

    if tab == "🔍 Search":
        render_search_tab()
    elif tab in ("📹 Cameras", "📹 Footage & Cameras"):
        render_cameras_tab()
    elif tab == "⚙️ Index":
        render_indexing_tab()
    elif tab == "📊 Evaluate":
        render_evaluation_tab()

    # Footer
    st.markdown("---")
    st.markdown(
        "<center style='color:#475569;font-size:0.75rem'>"
        "🎥 Multi-Stream Video Intelligence | Track-level open-vocabulary CCTV search | "
        "YOLO + ByteTrack + CLIP + FAISS"
        "</center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
