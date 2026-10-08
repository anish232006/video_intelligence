"""
scripts/reindex_all_sequential.py
Cleans database & FAISS index, registers 4 sequential CCTV cameras,
indexes them with YOLOv8 + ByteTrack + OpenCLIP, and verifies cross-camera Re-ID.
"""
import sys
import shutil
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
import src.database as db
from src.ingestion.metadata import register_video
from src.indexing.track_indexer import VideoIndexer
from src.indexing.faiss_store import get_store
from src.reid.cross_camera import find_cross_camera_matches


def clean_state():
    print("--- 1. Resetting Database and FAISS Index ---")
    db.init_db()
    with db.get_conn() as conn:
        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM track_observations")
        conn.execute("DELETE FROM tracks")
        conn.execute("DELETE FROM cameras")
        conn.execute("DELETE FROM knowledge")
        conn.execute("DELETE FROM aliases")
    
    # Remove crops directory
    if cfg.crops_dir.exists():
        shutil.rmtree(cfg.crops_dir, ignore_errors=True)
    cfg.crops_dir.mkdir(parents=True, exist_ok=True)

    # Remove FAISS index & metadata
    if cfg.faiss_index_path.exists():
        cfg.faiss_index_path.unlink()
    if cfg.faiss_meta_path.exists():
        cfg.faiss_meta_path.unlink()
        
    # Reset in-memory store
    store = get_store()
    store._index = None
    store._meta = []
    print("Cleaned existing database, crops, and FAISS store.")


def register_all_cameras():
    print("\n--- 2. Registering 4 Sequential Cameras ---")
    cams = [
        ("CAM_01", "CAM 01 - North Highway Inbound", "data/videos/camera_01.mp4", "Wide-angle overview of highway sector with traffic, trucks, cars, and pedestrians"),
        ("CAM_02", "CAM 02 - Gate 2 West Checkpoint", "data/videos/camera_02.mp4", "Telephoto zoomed reverse-angle checkpoint feed showing vehicles passing barrier lane"),
        ("CAM_03", "CAM 03 - East Commercial Walkway", "data/videos/camera_03.mp4", "Commercial pedestrian corridor view showing pedestrians, shoppers, and walkway movement"),
        ("CAM_04", "CAM 04 - Metro Central Intersection", "data/videos/camera_04.mp4", "High-contrast IR monochrome surveillance feed of central intersection cross-traffic"),
    ]
    for cid, name, vpath, desc in cams:
        info = register_video(cid, name, vpath, desc)
        print(f"Registered {cid}: {name} ({vpath}) - duration={info.duration:.1f}s, fps={info.fps}")


def index_all_cameras():
    print("\n--- 3. Running VideoIndexer on All 4 Cameras ---")
    cams = db.get_all_cameras()
    for cam in cams:
        cid = cam["camera_id"]
        vpath = cam["video_path"]
        print(f"\nIndexing {cid} ({vpath})...")
        
        def progress(p, msg):
            if int(p * 100) % 25 == 0:
                print(f"  [{cid}] {int(p*100)}% - {msg}")
                
        indexer = VideoIndexer(
            camera_id=cid,
            video_path=vpath,
            progress_callback=progress,
            sample_fps=2.0
        )
        res = indexer.run()
        print(f"Completed {cid}: processed {res.get('frames_processed', 0)} frames, found {res.get('tracks_found', 0)} tracks, added {res.get('embeddings_added', 0)} embeddings.")


def verify_tracks_and_reid():
    print("\n--- 4. Verifying Tracks & Cross-Camera Re-ID ---")
    all_tracks = db.get_all_tracks()
    print(f"Total tracks indexed across all cameras: {len(all_tracks)}")
    
    for t in all_tracks:
        print(f"  Track DB ID {t['id']} | Camera: {t['camera_id']} | Track #{t['track_id']} | Class: {t['class_name']} | Color: {t['dominant_color']} | Time: {t['first_seen']:.1f}s - {t['last_seen']:.1f}s")
        
    # Test Cross-Camera Re-ID starting from CAM_01
    cam1_tracks = [t for t in all_tracks if t["camera_id"] == "CAM_01"]
    if not cam1_tracks:
        print("ERROR: No tracks found on CAM_01!")
        return
        
    target_track = cam1_tracks[0]
    print(f"\nTracing Target Entity from CAM_01 (Track DB ID {target_track['id']}, class={target_track['class_name']}, color={target_track['dominant_color']}):")
    
    matches = find_cross_camera_matches(target_track["id"], max_results=10)
    print(f"Cross-camera matches found: {len(matches)}")
    for m in matches:
        print(f"  -> Matched {m.camera_b} at t={m.time_b:.1f}s (sim={m.similarity:.3f}, confidence={m.confidence_label})")


if __name__ == "__main__":
    clean_state()
    register_all_cameras()
    index_all_cameras()
    verify_tracks_and_reid()
