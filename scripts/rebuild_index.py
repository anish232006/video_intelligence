"""
scripts/rebuild_index.py — Rebuild FAISS index from existing SQLite data.

Used when the FAISS index file is lost but SQLite metadata and crops are intact.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Rebuild FAISS index from crops")
    parser.add_argument("--camera", help="Specific camera to rebuild (default: all)")
    args = parser.parse_args()

    from src.database import init_db, get_all_cameras, get_tracks_for_camera, get_embeddings_for_track
    from src.embeddings.clip_encoder import CLIPEncoder
    from src.indexing.faiss_store import FAISSStore
    from src.config import cfg

    init_db()

    cameras = get_all_cameras()
    if args.camera:
        cameras = [c for c in cameras if c["camera_id"] == args.camera]

    if not cameras:
        logger.error("No cameras found.")
        sys.exit(1)

    store = FAISSStore()
    store._index = None  # Force fresh creation
    store.load_or_create()

    encoder = CLIPEncoder()
    encoder.load()

    for cam in cameras:
        cam_id = cam["camera_id"]
        logger.info("Rebuilding index for %s...", cam_id)
        tracks = get_tracks_for_camera(cam_id)

        for track in tracks:
            crop_path = track.get("best_crop")
            if not crop_path or not Path(crop_path).exists():
                logger.warning("No crop for track %d in %s", track["track_id"], cam_id)
                continue

            emb = encoder.encode_single_image(crop_path)
            meta = {
                "track_db_id": track["id"],
                "camera_id": cam_id,
                "crop_path": crop_path,
                "is_representative": True,
                "class_name": track.get("class_name", ""),
                "dominant_color": track.get("dominant_color"),
                "first_seen": track.get("first_seen", 0),
                "last_seen": track.get("last_seen", 0),
            }
            store.add(emb.reshape(1, -1), [meta])

        logger.info("Camera %s: %d tracks processed", cam_id, len(tracks))

    encoder.unload()
    store.save()
    logger.info("Index rebuilt. Total vectors: %d", store.total_vectors)


if __name__ == "__main__":
    main()
