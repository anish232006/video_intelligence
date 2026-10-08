"""
scripts/index_video.py — Command-line video indexing script.

Usage:
    python scripts/index_video.py --camera CAM_01 --name "Main Gate" --video data/videos/camera_01.mp4
    python scripts/index_video.py --camera CAM_01 --name "Main Gate" --video data/videos/camera_01.mp4 --fps 2
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Index a camera video")
    parser.add_argument("--camera", required=True, help="Camera ID (e.g. CAM_01)")
    parser.add_argument("--name", required=True, help="Camera name")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--desc", default="", help="Location description")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling FPS (1-5)")
    args = parser.parse_args()

    from src.database import init_db
    from src.ingestion.metadata import register_video
    from src.indexing.track_indexer import VideoIndexer

    init_db()

    # Register camera
    logger.info("Registering camera %s...", args.camera)
    info = register_video(args.camera, args.name, args.video, args.desc)
    if not info:
        logger.error("Failed to register video. Aborting.")
        sys.exit(1)

    # Index
    def progress_cb(pct: float, msg: str):
        bar = "█" * int(pct * 40) + "░" * (40 - int(pct * 40))
        print(f"\r[{bar}] {int(pct*100)}% — {msg}", end="", flush=True)

    logger.info("Starting indexing...")
    indexer = VideoIndexer(
        camera_id=args.camera,
        video_path=args.video,
        progress_callback=progress_cb,
        sample_fps=args.fps,
    )

    result = indexer.run()
    print()  # newline after progress
    logger.info("Done: %s", result)


if __name__ == "__main__":
    main()
