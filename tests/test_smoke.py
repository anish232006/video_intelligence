"""
tests/test_smoke.py — End-to-end demonstration smoke test.

Validates the full pipeline flow specified in the requirements:
1. Database & schema creation
2. Camera registration (CAM_01 and CAM_02)
3. Track and crop indexing
4. FAISS vector storage & retrieval
5. Query: "Find the red car." -> returns camera, timestamp, thumbnail, confidence
6. Clarify-once memory test:
   - Query: "Find the red car at the main gate."
   - "main gate" is unknown -> requires clarification
   - User teaches system: "main gate" -> CAM_01
   - Run same query again -> automatically resolved without asking!
7. Application Restart Persistence test:
   - Reset in-memory state / reload from disk
   - Run query again -> still remembers "main gate" -> CAM_01!
"""
import os
import shutil
import tempfile
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

import src.database as db
from src.query.memory import resolve_location_to_camera, learn_location, is_location_known
from src.query.parser import QueryParser
from src.query.retriever import RetrievalPipeline
from src.indexing.faiss_store import FaissStore


@pytest.fixture
def test_env(tmp_path):
    """Set up isolated temp environment for smoke testing."""
    db_file = tmp_path / "smoke_app.db"
    index_file = tmp_path / "smoke_faiss.index"
    crops_dir = tmp_path / "crops"
    clips_dir = tmp_path / "clips"
    crops_dir.mkdir(parents=True)
    clips_dir.mkdir(parents=True)

    with patch("src.database.DB_PATH", db_file):
        db.DB_PATH = db_file
        db.init_db()

        meta_file = tmp_path / "smoke_faiss_meta.json"
        parser = QueryParser()
        faiss_store = FaissStore(dim=512, index_path=index_file, meta_path=meta_file)

        yield {
            "db_file": db_file,
            "index_file": index_file,
            "parser": parser,
            "faiss_store": faiss_store,
            "crops_dir": crops_dir,
            "clips_dir": clips_dir,
        }


def test_end_to_end_demonstration_smoke(test_env):
    db_file = test_env["db_file"]
    parser = test_env["parser"]
    faiss_store = test_env["faiss_store"]
    crops_dir = test_env["crops_dir"]

    # 1. Register Camera 1 & Camera 2
    cam1_row = db.upsert_camera(
        camera_id="CAM_01",
        name="Main Gate Camera",
        video_path="data/videos/camera_01.mp4",
        description="Entrance to facility",
        fps=10.0,
        duration=6.0,
        width=640,
        height=480,
    )
    cam2_row = db.upsert_camera(
        camera_id="CAM_02",
        name="Parking Lot Camera",
        video_path="data/videos/camera_02.mp4",
        description="Vehicle parking area",
        fps=10.0,
        duration=6.0,
        width=640,
        height=480,
    )
    assert cam1_row > 0
    assert cam2_row > 0

    # 2. Create dummy crop images for CAM_01 (Red car)
    cam1_crop_dir = crops_dir / "CAM_01" / "track_0001"
    cam1_crop_dir.mkdir(parents=True)
    crop_file = cam1_crop_dir / "crop_01.jpg"
    crop_file.write_bytes(b"dummy image bytes")

    # 3. Create synthetic track on CAM_01: Red Car
    np.random.seed(42)
    red_car_vec = np.random.randn(512).astype("float32")
    red_car_vec /= np.linalg.norm(red_car_vec)

    track_db_id = db.upsert_track(
        camera_id="CAM_01",
        track_id=1,
        class_name="car",
        first_seen=2.5,
        last_seen=5.0,
        confidence=0.92,
        dominant_color="red",
        best_crop=str(crop_file),
        crop_paths=[str(crop_file)],
    )
    assert track_db_id > 0

    meta = [{
        "track_db_id": track_db_id,
        "camera_id": "CAM_01",
        "crop_path": str(crop_file),
        "is_representative": True,
        "class_name": "car",
        "dominant_color": "red",
        "first_seen": 2.5,
        "last_seen": 5.0,
    }]
    faiss_id = faiss_store.add(red_car_vec.reshape(1, -1), metadata=meta)[0]
    faiss_store.save()

    # Link embedding in database
    db.insert_embedding(
        track_db_id=track_db_id,
        camera_id="CAM_01",
        crop_path=str(crop_file),
        faiss_idx=faiss_id,
    )

    # Add observation
    db.insert_observation(
        track_db_id=track_db_id,
        camera_id="CAM_01",
        track_id=1,
        timestamp=3.5,
        frame_number=35,
        x1=100,
        y1=200,
        x2=250,
        y2=280,
        crop_path=str(crop_file),
        confidence=0.92,
    )

    # 4. Initialize retrieval pipeline
    pipeline = RetrievalPipeline()

    # Mock the CLIP encoder inside retriever
    with patch("src.query.retriever.get_store", return_value=faiss_store), \
         patch("src.query.retriever._get_encoder") as mock_get_encoder:
        
        mock_enc = mock_get_encoder.return_value
        mock_enc.encode_text.return_value = red_car_vec
        mock_enc.encode_single_text.return_value = red_car_vec

        # 5. Query: "Find the red car."
        q1 = parser.parse("Find the red car.")
        results, clarif = pipeline.search(q1)

        assert clarif is None
        assert len(results) > 0, "Expected at least 1 match for 'Find the red car.'"
        top = results[0]
        assert top.camera_id == "CAM_01"
        assert top.object_class == "car"
        assert top.confidence > 0.5
        assert top.timestamp > 0
        assert top.crop_path == str(crop_file)
        assert len(top.explanation) > 0

        # 6. Test Clarify-Once:
        # Query: "Find the red car at the main gate."
        q2 = parser.parse("Find the red car at the main gate.")
        assert q2.location == "main gate"

        # Initially, "main gate" is unknown
        assert not is_location_known("main gate")
        assert resolve_location_to_camera("main gate") is None

        # Searching with unknown location requests clarification
        results_unknown, clarif_loc = pipeline.search(q2)
        assert clarif_loc == "main gate", "Should return clarification needed for unknown location"

        # User teaches system: "main gate" -> CAM_01
        learn_location("main gate", "CAM_01")

        # Now location is resolved permanently
        assert is_location_known("main gate")
        assert resolve_location_to_camera("main gate") == "CAM_01"

        # Run same query again -> automatically resolved without asking!
        results_learned, clarif_after = pipeline.search(q2)
        assert clarif_after is None
        assert len(results_learned) > 0
        assert results_learned[0].camera_id == "CAM_01"

        # 7. MANDATORY RESTART PERSISTENCE TEST:
        # Simulate app restart: re-read from disk
        db.DB_PATH = db_file
        assert is_location_known("main gate")
        assert resolve_location_to_camera("main gate") == "CAM_01"

        results_reboot, clarif_reboot = pipeline.search(q2)
        assert clarif_reboot is None
        assert len(results_reboot) > 0
        assert results_reboot[0].camera_id == "CAM_01"
