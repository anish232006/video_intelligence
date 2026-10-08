"""
tests/test_database.py — Database layer tests.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def temp_db(tmp_path):
    """Provide a fresh temporary database for each test."""
    db_file = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_file):
        import src.database as db
        # Reload to use the patched path
        db.DB_PATH = db_file
        db.init_db()
        yield db
        # Cleanup
        if db_file.exists():
            db_file.unlink()


class TestDatabase:
    def test_init_db(self, temp_db):
        """DB initialization should not raise."""
        # Already initialized in fixture
        stats = temp_db.get_stats()
        assert stats["cameras"] == 0

    def test_upsert_camera(self, temp_db):
        row_id = temp_db.upsert_camera(
            camera_id="CAM_01",
            name="Test Camera",
            video_path="/test/video.mp4",
            description="Test",
        )
        assert row_id is not None
        assert row_id > 0
        cameras = temp_db.get_all_cameras()
        assert len(cameras) == 1
        assert cameras[0]["camera_id"] == "CAM_01"

    def test_upsert_camera_update(self, temp_db):
        temp_db.upsert_camera("CAM_01", "Old Name", "/video.mp4")
        temp_db.upsert_camera("CAM_01", "New Name", "/video.mp4")
        camera = temp_db.get_camera("CAM_01")
        assert camera["name"] == "New Name"

    def test_upsert_track(self, temp_db):
        # Need camera first
        temp_db.upsert_camera("CAM_01", "Test", "/video.mp4")
        track_id = temp_db.upsert_track(
            camera_id="CAM_01",
            track_id=1,
            class_name="car",
            first_seen=10.0,
            last_seen=20.0,
            confidence=0.85,
            dominant_color="red",
        )
        assert track_id > 0
        tracks = temp_db.get_tracks_for_camera("CAM_01")
        assert len(tracks) == 1
        assert tracks[0]["class_name"] == "car"
        assert tracks[0]["dominant_color"] == "red"

    def test_knowledge_save_and_resolve(self, temp_db):
        temp_db.save_knowledge("main gate", "CAM_01")
        result = temp_db.resolve_location("main gate")
        assert result is not None
        assert result["camera_id"] == "CAM_01"

    def test_knowledge_alias(self, temp_db):
        temp_db.save_knowledge("main gate", "CAM_01")
        temp_db.add_alias("main gate", "front gate")
        result = temp_db.resolve_location("front gate")
        assert result is not None
        assert result["camera_id"] == "CAM_01"

    def test_unknown_location_returns_none(self, temp_db):
        result = temp_db.resolve_location("unknown place")
        assert result is None

    def test_camera_stats(self, temp_db):
        temp_db.upsert_camera("CAM_01", "Test", "/video.mp4")
        temp_db.upsert_camera("CAM_02", "Test 2", "/video2.mp4")
        stats = temp_db.get_stats()
        assert stats["cameras"] == 2

    def test_track_filter(self, temp_db):
        temp_db.upsert_camera("CAM_01", "Test", "/video.mp4")
        temp_db.upsert_track("CAM_01", 1, "car", 0, 100, dominant_color="red")
        temp_db.upsert_track("CAM_01", 2, "person", 0, 100)

        cars = temp_db.get_tracks_filtered(class_name="car")
        assert len(cars) == 1
        assert cars[0]["class_name"] == "car"

        red = temp_db.get_tracks_filtered(color="red")
        assert len(red) == 1

    def test_mark_camera_indexed(self, temp_db):
        temp_db.upsert_camera("CAM_01", "Test", "/video.mp4")
        temp_db.mark_camera_indexed("CAM_01")
        cam = temp_db.get_camera("CAM_01")
        assert cam["indexed"] == 1

    def test_delete_knowledge(self, temp_db):
        temp_db.save_knowledge("main gate", "CAM_01")
        temp_db.delete_knowledge("main gate")
        result = temp_db.resolve_location("main gate")
        assert result is None
