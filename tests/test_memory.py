"""
tests/test_memory.py — Tests for clarify-once persistent memory.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestMemory:
    def test_unknown_location_returns_none(self):
        """Unknown location should return None (needs clarification)."""
        from unittest.mock import patch
        with patch("src.database.resolve_location", return_value=None):
            from src.query.memory import resolve_location_to_camera
            result = resolve_location_to_camera("unknown place xyz")
            assert result is None

    def test_known_location_resolves(self):
        """Known location should return camera_id."""
        mock_entry = {"camera_id": "CAM_01", "term": "main gate", "type": "location"}
        with patch("src.database.resolve_location", return_value=mock_entry):
            from src.query.memory import resolve_location_to_camera
            result = resolve_location_to_camera("main gate")
            assert result == "CAM_01"

    def test_learn_location(self):
        """Learning a location should call db.save_knowledge."""
        with patch("src.database.save_knowledge") as mock_save, \
             patch("src.database.add_alias"):
            from src.query.memory import learn_location
            learn_location("main gate", "CAM_01")
            mock_save.assert_called_once()
            args = mock_save.call_args[0]
            assert args[0] == "main gate"
            assert args[1] == "CAM_01"

    def test_learn_location_with_aliases(self):
        """Learning with aliases should call add_alias for each."""
        with patch("src.database.save_knowledge"), \
             patch("src.database.add_alias") as mock_alias:
            from src.query.memory import learn_location
            learn_location("main gate", "CAM_01", aliases=["front gate", "entrance gate"])
            assert mock_alias.call_count == 2

    def test_is_location_known_true(self):
        """is_location_known should return True for known locations."""
        mock_entry = {"camera_id": "CAM_01"}
        with patch("src.database.resolve_location", return_value=mock_entry):
            from src.query.memory import is_location_known
            assert is_location_known("main gate") is True

    def test_is_location_known_false(self):
        """is_location_known should return False for unknown locations."""
        with patch("src.database.resolve_location", return_value=None):
            from src.query.memory import is_location_known
            assert is_location_known("unknown place") is False
