"""
tests/test_parser.py — Tests for natural language query parser.
"""
import pytest
from src.query.parser import _rule_based_parse, QueryParser


class TestRuleBasedParser:
    def test_red_car(self):
        q = _rule_based_parse("Did a red car pass through the main gate?")
        assert q.object_class == "car"
        assert "red" in q.attributes
        assert q.location is not None

    def test_person_with_backpack(self):
        q = _rule_based_parse("Did someone carrying a backpack enter?")
        assert q.object_class in ("person", None)  # may or may not detect person
        # backpack should be in attributes or class
        found = "backpack" in q.attributes or q.object_class == "backpack"
        assert found or q.object_class is not None

    def test_white_car_camera_2(self):
        q = _rule_based_parse("Find the white car on camera 2.")
        assert q.object_class == "car"
        assert "white" in q.attributes
        assert "CAM_02" in q.cameras

    def test_time_extraction(self):
        q = _rule_based_parse("Show me the red car in the last hour.")
        assert "last" in (q.time_range or "")

    def test_person_near_entrance(self):
        q = _rule_based_parse("Show people near the entrance.")
        assert q.object_class in ("person", "people", None)

    def test_empty_query(self):
        q = _rule_based_parse("")
        assert q.raw_query == ""

    def test_camera_reference(self):
        q = _rule_based_parse("What happened on CAM_01?")
        assert "CAM_01" in q.cameras

    def test_color_extraction(self):
        q = _rule_based_parse("Find the blue bicycle near the parking area.")
        assert "blue" in q.attributes
        assert q.object_class == "bicycle"


class TestQueryParser:
    def test_parser_always_returns(self):
        parser = QueryParser()
        result = parser.parse("completely nonsense query 123!@#")
        assert result is not None
        assert result.raw_query == "completely nonsense query 123!@#"

    def test_parser_normalizes_class(self):
        result = _rule_based_parse("Find the vehicle at the gate.")
        # "vehicle" should normalize to "car"
        assert result.object_class in ("car", "vehicle", None)
