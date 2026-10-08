"""
tests/test_time_parser.py — Tests for time range parsing.
"""
import pytest
from src.query.time_parser import parse_time_range, format_timestamp


class TestParseTimeRange:
    def test_all_returns_none(self):
        start, end = parse_time_range("all")
        assert start is None
        assert end is None

    def test_empty_returns_none(self):
        start, end = parse_time_range("")
        assert start is None

    def test_last_hour(self):
        start, end = parse_time_range("last hour", max_video_seconds=7200)
        assert start == pytest.approx(3600, abs=1)
        assert end == pytest.approx(7200, abs=1)

    def test_last_30_minutes(self):
        start, end = parse_time_range("last 30 minutes", max_video_seconds=3600)
        assert start == pytest.approx(1800, abs=1)
        assert end == pytest.approx(3600, abs=1)

    def test_last_1_hour(self):
        start, end = parse_time_range("last 1 hour", max_video_seconds=10000)
        assert start == pytest.approx(6400, abs=1)

    def test_last_5_seconds(self):
        start, end = parse_time_range("last 5 seconds", max_video_seconds=100)
        assert start == pytest.approx(95, abs=1)
        assert end == pytest.approx(100, abs=1)

    def test_first_5_minutes(self):
        start, end = parse_time_range("first 5 minutes")
        assert start == pytest.approx(0)
        assert end == pytest.approx(300, abs=1)

    def test_between_times(self):
        start, end = parse_time_range("between 00:10 and 00:30")
        assert start == pytest.approx(600)
        assert end == pytest.approx(1800)

    def test_at_time(self):
        start, end = parse_time_range("at 01:00:00")
        assert start == pytest.approx(3300, abs=1)
        assert end == pytest.approx(3900, abs=1)

    def test_today_returns_none(self):
        start, end = parse_time_range("today")
        assert start is None

    def test_clamp_start_to_zero(self):
        start, end = parse_time_range("last hour", max_video_seconds=30)
        assert start == pytest.approx(0)


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0) == "00:00:00"

    def test_one_hour(self):
        assert format_timestamp(3600) == "01:00:00"

    def test_complex(self):
        assert format_timestamp(3723) == "01:02:03"
