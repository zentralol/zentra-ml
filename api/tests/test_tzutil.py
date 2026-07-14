"""Unit tests for tzutil timezone helpers."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from tzutil import MANHATTAN_TZ, now_in_manhattan, to_manhattan_time


class TestToManhattanTime:
    def test_naive_datetime_is_treated_as_manhattan_local(self):
        dt = datetime(2026, 7, 14, 15, 0, 0)
        result = to_manhattan_time(dt)

        assert result.tzinfo is MANHATTAN_TZ
        assert result.year == 2026
        assert result.month == 7
        assert result.day == 14
        assert result.hour == 15
        assert result.minute == 0

    def test_utc_datetime_is_converted_to_manhattan(self):
        dt = datetime(2026, 7, 14, 19, 0, 0, tzinfo=timezone.utc)
        result = to_manhattan_time(dt)

        assert result.tzinfo is MANHATTAN_TZ
        # 19:00 UTC = 15:00 EDT (Manhattan, July)
        assert result.hour == 15

    def test_other_timezone_is_converted_to_manhattan(self):
        tokyo = ZoneInfo("Asia/Tokyo")
        dt = datetime(2026, 7, 14, 15, 0, 0, tzinfo=tokyo)
        result = to_manhattan_time(dt)

        assert result.tzinfo is MANHATTAN_TZ
        # Tokyo is ahead of Manhattan, so the hour should be earlier in NY
        assert result.hour != 15


class TestNowInManhattan:
    def test_returns_aware_datetime_in_manhattan_timezone(self):
        result = now_in_manhattan()

        assert result.tzinfo is MANHATTAN_TZ
