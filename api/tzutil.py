"""Timezone helpers for the Manhattan-specific Zentra API.

All prediction datetimes are normalised to America/New_York (Manhattan local time)
before they are passed to inference or compared to 'now'.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

MANHATTAN_TZ = ZoneInfo("America/New_York")


def now_in_manhattan() -> datetime:
    """Current time in America/New_York."""
    return datetime.now(MANHATTAN_TZ)


def to_manhattan_time(v: datetime) -> datetime:
    """Convert a datetime to America/New_York.

    Naive datetimes are treated as already being in Manhattan local time.
    Offset-aware datetimes are converted to Manhattan local time.
    """
    if v.tzinfo is None:
        return v.replace(tzinfo=MANHATTAN_TZ)
    return v.astimezone(MANHATTAN_TZ)
