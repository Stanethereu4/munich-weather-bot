"""
timeutil.py
-----------
Explicit, timezone-aware time helpers.

Plain datetime.now() returns the clock of whatever machine runs the code
WITHOUT a timezone: UTC on GitHub's servers, Paris time on a Mac. That
silently shifted "today" and lead times by 1-2 hours depending on where
the bot ran. Use these instead:
  - utc_now() / utc_now_iso()  for timestamps stored in the database
  - berlin_now() / berlin_today()  for anything about the market's
    calendar day (markets are defined on Munich local dates)
"""

import uuid
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """e.g. '2026-11-25T09:30:00.123456+00:00' -- sorts correctly as text."""
    return utc_now().isoformat()


def berlin_now() -> datetime:
    return datetime.now(BERLIN)


def berlin_today() -> date:
    return berlin_now().date()


def hours_until_local_day_start(target: date) -> float:
    """Hours from now until 00:00 Munich time on the target date
    (0 if that day has already started). This is the bot's 'lead time'."""
    start = datetime(target.year, target.month, target.day, tzinfo=BERLIN)
    return max(0.0, (start - berlin_now()).total_seconds() / 3600.0)


def new_snapshot_id() -> str:
    """One id shared by every row written in the same snapshot."""
    return uuid.uuid4().hex
