"""
backtest/timeconv.py -- datetime -> unix seconds, independent of pandas' internal unit.

pandas 3 stores parsed datetimes in microseconds (pandas 2 used nanoseconds),
so `.astype("int64") // 10**9` silently gives wrong numbers there. Subtracting
the epoch and dividing by one second works with any unit.
"""
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def to_unix_seconds(values):
    """ISO strings / datetimes (any unit) -> int64 unix seconds UTC."""
    dt = pd.to_datetime(values, utc=True, format="ISO8601")
    return ((dt - EPOCH) // pd.Timedelta(seconds=1)).astype("int64")
