"""
daymax.py
---------
One shared definition of "the daily maximum" for model data.

Polymarket's Munich markets resolve on the highest temperature over the
WHOLE local calendar day (00:00-23:59 Europe/Berlin), and station_obs.py
measures the truth that way. Model forecasts must be reduced the same
way. The old code only looked at 10:00-19:00, which is usually fine in
summer but can miss the real high in winter, when the warmest moment
is sometimes at night (a warm front arriving after dark).
"""

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Berlin"

# Hourly models give 24 values per day (23/25 on DST change days). If a
# day has fewer valid hours than this, its max is probably missing part
# of the day (e.g. the end of a model's forecast range), so we report
# NaN instead of a max that is silently too low.
MIN_VALID_HOURS = 20


def hourly_times_local(hourly, n_values):
    """Timestamps (Europe/Berlin) for an Open-Meteo SDK Hourly() block."""
    start, step = hourly.Time(), hourly.Interval()
    ts_utc = pd.to_datetime(np.arange(start, start + n_values * step, step), unit="s", utc=True)
    return ts_utc.tz_convert(LOCAL_TZ)


def daily_max_by_date(ts_local, values):
    """{'YYYY-MM-DD': max over the full local day} for an hourly series.
    Days with too few valid hours map to NaN (see MIN_VALID_HOURS)."""
    df = pd.DataFrame({"date": pd.DatetimeIndex(ts_local).strftime("%Y-%m-%d"),
                       "temp": values}).dropna(subset=["temp"])
    out = {}
    for d, grp in df.groupby("date"):
        out[d] = float(grp["temp"].max()) if len(grp) >= MIN_VALID_HOURS else float("nan")
    return out


def daily_max_for(target_date_str, ts_local, values):
    """Full-day max for one specific date (NaN if missing or incomplete)."""
    return daily_max_by_date(ts_local, values).get(target_date_str, float("nan"))
