"""
station_obs.py
--------------
Ground truth = what Polymarket actually resolves on.

Munich markets resolve on the highest temperature in the Wunderground
"Daily Observations" table for Munich Airport (EDDM), in whole degrees
Celsius, for the local calendar day (Europe/Berlin). That table is
built from the airport's METAR reports, so the daily max of the
METAR temperatures over the local day is the closest reproducible
proxy. Gridded reanalysis (Open-Meteo archive) is NOT: it can differ
from the station by around a degree, which is a whole bracket.

Sources, in order:
  1. Iowa Environmental Mesonet (IEM) METAR archive: years of history,
     used for calibration windows and older settlements.
  2. aviationweather.gov (NOAA AWC): last ~15 days only, used to fill
     any recent day IEM doesn't have yet.

Run directly to compare both sources over the last few days:
    python3 station_obs.py
"""

import io
import math
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

STATION = "EDDM"
LOCAL_TZ = "Europe/Berlin"
HEADERS = {"User-Agent": "munich-weather-bot (research; github.com/Stanethereu4)"}


def _round_half_up(x):
    """Whole-degree rounding as displayed by the resolution source
    (Python's round() is banker's rounding: round(0.5) == 0)."""
    return int(math.floor(x + 0.5))


# EDDM reports every 30 min (48/day) plus specials. A day with fewer
# reports than this is treated as incomplete (not yet fully archived,
# or a data gap) rather than settled on a partial max.
MIN_REPORTS_PER_DAY = 40


def _daily_max(df):
    """df: columns local_dt (tz-aware, Europe/Berlin), temp_c (float).
    Returns {YYYY-MM-DD: whole-degree max} per complete local day."""
    df = df.dropna(subset=["local_dt", "temp_c"])
    if df.empty:
        return {}
    df = df.assign(date=df["local_dt"].dt.strftime("%Y-%m-%d"))
    return {d: _round_half_up(g["temp_c"].max())
            for d, g in df.groupby("date") if len(g) >= MIN_REPORTS_PER_DAY}


def fetch_iem(start: date, end: date):
    """All routine + special METARs for EDDM between start and end
    (inclusive, local dates). Returns a DataFrame (local_dt, temp_c)."""
    end_excl = end + timedelta(days=1)
    params = {
        "station": STATION, "data": "tmpc", "tz": LOCAL_TZ,
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end_excl.year, "month2": end_excl.month, "day2": end_excl.day,
        "format": "onlycomma", "latlon": "no", "missing": "M",
        "direct": "no", "report_type": [3, 4],
    }
    r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
                     params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text), na_values=["M"])
    if raw.empty or "tmpc" not in raw.columns:
        return pd.DataFrame(columns=["local_dt", "temp_c"])
    local_dt = pd.to_datetime(raw["valid"]).dt.tz_localize(
        LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward")
    return pd.DataFrame({"local_dt": local_dt, "temp_c": pd.to_numeric(raw["tmpc"], errors="coerce")})


def fetch_awc_day(day: date):
    """METARs from aviationweather.gov covering one local day (AWC only
    keeps ~15 days). Returns a DataFrame (local_dt, temp_c)."""
    # local midnight..midnight is at most 23:00Z-23:00Z (CET) / 22:00Z (CEST);
    # ask for a 27h window ending 02:00Z the next day, then filter.
    end_utc = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(days=1, hours=2)
    params = {"ids": STATION, "format": "json", "hours": 27,
              "date": end_utc.strftime("%Y%m%d_%H%M")}
    r = requests.get("https://aviationweather.gov/api/data/metar",
                     params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    rows = r.json() if r.text.strip() else []
    if not rows:
        return pd.DataFrame(columns=["local_dt", "temp_c"])
    df = pd.DataFrame({
        "local_dt": pd.to_datetime([x.get("obsTime") for x in rows], unit="s", utc=True).tz_convert(LOCAL_TZ),
        "temp_c": [x.get("temp") for x in rows],
    })
    df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    return df[df["local_dt"].dt.strftime("%Y-%m-%d") == day.strftime("%Y-%m-%d")]


def fetch_daily_max(start, end, verbose=False):
    """{YYYY-MM-DD: whole-degree station max} for start..end (dates or
    'YYYY-MM-DD' strings). Days with no data at all are left out, so
    callers can tell 'unknown' from a real value."""
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)

    result = {}
    try:
        result.update(_daily_max(fetch_iem(start, end)))
    except Exception as e:
        print(f"[station] IEM fetch failed ({start}..{end}): {e}")

    awc_horizon = date.today() - timedelta(days=14)
    d = max(start, awc_horizon)
    while d <= end:
        key = d.strftime("%Y-%m-%d")
        if key not in result:
            try:
                result.update(_daily_max(fetch_awc_day(d)))
            except Exception as e:
                print(f"[station] AWC fetch failed for {key}: {e}")
        d += timedelta(days=1)

    if verbose:
        missing = (end - start).days + 1 - len(result)
        print(f"[station] {STATION} daily max: {len(result)} day(s) found, {missing} missing")
    return result


def _compare_sources(days=7):
    """Diagnostic: IEM vs AWC vs Open-Meteo grid for recent days."""
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    iem = _daily_max(fetch_iem(start, end))
    print(f"{'date':<12}{'IEM':>6}{'AWC':>6}   (whole °C, local day, {STATION})")
    for i in range(days):
        d = start + timedelta(days=i)
        k = d.strftime("%Y-%m-%d")
        try:
            awc = _daily_max(fetch_awc_day(d)).get(k)
        except Exception as e:
            awc = f"err:{e}"[:20]
        print(f"{k:<12}{str(iem.get(k)):>6}{str(awc):>6}")


if __name__ == "__main__":
    _compare_sources()
