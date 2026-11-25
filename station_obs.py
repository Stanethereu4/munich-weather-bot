"""
station_obs.py
--------------
Ground truth = what Polymarket actually resolves on.

Works for both market kinds: kind="high" -> daily max ("Highest
temperature" markets), kind="low" -> daily min ("Lowest temperature"
markets). The rules below are identical for both, with min in place of max.

Polymarket's rules (read from the live market description, 25 Nov 2026):
  - source: ROUTINE METAR reports for Munich Airport (EDDM) published by
    aviationweather.gov (NOAA AWC). SPECI (special) reports do NOT count.
  - value: the hourly T-group remark in tenths if present, otherwise the
    whole-degree temperature in the report body. (EDDM reports carry no
    T-group, so in practice whole degrees.)
  - the max over all reports whose time falls on the local date
    (Europe/Berlin), rounded half UP to a whole degree.
  - Wunderground is only a fallback if AWC publishes nothing.
Gridded reanalysis (Open-Meteo archive) can differ from the station by
about a degree, i.e. a whole bracket, so it is never used as truth.

Sources, in order:
  1. aviationweather.gov (AWC): the official source; SPECI filtered out.
     Only keeps ~15 days, so used for settlement and recent days.
  2. Iowa Environmental Mesonet (IEM) archive: years of history, used for
     older days (calibration windows). Caveat: IEM cannot tell SPECI
     from routine half-hourly reports for EDDM (it strips the prefix),
     so specials are included there -- a small upward bias on the rare
     days a SPECI is warmer than every routine report.

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


def _daily_extreme(df, kind="high"):
    """df: columns local_dt (tz-aware, Europe/Berlin), temp_c (float).
    Returns {YYYY-MM-DD: whole-degree max (high) or min (low)} per complete
    local day. Rounding comes after taking the extreme; same result as
    rounding each report first, since rounding never changes the order."""
    df = df.dropna(subset=["local_dt", "temp_c"])
    if df.empty:
        return {}
    df = df.assign(date=df["local_dt"].dt.strftime("%Y-%m-%d"))
    pick = (lambda g: g["temp_c"].max()) if kind == "high" else (lambda g: g["temp_c"].min())
    return {d: _round_half_up(pick(g))
            for d, g in df.groupby("date") if len(g) >= MIN_REPORTS_PER_DAY}


def _daily_max(df):
    return _daily_extreme(df, "high")


def fetch_iem(start: date, end: date):
    """All routine + special METARs for EDDM between start and end
    (inclusive, local dates). Returns a DataFrame (local_dt, temp_c)."""
    end_excl = end + timedelta(days=1)
    params = {
        "station": STATION, "data": "tmpc", "tz": LOCAL_TZ,
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end_excl.year, "month2": end_excl.month, "day2": end_excl.day,
        "format": "onlycomma", "latlon": "no", "missing": "M",
        # IEM labels EDDM's :20 half-hourly routine reports as type 4
        # ("special"), so both types are needed to get all 48 per day.
        "direct": "no", "report_type": [3, 4],
    }
    r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
                     params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    if not r.text.startswith("station,"):          # e.g. "Too many requests ... slow down"
        raise RuntimeError(f"IEM returned no data: {r.text[:80]!r}")
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
    # Polymarket rule: SPECI (special) reports don't count, only routine METARs.
    rows = [x for x in rows if x.get("metarType") == "METAR"]
    if not rows:
        return pd.DataFrame(columns=["local_dt", "temp_c"])
    df = pd.DataFrame({
        "local_dt": pd.to_datetime([x.get("obsTime") for x in rows], unit="s", utc=True).tz_convert(LOCAL_TZ),
        "temp_c": [x.get("temp") for x in rows],
    })
    df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    return df[df["local_dt"].dt.strftime("%Y-%m-%d") == day.strftime("%Y-%m-%d")]


def fetch_daily_max(start, end, verbose=False):
    """Daily max (Highest-temperature markets). See fetch_daily_extreme."""
    return fetch_daily_extreme(start, end, "high", verbose)


def fetch_daily_min(start, end, verbose=False):
    """Daily min (Lowest-temperature markets). See fetch_daily_extreme."""
    return fetch_daily_extreme(start, end, "low", verbose)


def fetch_daily_extreme(start, end, kind="high", verbose=False):
    """{YYYY-MM-DD: whole-degree station max (kind='high') or min ('low')}
    for start..end (dates or 'YYYY-MM-DD' strings). Days with no data at
    all are left out, so callers can tell 'unknown' from a real value."""
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)

    result = {}
    # 1) AWC (official source, SPECI excluded) for days it still keeps.
    awc_horizon = date.today() - timedelta(days=14)
    d = max(start, awc_horizon)
    while d <= end:
        try:
            result.update(_daily_extreme(fetch_awc_day(d), kind))
        except Exception as e:
            print(f"[station] AWC fetch failed for {d}: {e}")
        d += timedelta(days=1)

    # 2) IEM archive for anything older or still missing.
    missing = [start + timedelta(days=i) for i in range((end - start).days + 1)
               if (start + timedelta(days=i)).strftime("%Y-%m-%d") not in result]
    if missing:
        try:
            iem = _daily_extreme(fetch_iem(min(missing), max(missing)), kind)
            for m in missing:
                k = m.strftime("%Y-%m-%d")
                if k in iem:
                    result[k] = iem[k]
        except Exception as e:
            print(f"[station] IEM fetch failed ({min(missing)}..{max(missing)}): {e}")

    if verbose:
        missing = (end - start).days + 1 - len(result)
        print(f"[station] {STATION} daily {'max' if kind == 'high' else 'min'}: "
              f"{len(result)} day(s) found, {missing} missing")
    return result


def observed_running_extreme(day: date, kind="high"):
    """
    For a day that is still in progress: the qualifying max (kind='high')
    or min ('low') so far (routine METARs only, local date, rounded half
    up), from AWC. Returns (whole_degrees, n_reports, last_report_local_time)
    or None if nothing has been reported yet. No completeness check here --
    the day isn't over, that's the point.
    """
    df = fetch_awc_day(day).dropna(subset=["temp_c"])
    if df.empty:
        return None
    v = df["temp_c"].max() if kind == "high" else df["temp_c"].min()
    return _round_half_up(v), len(df), df["local_dt"].max()


def observed_running_max(day: date):
    return observed_running_extreme(day, "high")


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
