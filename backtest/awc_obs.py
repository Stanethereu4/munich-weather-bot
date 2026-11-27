"""
backtest/awc_obs.py
-------------------
Downloads the routine airport reports for the 7 same-airport US cities from
aviationweather.gov: the SAME service Polymarket names as its resolution source
(e.g. https://aviationweather.gov/api/data/metar?ids=KDEN&format=raw&date=2026-11-15T10:00:00Z&hours=30).
We use format=json from the same endpoint because it also gives receiptTime
(when the service received the report = when it became public).
History available (awc-history probe, 27 Nov 2026): back to 1 Nov 2026; 28 Oct returns nothing.

Parsing follows Polymarket's written rules exactly:
  * only routine reports: a line beginning with SPECI does not count
  * temperature = hourly remark T + 8 digits (sign digit, 3 digits of tenths °C), e.g. T01390133 = 13.9°C;
    if a routine report has no remark, the body group (e.g. 18/12, M11/M17) in whole °C
  * °F = °C * 9/5 + 32, rounded to the nearest whole degree, exact halves rounded up
  * a report counts for the local calendar date of its time stamp

Output: backtest_data/kalshi/awc_obs.csv  (icao, obs_ts, receipt_ts, raw, routine, temp_c, temp_f, local_date)
Run:    python3 -m backtest.awc_obs
"""
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

OUT = os.path.join("backtest_data", "kalshi")
H = {"User-Agent": "munich-weather-bot backtest (research)"}
STATIONS = {   # Polymarket city -> (ICAO in its rules, local time zone)
    "austin": ("KAUS", "America/Chicago"), "denver": ("KDEN", "America/Denver"),
    "los-angeles": ("KLAX", "America/Los_Angeles"), "miami": ("KMIA", "America/New_York"),
    "houston": ("KHOU", "America/Chicago"), "atlanta": ("KATL", "America/New_York"),
    "san-francisco": ("KSFO", "America/Los_Angeles"),
}
FIRST = datetime(2026, 11, 1, tzinfo=timezone.utc)


def temp_c(raw):
    """Polymarket rule: hourly remark T sddd (tenths °C) if present, else body group in whole °C."""
    m = re.search(r"\bT([01])(\d{3})([01])(\d{3})\b", raw)
    if m:
        v = int(m.group(2)) / 10
        return -v if m.group(1) == "1" else v
    m = re.search(r"\s(M?\d{2})/(M?\d{2})?\s", raw)
    if m:
        t = m.group(1)
        return -int(t[1:]) if t.startswith("M") else int(t)
    return None


def to_f(c):
    """°F rounded to nearest whole degree, exact halves rounded up (e.g. -8.5 -> -8)."""
    f = round(c * 9 / 5 + 32, 6)       # strip float noise before rounding
    return math.floor(f + 0.5)


def main():
    rows = []
    now = datetime.now(timezone.utc)
    for city, (icao, tz) in STATIONS.items():
        t = FIRST + timedelta(hours=30)
        while t < now + timedelta(hours=30):
            for k in range(5):
                try:
                    r = requests.get("https://aviationweather.gov/api/data/metar",
                                     params={"ids": icao, "format": "json", "date": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                             "hours": 30}, headers=H, timeout=30)
                    j = r.json() if r.status_code == 200 else []
                    break
                except (requests.RequestException, ValueError):
                    time.sleep(3 * (k + 1)); j = []
            for x in j:
                raw = x.get("rawOb") or ""
                routine = not raw.lstrip().startswith("SPECI")
                c = temp_c(raw)
                rows.append((city, icao, int(x["obsTime"]),
                             int(pd.Timestamp(x["receiptTime"]).timestamp()) if x.get("receiptTime") else None,
                             raw, routine, c, to_f(c) if c is not None else None,
                             datetime.fromtimestamp(int(x["obsTime"]), ZoneInfo(tz)).strftime("%Y-%m-%d")))
            t += timedelta(hours=24)
        print(f"{icao}: done")
    df = pd.DataFrame(rows, columns=["city", "icao", "obs_ts", "receipt_ts", "raw", "routine", "temp_c", "temp_f",
                                     "local_date"]).drop_duplicates(["icao", "obs_ts", "raw"])
    df.to_csv(os.path.join(OUT, "awc_obs.csv"), index=False)
    has_remark = df.raw.str.contains(r" T[01]\d{7}", regex=True).mean()
    print(f"{len(df)} reports, {df.routine.mean():.0%} routine, {df.temp_c.notna().mean():.0%} with a temperature, "
          f"{has_remark:.0%} with the hourly remark")


if __name__ == "__main__":
    main()
