"""
backtest/metar_history.py
-------------------------
Every EDDM METAR report since 1 March 2026, from the Iowa Environmental
Mesonet archive, for the same-day analysis.

Polymarket counts ROUTINE reports only (SPECI excluded). IEM doesn't label
SPECI for EDDM, but EDDM's routine reports are issued at HH:20 and HH:50
UTC (checked on the AWC feed, where every report carries metarType), so a
report is treated as routine when its minute is 20 or 50.

Output: backtest_data/metars.parquet
    t (unix s UTC), temp_c (float), minute, routine (bool), raw (METAR text)
"""
import io
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

from backtest.timeconv import to_unix_seconds

DATA = "backtest_data"
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
HEADERS = {"User-Agent": "munich-weather-bot backtest (research)"}


def fetch(start, end, tries=5):
    params = {"station": "EDDM", "data": ["tmpc", "metar"], "tz": "Etc/UTC",
              "year1": start.year, "month1": start.month, "day1": start.day,
              "year2": end.year, "month2": end.month, "day2": end.day,
              "format": "onlycomma", "latlon": "no", "missing": "M", "direct": "no",
              "report_type": [3, 4]}
    for k in range(tries):
        try:
            r = requests.get(URL, params=params, headers=HEADERS, timeout=(10, 120))
            if r.text.startswith("station,"):
                return pd.read_csv(io.StringIO(r.text), na_values=["M"])
            print(f"  IEM said: {r.text[:80]!r}")
        except requests.RequestException as e:
            print(f"  {type(e).__name__}")
        time.sleep(15 * (k + 1))
    raise RuntimeError("IEM download failed")


def main():
    start, end = date(2026, 3, 1), date.today()
    parts, d = [], start
    while d < end:                      # one month per request, gently
        nxt = min(d + timedelta(days=31), end)
        parts.append(fetch(d, nxt))
        print(f"{d}..{nxt}: {len(parts[-1])} reports")
        d = nxt
        time.sleep(3)
    raw = pd.concat(parts).drop_duplicates(subset=["valid", "metar"])
    t = to_unix_seconds(raw["valid"].astype(str) + "Z")
    df = pd.DataFrame({"t": t.to_numpy(), "temp_c": pd.to_numeric(raw["tmpc"], errors="coerce").to_numpy(),
                       "raw": raw["metar"].astype(str).to_numpy()})
    df["minute"] = pd.to_datetime(df["t"], unit="s").dt.minute
    df["routine"] = df["minute"].isin([20, 50])
    df = df.sort_values("t").drop_duplicates("t", keep="last")
    df.to_parquet(os.path.join(DATA, "metars.parquet"), index=False)
    print(f"metars.parquet: {len(df)} reports, {df['routine'].mean():.1%} routine (:20/:50); "
          f"minutes seen: {df['minute'].value_counts().head(6).to_dict()}")


if __name__ == "__main__":
    main()
