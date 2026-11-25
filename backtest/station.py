"""
backtest/station.py
-------------------
EDDM daily max and min (whole °C, local day, routine METARs per Polymarket's
rules -- see station_obs.py) for every day of the backtest period. Used to
fit the blend weights without look-ahead and to check the model's errors.

Output: backtest_data/station.csv  (date, high, low)
"""
import os
from datetime import date, timedelta

import pandas as pd

from station_obs import fetch_daily_extreme

DATA = "backtest_data"


def main():
    markets = pd.read_csv(os.path.join(DATA, "markets.csv"))
    start = date(2026, 2, 1)                      # 30-day blend window before the first runs
    end = date.fromisoformat(markets["target_date"].max())
    high = fetch_daily_extreme(start, end, "high", verbose=True)
    low = fetch_daily_extreme(start, end, "low", verbose=True)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    df = pd.DataFrame({"date": [d.isoformat() for d in days]})
    df["high"] = df["date"].map(high)
    df["low"] = df["date"].map(low)
    df.to_csv(os.path.join(DATA, "station.csv"), index=False)
    print(f"station.csv: {len(df)} days, {df['high'].notna().sum()} with max, {df['low'].notna().sum()} with min")


if __name__ == "__main__":
    main()
