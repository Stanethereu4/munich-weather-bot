"""
backtest/telonex_depth.py
-------------------------
How much of each Kalshi/Polymarket opportunity could really have been bought?
Uses the real Polymarket order books (Telonex book_snapshot_full, backtest/telonex_books.py)
for the downloaded legs. Other legs as before:
  * Kalshi legs: real best ask from the 1-minute candles (size unknown: assumed not limiting)
  * cheap Polymarket legs not downloaded: mid + 1c (assumed not limiting)

For N pairs we walk up each Polymarket book (cheapest asks first) and pay the average price
for N shares, plus the 0.01*p*(1-p) fee; Kalshi legs ask + 0.07*p*(1-p).
    profit(N) = N * (payout_min $1 - cost per pair at size N)
Example: a leg whose book is 20 shares at 0.44 then 50 at 0.46: 30 shares cost 20*0.44 + 10*0.46.

Run:  python3 -m backtest.telonex_depth
"""
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from backtest.kalshi_fills import load, load_floor, signals, validate
from backtest.kalshi_test import K_FEE, PM_FEE

DIR = os.path.join("backtest_data", "telonex")
SIZES = [1, 10, 25, 50, 100, 250, 500, 1000]


def book_at(token, t_s, day):
    """Last Polymarket snapshot at or before t_s for this YES token: asks as [(price, size)] cheapest first."""
    path = glob.glob(os.path.join(DIR, f"{token[:16]}_{day}.parquet"))[0]
    tus = t_s * 1_000_000
    d = ds.dataset(path)
    tb = d.to_table(columns=["timestamp_us", "asks"],
                    filter=(pc.field("timestamp_us") <= tus) & (pc.field("timestamp_us") >= tus - 1800 * 1_000_000))
    if tb.num_rows == 0:
        return None, None
    i = int(np.argmax(tb.column("timestamp_us").to_numpy()))
    row = tb.slice(i, 1).to_pylist()[0]
    asks = sorted((float(a["price"]), float(a["size"])) for a in row["asks"])
    return asks, (tus - row["timestamp_us"]) / 1e6


def cost_for(asks, n):
    """Average price to buy n shares walking up the book; None if the book is too thin."""
    left, spent = n, 0.0
    for p, s in asks:
        q = min(left, s)
        spent += q * (p + PM_FEE * p * (1 - p))
        left -= q
        if left <= 1e-9:
            return spent / n
    return None


def main():
    km, pm = load()
    ob, fl = load_floor()
    valid = validate(ob, pm)
    km = km[pd.to_datetime(km.date) >= "2026-11-01"]; pm = pm[pd.to_datetime(pm.date) >= "2026-11-01"]
    targets = pd.read_csv(os.path.join("backtest", "telonex_targets.csv"), dtype={"token": str})
    have = set(targets.token)
    for (city, date, utc), g in targets.groupby(["city", "target_date", "utc_time"], sort=False):
        t0 = int(pd.Timestamp(utc).timestamp())
        s = [x for x in signals(km[(km.city == city) & (km.date == date)], pm[(pm.city == city) & (pm.date == date)],
                                thr=0.01, fl=fl, valid_days=valid) if x["t"] == t0][0]
        day = utc[:10]
        print(f"\n=== {city}, high of {date}, opportunity at {utc} (paper edge {s['edge']*100:.1f}c, floor {s['floor']}°F, "
              f"Kalshi >= {s['X']}°F + Polymarket below {s['Y']}°F)")
        fixed = 0.0
        books = []
        for plat, key, price, _ in s["legs"]:
            if plat == "K":
                fixed += price + K_FEE * price * (1 - price)
                print(f"   Kalshi leg ask {price:.2f} (real best ask, size unknown)")
            elif key in have:
                asks, age = book_at(key, t0, day)
                books.append(asks)
                top = ", ".join(f"{q:.0f}@{p:.3f}" for p, q in asks[:4])
                print(f"   Polymarket leg (assumed {price:.3f}): REAL book {age:.0f}s old -> best asks {top}")
            else:
                fixed += price + PM_FEE * price * (1 - price)
                print(f"   Polymarket leg {price:.3f} (cheap, not downloaded: mid + 1c)")
        print(f"   {'pairs':>6} {'cost per pair':>14} {'profit per pair':>16} {'total profit':>13}")
        for n in SIZES:
            parts = [cost_for(b, n) for b in books]
            if any(p is None for p in parts):
                print(f"   {n:>6}  book too thin"); break
            c = fixed + sum(parts)
            print(f"   {n:>6} {c:>14.3f} {100*(1-c):>14.1f}c {n*(1-c):>12.2f}$")


if __name__ == "__main__":
    main()
