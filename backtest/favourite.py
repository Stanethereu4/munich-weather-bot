"""
backtest/favourite.py
---------------------
"Buy the market favourite and hold to resolution" (favourite-longshot bias test),
daily HIGHS only (low-market prices are incoherent).

At a fixed local time before the target day D, take the bracket with the highest
price, buy one share at price + HALF_SPREAD (+ taker fee), hold to the end:
    profit = won - cost.   Example: favourite at 0.45, cost 0.46 + 0.0025 fee;
    if it wins: +0.54, if not: -0.46.
Also: calibration by price bucket (do 40c brackets win 40% of the time?).

Data: backtest_data/cities (10 cities, 10-min prices, from 24 Aug)
      backtest_data/prices + markets.csv (Munich, 1-min prices, from March)
Run:  python3 -m backtest.favourite
"""
import glob, os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from backtest.cities import CITIES

HALF_SPREAD, FEE_RATE, STALE = 0.01, 0.01, 2 * 3600
TIMES = [("2 days before 10:00", -2, 10), ("day before 10:00", -1, 10), ("day before 20:00", -1, 20),
         ("same day 00:00", 0, 0), ("same day 10:00", 0, 10)]


def snapshots(mk, prices, tz):
    rows = []
    for d, b in mk.groupby("target_date"):
        for name, off, hour in TIMES:
            ts = int((datetime.fromisoformat(d) + timedelta(days=off)).replace(hour=hour, tzinfo=tz).timestamp())
            px = []
            for tok in b["yes_token"]:
                t, p = prices.get(tok, (None, None))
                if t is None: px.append(np.nan); continue
                i = np.searchsorted(t, ts, side="right") - 1
                px.append(p[i] if i >= 0 and ts - t[i] <= STALE else np.nan)
            px = np.array(px)
            if np.isnan(px).any() or not 0.9 <= px.sum() <= 1.1:
                continue
            for p, w, rank in zip(px, b["won"], (-px).argsort().argsort()):
                rows.append((d, name, p, int(w), int(rank)))
    return rows


def load_prices(paths):
    p = pd.concat([pd.read_parquet(x) for x in paths])
    return {tok: (g.t.to_numpy(), g.p.to_numpy(float)) for tok, g in p.sort_values("t").groupby("yes_token")}


def main():
    rows = []
    cm = pd.read_csv("backtest_data/cities/markets.csv", dtype={"yes_token": str})
    for city, g in cm[cm.kind == "high"].groupby("city"):
        pr = load_prices([f"backtest_data/cities/prices/{city}.parquet"])
        rows += [(city, "Aug-Nov") + r for r in snapshots(g, pr, ZoneInfo(CITIES[city][1]))]
    mm = pd.read_csv("backtest_data/markets.csv", dtype={"yes_token": str})
    mm = mm[(mm.kind == "high") & (mm.won.notna()) & (mm.target_date < "2026-08-24")]
    pr = load_prices(glob.glob("backtest_data/prices/highest-*.parquet"))
    rows += [("munich", "Mar-Aug") + r for r in snapshots(mm, pr, ZoneInfo("Europe/Berlin"))]
    df = pd.DataFrame(rows, columns=["city", "period", "date", "when", "p", "won", "rank"])
    df["cost"] = np.minimum(df.p + HALF_SPREAD, 0.999)
    df["cost"] += FEE_RATE * df.cost * (1 - df.cost)
    df["pnl"] = df.won - df.cost

    fav = df[df["rank"] == 0]
    t = lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    print("BUY THE FAVOURITE, hold to resolution (1 share per market-day)")
    out = fav.groupby("when", sort=False).agg(days=("pnl", "size"), avg_price=("p", "mean"), win_rate=("won", "mean"),
                                               profit_per_bet=("pnl", "mean"), total=("pnl", "sum"), t=("pnl", t))
    print(out.reindex([x[0] for x in TIMES]).round(3).to_string())
    print("\nper city, day before 10:00:")
    c = fav[fav.when == "day before 10:00"].groupby(["city", "period"]).agg(
        days=("pnl", "size"), avg_price=("p", "mean"), win_rate=("won", "mean"), profit_per_bet=("pnl", "mean"), t=("pnl", t))
    print(c.round(3).to_string())
    print("\nCALIBRATION, all brackets, day before 10:00 (does a 40c bracket win 40% of the time?)")
    x = df[df.when == "day before 10:00"].copy()
    x["bucket"] = pd.cut(x.p, [0, .02, .05, .1, .2, .3, .4, .5, .6, .8, 1])
    cal = x.groupby("bucket", observed=True).agg(n=("won", "size"), avg_price=("p", "mean"), win_rate=("won", "mean"))
    cal["win - price"] = cal.win_rate - cal.avg_price
    cal["se"] = np.sqrt(cal.avg_price * (1 - cal.avg_price) / cal.n)
    print(cal.round(3).to_string())


if __name__ == "__main__":
    main()
