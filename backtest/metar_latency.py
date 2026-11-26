"""
backtest/metar_latency.py
-------------------------
Same-day speed: when a new EDDM report rules out a bracket of today's HIGH
market (the max already went above it), how fast do traders sell that
bracket, and how does that compare with when the report became available?

Report availability (metar-timing probe, 26 Nov 2026):
    obsTime        the :20 / :50 observation
    receiptTime    aviationweather.gov (AWC) received it, ~6 min after obs
    NOAA tgftp     raw text file updated ~3.5 min after obs (live only, no history)

Download (network):  python3 -m backtest.metar_latency download
    AWC /api/data/metar?ids=EDDM&format=json&hours=360 -> backtest_data/metar_receipts.csv
Analysis (local):    python3 -m backtest.metar_latency
    needs backtest_data/trades (backtest/trades.py) and markets.csv

"Stale" fill = after the report, someone sold YES (or bought NO) of a
ruled-out bracket at a YES price above 3 cents. The taker on the other side
of a still-standing bid earns that price per share, risk-free.
Example: max reaches 6°C, the "5°C" bracket can no longer win; a stale bid at
0.25 for 200 shares gives the seller 200 * 0.25 = $50.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import requests

from munich_bot import parse_label_to_bounds

DATA = "backtest_data"
STALE = 0.03


def download():
    r = requests.get("https://aviationweather.gov/api/data/metar",
                     params={"ids": "EDDM", "format": "json", "hours": 360}, timeout=60)
    r.raise_for_status()
    rows = [(x["obsTime"], x["receiptTime"], x.get("metarType"), x.get("temp"), x.get("rawOb")) for x in r.json()]
    df = pd.DataFrame(rows, columns=["obs_ts", "receipt", "type", "temp", "raw"])
    df["receipt_ts"] = (pd.to_datetime(df["receipt"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")
    df.to_csv(os.path.join(DATA, "metar_receipts.csv"), index=False)
    print(f"{len(df)} reports, {pd.to_datetime(df.obs_ts.min(), unit='s')} .. {pd.to_datetime(df.obs_ts.max(), unit='s')}")
    print(f"receipt delay after obs (min): median {((df.receipt_ts - df.obs_ts) / 60).median():.1f}")


def analyse():
    m = pd.read_csv(os.path.join(DATA, "metar_receipts.csv"))
    m = m[m["type"] == "METAR"].copy()                       # routine reports only (SPECI don't count)
    m["t"] = np.floor(m["temp"].astype(float) + 0.5)          # whole degrees, half rounded up
    m["date"] = pd.to_datetime(m["obs_ts"], unit="s", utc=True).dt.tz_convert("Europe/Berlin").dt.strftime("%Y-%m-%d")
    m = m.sort_values("obs_ts")
    mk = pd.read_csv(os.path.join(DATA, "markets.csv"), dtype={"yes_token": str, "no_token": str})
    mk = mk[mk["kind"] == "high"]

    events, fills = [], []
    for d, g in m.groupby("date"):
        b = mk[mk.target_date == d]
        path = os.path.join(DATA, "trades", f"{b.event_slug.iloc[0]}.parquet") if len(b) else None
        if not path or not os.path.exists(path):
            continue
        tr = pd.read_parquet(path)
        run_max = -99
        for r in g.itertuples():
            new_max = max(run_max, r.t)
            for br in b.itertuples():
                lo, hi = parse_label_to_bounds(br.bracket_label)
                if hi is None or not (run_max < hi < new_max):     # newly ruled out by THIS report
                    continue
                yes = tr[tr.token_id == br.yes_token].assign(yes_px=lambda x: x.price,
                                                             sells_yes=lambda x: x.side == "SELL")
                no = tr[tr.token_id == br.no_token].assign(yes_px=lambda x: 1 - x.price,
                                                           sells_yes=lambda x: x.side == "BUY")
                f = pd.concat([yes, no])
                f = f[(f.role == "taker") & f.sells_yes & (f.yes_px > STALE) &
                      (f.timestamp >= r.obs_ts - 600) & (f.timestamp <= r.obs_ts + 3600)]
                f = f.assign(dt_obs=(f.timestamp - r.obs_ts) / 60, dt_awc=(f.timestamp - r.receipt_ts) / 60,
                             usd=f["size"] * f.yes_px, report=r.obs_ts, bracket=br.bracket_label, date=d)
                fills.append(f)
                events.append((d, pd.to_datetime(r.obs_ts, unit="s").strftime("%H:%M"), br.bracket_label,
                               (r.receipt_ts - r.obs_ts) / 60, len(f), f.usd.sum(),
                               f.dt_obs.min() if len(f) else np.nan, f.dt_obs.max() if len(f) else np.nan))
            run_max = new_max
    ev = pd.DataFrame(events, columns=["date", "obs UTC", "ruled out", "AWC delay", "stale fills", "$ captured",
                                       "first fill (min after obs)", "last fill"])
    f = pd.concat(fills) if fills else pd.DataFrame()
    pd.set_option("display.width", 200)
    print(f"{len(ev)} bracket rule-outs on {ev.date.nunique()} days; reports reached AWC a median "
          f"{ev['AWC delay'].median():.1f} min after the observation\n")
    print(ev.round(1).to_string(index=False))
    if f.empty:
        return
    print(f"\nTOTAL stale money taken: ${f.usd.sum():,.0f} in {len(f)} fills")
    bins = [-10, 0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 30, 60]
    f["window"] = pd.cut(f.dt_obs, bins)
    w = f.groupby("window", observed=True).agg(fills=("usd", "size"), usd=("usd", "sum"))
    w["share $"] = w.usd / w.usd.sum()
    print("\nWhen the stale money was taken (minutes after the observation time):")
    print(w.round(2).to_string())
    print(f"\nshare of $ taken BEFORE AWC had the report: {(f.loc[f.dt_awc < 0, 'usd'].sum() / f.usd.sum()):.0%}; "
          f"before obs + 3.5 min (NOAA file): {(f.loc[f.dt_obs < 3.5, 'usd'].sum() / f.usd.sum()):.0%}")
    who = f.groupby(["wallet"]).agg(name=("name", "first"), usd=("usd", "sum"), fills=("usd", "size"),
                                    median_min_after_obs=("dt_obs", "median"))
    print("\nWho took it:")
    print(who.sort_values("usd", ascending=False).head(8).round(1).to_string())


if __name__ == "__main__":
    download() if len(sys.argv) > 1 and sys.argv[1] == "download" else analyse()
