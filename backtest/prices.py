"""
backtest/prices.py  (Part C2)
-----------------------------
Downloads the 1-minute price history of every bracket's YES token over the
life of its market, for every market in backtest_data/markets.csv.

Source (chosen after the 25 Nov probes):
    CLOB  GET https://clob.polymarket.com/prices-history
          ?market=<YES token>&startTs=<unix s>&endTs=<unix s>&fidelity=1
    -> {"history": [{"t": unix_seconds, "p": price}, ...]}, one point per minute,
       available for resolved markets back to at least July 2026.
    The value is NOT the last trade price (it differs from fills); it behaves
    like a per-minute midpoint snapshot. Treated as "market probability".
    Asking with interval=max instead gives only 10-minute points.

Output: backtest_data/prices/<event_slug>.parquet with columns
    yes_token (str), bracket_label (str), t (int, unix s UTC), p (float32)
One file per market (all its brackets). Markets already downloaded are
skipped, so the script can be re-run after an interruption.

Run:  python3 -m backtest.prices
"""

import os
import time
from datetime import datetime

import pandas as pd
import requests

CLOB = "https://clob.polymarket.com/prices-history"
HEADERS = {"User-Agent": "munich-weather-bot backtest (research)"}
DATA = "backtest_data"
OUT = os.path.join(DATA, "prices")
MAX_MINUTES = 280      # stop before the 300-min job limit; re-run to continue


def to_ts(iso):
    return int(datetime.fromisoformat(str(iso).replace("Z", "+00:00").replace(" ", "T")).timestamp())


def fetch_history(token, start, end, tries=5):
    for k in range(tries):
        try:
            r = requests.get(CLOB, params={"market": token, "startTs": start, "endTs": end, "fidelity": 1},
                             headers=HEADERS, timeout=(10, 60))
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 10)))
                continue
            r.raise_for_status()
            return r.json().get("history", [])
        except requests.RequestException as e:
            print(f"    retry {k + 1} for token {token[:10]}...: {type(e).__name__}")
            time.sleep(3 * (k + 1))
    return None          # give up on this token for now; next run retries the market


def main():
    os.makedirs(OUT, exist_ok=True)
    markets = pd.read_csv(os.path.join(DATA, "markets.csv"), dtype={"yes_token": str, "no_token": str})
    events = list(markets.groupby("event_slug"))
    t_start = time.time()
    done = skipped = failed = 0

    for slug, rows in events:
        path = os.path.join(OUT, f"{slug}.parquet")
        if os.path.exists(path):
            skipped += 1
            continue
        if (time.time() - t_start) / 60 > MAX_MINUTES:
            print("time budget reached -- re-run to continue")
            break

        parts, ok = [], True
        for _, m in rows.iterrows():
            start = to_ts(m["market_start"]) - 3600
            end = to_ts(m["closed_time"]) + 3600 if pd.notna(m["closed_time"]) else start + 5 * 86400
            hist = fetch_history(m["yes_token"], start, end)
            if hist is None:
                ok = False
                break
            if hist:
                df = pd.DataFrame(hist)
                df["yes_token"] = m["yes_token"]
                df["bracket_label"] = m["bracket_label"]
                parts.append(df)
            time.sleep(0.1)
        if not ok:
            failed += 1
            print(f"  {slug}: incomplete, will retry next run")
            continue

        out = pd.concat(parts) if parts else pd.DataFrame(columns=["t", "p", "yes_token", "bracket_label"])
        out = out[["yes_token", "bracket_label", "t", "p"]].astype({"t": "int64", "p": "float32"})
        out.to_parquet(path, compression="zstd", index=False)
        done += 1
        if done % 20 == 0:
            print(f"  {done} markets downloaded ({(time.time() - t_start) / 60:.0f} min)")

    print(f"\nprices: {done} new markets, {skipped} already present, {failed} incomplete "
          f"(of {len(events)}) in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
