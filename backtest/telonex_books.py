"""
backtest/telonex_books.py
-------------------------
Real Polymarket order books (Telonex) for the live Polymarket legs of the best
Kalshi/Polymarket opportunities (listed in backtest/telonex_targets.csv).

Telonex API (read from their official Python package telonex 0.4.0, 27 Nov 2026):
  availability (public, free, no key):  GET https://api.telonex.io/v1/availability/polymarket?asset_id=<token>
  download (needs key, uses 1 of the 5 free-trial downloads per file):
      GET https://api.telonex.io/v1/downloads/polymarket/<channel>/<YYYY-MM-DD>?asset_id=<token>
      header  Authorization: Bearer <TELONEX_API_KEY>
      -> redirect to a Parquet file (one day of one token); header X-Downloads-Remaining
      401 = bad key, 403 = no downloads left
Steps:
  python3 -m backtest.telonex_books availability   free; checks every target before spending downloads
  python3 -m backtest.telonex_books download       spends one download per target (5)
Files: backtest_data/telonex/<token-prefix>_<date>.parquet ; the first file's columns and rows are printed
so the analysis can be written against the real structure.
"""
import os
import sys

import pandas as pd
import requests

BASE = "https://api.telonex.io"
CHANNEL = "book_snapshot_full"
OUT = os.path.join("backtest_data", "telonex")
TARGETS = os.path.join("backtest", "telonex_targets.csv")


def availability():
    t = pd.read_csv(TARGETS, dtype={"token": str})
    for r in t.itertuples():
        a = requests.get(f"{BASE}/v1/availability/polymarket", params={"asset_id": r.token}, timeout=30)
        print(f"{r.city} {r.target_date} leg {r.assumed}: {a.status_code} {a.text[:400]}")


def download():
    key = os.environ.get("TELONEX_API_KEY")
    if not key:
        sys.exit("TELONEX_API_KEY is not set (GitHub secret missing?)")
    os.makedirs(OUT, exist_ok=True)
    t = pd.read_csv(TARGETS, dtype={"token": str})
    for i, r in enumerate(t.itertuples()):
        day = r.utc_time[:10]
        path = os.path.join(OUT, f"{r.token[:16]}_{day}.parquet")
        if os.path.exists(path):
            print(f"already have {path}"); continue
        resp = requests.get(f"{BASE}/v1/downloads/polymarket/{CHANNEL}/{day}", params={"asset_id": r.token},
                            headers={"Authorization": f"Bearer {key}"}, timeout=300, allow_redirects=True)
        left = resp.history[0].headers.get("X-Downloads-Remaining") if resp.history else resp.headers.get("X-Downloads-Remaining")
        print(f"{r.city} {day} leg {r.assumed}: status {resp.status_code}, {len(resp.content)} bytes, downloads remaining {left}")
        if resp.status_code != 200:
            print("   ", resp.text[:300])
            if resp.status_code in (401, 403):
                break
            continue
        with open(path, "wb") as fh:
            fh.write(resp.content)
        if i == 0:
            df = pd.read_parquet(path)
            print("   columns:", list(df.columns), "| rows:", len(df))
            with pd.option_context("display.width", 250, "display.max_colwidth", 200):
                print(df.head(3).T.to_string()[:3000])


if __name__ == "__main__":
    {"availability": availability, "download": download}[sys.argv[1]]()
