"""
backtest/trades.py
------------------
Downloads every fill (both sides) of the Munich high/low markets, with the
wallet that made it and whether it was the TAKER (crossed the spread) or the
MAKER (had a resting order that got hit).

Source (checked with the trades-shape / trades-shape2 probes, 26 Nov 2026):
    GET https://data-api.polymarket.com/v2/trades?condition=<conditionId>&limit=500[&taker_only=false]
    -> {"data": [{"proxy_wallet", "side": "BUY"/"SELL", "token_id", "condition_id",
                  "size", "price", "timestamp", "outcome", "name", "pseudonym",
                  "transaction_hash", ...}],
        "pagination": {"limit", "offset", "has_more", "next_cursor"}}
    Default (taker_only=true) returns only the taker's side of each fill.
    taker_only=false also returns the makers' sides. So:
        maker rows = all rows minus taker rows (matched on tx hash, wallet, token, side, size, price).

Output: backtest_data/trades/<event_slug>.parquet
    wallet, name, role (taker/maker), side, token_id, condition_id, size, price, timestamp, tx
Markets already downloaded are skipped (re-run to continue).

Run:  python3 -m backtest.trades [first_date]      (default first_date 2026-08-24)
"""
import os
import sys
import time
from collections import Counter

import pandas as pd
import requests

URL = "https://data-api.polymarket.com/v2/trades"
H = {"User-Agent": "munich-weather-bot backtest (research)"}
DATA = "backtest_data"
OUT = os.path.join(DATA, "trades")
MAX_MINUTES = 280


def fetch_all(cond, taker_only):
    """All fills of one bracket market, following the cursor."""
    rows, cursor = [], None
    for _ in range(200):
        params = {"condition": cond, "limit": 500, "taker_only": "true" if taker_only else "false"}
        if cursor:
            params["cursor"] = cursor
        for k in range(5):
            try:
                r = requests.get(URL, params=params, headers=H, timeout=(10, 60))
                if r.status_code == 429:
                    time.sleep(5 * (k + 1)); continue
                r.raise_for_status()
                j = r.json()
                break
            except requests.RequestException as e:
                print(f"    retry {k + 1}: {type(e).__name__}"); time.sleep(3 * (k + 1))
        else:
            return None
        page = j.get("data") or []
        if rows and page and key(page[0]) == key(rows[0]):
            print(f"    WARNING: cursor ignored for {cond[:12]}.. (same page again) -- stopping at {len(rows)} rows")
            break
        rows += page
        pg = j.get("pagination") or {}
        cursor = pg.get("next_cursor")
        if not pg.get("has_more") or not cursor:
            break
    return rows


def key(r):
    return (r["transaction_hash"], r["proxy_wallet"], r["token_id"], r["side"], float(r["size"]), float(r["price"]))


def main():
    first = sys.argv[1] if len(sys.argv) > 1 else "2026-08-24"
    os.makedirs(OUT, exist_ok=True)
    markets = pd.read_csv(os.path.join(DATA, "markets.csv"))
    markets = markets[markets["target_date"] >= first]
    t0 = time.time()
    done = 0
    for slug, g in markets.groupby("event_slug"):
        path = os.path.join(OUT, f"{slug}.parquet")
        if os.path.exists(path):
            continue
        if (time.time() - t0) / 60 > MAX_MINUTES:
            print("time budget reached -- re-run to continue"); break
        out, ok = [], True
        for cond in g["condition_id"]:
            allr, taker = fetch_all(cond, False), fetch_all(cond, True)
            if allr is None or taker is None:
                ok = False; break
            left = Counter(key(r) for r in taker)     # taker rows to tick off
            for r in allr:
                k = key(r)
                role = "taker" if left[k] > 0 else "maker"
                if role == "taker":
                    left[k] -= 1
                out.append((r["proxy_wallet"], r.get("name") or r.get("pseudonym"), role, r["side"], str(r["token_id"]),
                            r["condition_id"], float(r["size"]), float(r["price"]), int(r["timestamp"]), r["transaction_hash"]))
        if not ok:
            print(f"  {slug}: failed, will retry next run"); continue
        pd.DataFrame(out, columns=["wallet", "name", "role", "side", "token_id", "condition_id", "size", "price",
                                   "timestamp", "tx"]).to_parquet(path, index=False)
        done += 1
        if done % 20 == 0:
            print(f"  {done} markets saved ({(time.time() - t0) / 60:.1f} min)")
    have = len([f for f in os.listdir(OUT) if f.endswith(".parquet")])
    print(f"saved {done} new markets this run; {have} in total")


if __name__ == "__main__":
    main()
