"""
backtest/leaders.py
-------------------
Who are the top weather traders and what do they do? Downloads, for the top
wallets of the weather leaderboard, every weather fill of the last DAYS days
(maker and taker side), plus the official outcome of every market they touched.

Sources (checked with the leaders / leaders2 probes, 26 Nov 2026):
  GET data-api.polymarket.com/v2/leaderboard?category=weather&time_period=day|week|month|all
      &sort_by=PNL|VOLUME&limit=..  -> {"data": [{"rank", "user_id", "user_name", "pnl", "volume"}], "pagination"}
  GET data-api.polymarket.com/v2/trades?user=<wallet>&limit=500[&taker_only=false]&cursor=..
      -> same row format as backtest/trades.py, plus "event_slug", "outcome", "title"
      taker_only=false adds the wallet's MAKER fills; maker = all minus taker rows.
  GET gamma-api.polymarket.com/events?slug=<event_slug> -> outcomePrices per bracket (winner = "1")

Outputs (backtest_data/leaders/):
  board.csv          leaderboard rows used (month top + all-time top)
  fills.parquet      wallet, name, role, side, token_id, condition_id, event_slug, outcome, size, price, timestamp
  payout.csv         token_id -> 1 if it won, 0 if it lost (resolved markets only)

Run:  python3 -m backtest.leaders
"""
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

DATA = os.path.join("backtest_data", "leaders")
H = {"User-Agent": "munich-weather-bot backtest (research)"}
DAYS = 30
N_MONTH, N_ALL = 20, 10


def get(url, params, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=H, timeout=(10, 60))
            if r.status_code == 429:
                time.sleep(5 * (k + 1)); continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"    retry {k + 1} {url.split('/')[-1]}: {type(e).__name__}")
            time.sleep(3 * (k + 1))
    return None


def board(period, n):
    j = get("https://data-api.polymarket.com/v2/leaderboard",
            {"category": "weather", "time_period": period, "sort_by": "PNL", "limit": n})
    return [dict(x, period=period) for x in (j or {}).get("data", [])]


def key(r):
    return (r["transaction_hash"], r["token_id"], r["side"], float(r["size"]), float(r["price"]))


def wallet_fills(wallet, cutoff, taker_only):
    rows, cursor = [], None
    for _ in range(400):
        params = {"user": wallet, "limit": 500, "taker_only": "true" if taker_only else "false"}
        if cursor:
            params["cursor"] = cursor
        j = get("https://data-api.polymarket.com/v2/trades", params)
        if j is None:
            return None
        page = j.get("data") or []
        if rows and page and key(page[0]) == key(rows[0]):
            print(f"    WARNING cursor ignored for {wallet[:10]}"); break
        rows += page
        pg = j.get("pagination") or {}
        cursor = pg.get("next_cursor")
        if not page or min(int(r["timestamp"]) for r in page) < cutoff or not pg.get("has_more") or not cursor:
            break
    return [r for r in rows if int(r["timestamp"]) >= cutoff and "temperature" in (r.get("event_slug") or "")]


def one_wallet(args):
    wallet, name, cutoff = args
    allr, taker = wallet_fills(wallet, cutoff, False), wallet_fills(wallet, cutoff, True)
    if allr is None or taker is None:
        print(f"  {name}: FAILED"); return []
    left = Counter(key(r) for r in taker)
    out = []
    for r in allr:
        k = key(r)
        role = "taker" if left[k] > 0 else "maker"
        if role == "taker":
            left[k] -= 1
        out.append((wallet, name, role, r["side"], str(r["token_id"]), r["condition_id"], r.get("event_slug"),
                    r.get("outcome"), float(r["size"]), float(r["price"]), int(r["timestamp"])))
    print(f"  {name}: {len(out)} weather fills ({sum(1 for o in out if o[2] == 'maker')} maker)")
    return out


def one_event(slug):
    j = get("https://gamma-api.polymarket.com/events", {"slug": slug})
    out = []
    if not j or not j[0].get("closed"):
        return out
    for m in j[0].get("markets", []):
        try:
            p, t = json.loads(m.get("outcomePrices") or "[]"), json.loads(m.get("clobTokenIds") or "[]")
        except ValueError:
            continue
        if len(p) == 2 and len(t) == 2 and p[0] in ("0", "1"):
            out += [(t[0], int(p[0] == "1")), (t[1], int(p[1] == "1"))]
    return out


def main():
    os.makedirs(DATA, exist_ok=True)
    b = board("month", N_MONTH) + board("all", N_ALL)
    bd = pd.DataFrame(b).drop_duplicates("user_id")
    bd.to_csv(os.path.join(DATA, "board.csv"), index=False)
    print(f"{len(bd)} wallets")
    cutoff = int(time.time()) - DAYS * 86400
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one_wallet, [(w, n, cutoff) for w, n in zip(bd.user_id, bd.user_name)]))
    fills = pd.DataFrame([r for x in res for r in x], columns=["wallet", "name", "role", "side", "token_id", "condition_id",
                                                               "event_slug", "outcome", "size", "price", "timestamp"])
    fills.to_parquet(os.path.join(DATA, "fills.parquet"), index=False)
    slugs = sorted(fills.event_slug.dropna().unique())
    print(f"{len(fills)} fills in {len(slugs)} events; fetching outcomes")
    with ThreadPoolExecutor(8) as ex:
        pay = [r for x in ex.map(one_event, slugs) for r in x]
    pd.DataFrame(pay, columns=["token_id", "payout"]).to_csv(os.path.join(DATA, "payout.csv"), index=False)
    print(f"{len(pay) // 2} resolved brackets")


if __name__ == "__main__":
    main()
