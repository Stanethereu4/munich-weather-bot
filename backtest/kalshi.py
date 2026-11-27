"""
backtest/kalshi.py
------------------
Downloads 30 days of daily-HIGH markets for US cities on BOTH Kalshi and
Polymarket, to test cross-platform arbitrage (backtest/kalshi_test.py).

Kalshi public API (checked with the kalshi / kalshi-map probes, 27 Nov 2026; no login needed):
  GET /trade-api/v2/markets?series_ticker=KXHIGHNY&status=settled&limit=..&cursor=..
      -> {"markets": [{"ticker": "KXHIGHNY-26NOV26-B62.5", "event_ticker", "floor_strike", "cap_strike",
                       "strike_type": "between"|"greater"|"less", "result": "yes"|"no", "open_time", "close_time",
                       "rules_primary", ...}], "cursor"}
      strikes: between 62-63 = 62°F to 63°F ; greater 63 = 64°F or above ; less 56 = 55°F or below
      settlement: NWS daily climate report (e.g. CLINYC) via The Weather Company, whole °F
  GET /trade-api/v2/series/<series>/markets/<ticker>/candlesticks?start_ts&end_ts&period_interval=1
      -> {"candlesticks": [{"end_period_ts", "yes_bid": {"close_dollars"}, "yes_ask": {"close_dollars"},
                            "price": {"close_dollars"}, "volume_fp"}]}   (max 5000 candles per call)
  GET /trade-api/v2/markets/trades?ticker=..&limit=1000&cursor=..
      -> {"trades": [{"created_time", "yes_price_dollars", "count_fp", "taker_side"}], "cursor"}

Polymarket: Gamma events by slug (as backtest/cities.py) + CLOB /prices-history fidelity=1 (1-min).

Outputs: backtest_data/kalshi/{k_markets.csv, k_candles.parquet, k_trades.parquet, pm_markets.csv, pm_prices.parquet}
Run:  python3 -m backtest.kalshi
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

K = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com/prices-history"
H = {"User-Agent": "munich-weather-bot backtest (research)"}
OUT = os.path.join("backtest_data", "kalshi")
DAYS = 30

# Polymarket city slug -> (Kalshi series, same airport on both platforms?)
PAIRS = {
    "austin":        ("KXHIGHAUS",  True),    # Austin-Bergstrom
    "denver":        ("KXHIGHDEN",  True),    # Denver Intl
    "los-angeles":   ("KXHIGHLAX",  True),    # LAX
    "miami":         ("KXHIGHMIA",  True),    # Miami Intl
    "houston":       ("KXHIGHTHOU", True),    # Hobby
    "atlanta":       ("KXHIGHTATL", True),    # Hartsfield-Jackson
    "san-francisco": ("KXHIGHTSFO", True),    # SFO
    "nyc":           ("KXHIGHNY",   False),   # Kalshi Central Park vs Polymarket LaGuardia
    "chicago":       ("KXHIGHCHI",  False),   # Midway vs O'Hare
    "dallas":        ("KXHIGHTDAL", False),   # DFW vs Love Field
}


def get(url, params, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=H, timeout=(10, 60))
            if r.status_code == 429:
                time.sleep(3 * (k + 1)); continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if k == tries - 1:
                print(f"    FAILED {url.split('/')[-1][:40]}: {type(e).__name__}")
            time.sleep(2 * (k + 1))
    return None


def ts(s):
    return int(pd.Timestamp(s).timestamp())


# ------------------------------------------------------------------ Kalshi --
def k_markets(series, since):
    out, cursor = [], None
    for status in ("settled", "finalized", "closed"):
        cursor = None
        for _ in range(50):
            p = {"series_ticker": series, "status": status, "limit": 200}
            if cursor:
                p["cursor"] = cursor
            j = get(f"{K}/markets", p)
            if not j:
                break
            ms = j.get("markets", [])
            out += [m for m in ms if ts(m["close_time"]) >= since]
            cursor = j.get("cursor")
            if not cursor or not ms or min(ts(m["close_time"]) for m in ms) < since:
                break
    return list({m["ticker"]: m for m in out}.values())


def k_candles(args):
    series, m = args
    start, end = ts(m["open_time"]), ts(m["close_time"])
    rows = []
    for a in range(start, end, 4900 * 60):
        j = get(f"{K}/series/{series}/markets/{m['ticker']}/candlesticks",
                {"start_ts": a, "end_ts": min(a + 4900 * 60, end), "period_interval": 1})
        for c in (j or {}).get("candlesticks", []):
            f = lambda k: float((c.get(k) or {}).get("close_dollars") or "nan")
            rows.append((m["ticker"], int(c["end_period_ts"]), f("yes_bid"), f("yes_ask"), f("price"),
                         float(c.get("volume_fp") or 0)))
    return rows


def k_trades(ticker):
    rows, cursor = [], None
    for _ in range(100):
        p = {"ticker": ticker, "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        j = get(f"{K}/markets/trades", p)
        if not j:
            break
        for t in j.get("trades", []):
            rows.append((ticker, ts(t["created_time"]), float(t["yes_price_dollars"]), float(t["count_fp"]),
                         t.get("taker_side")))
        cursor = j.get("cursor")
        if not cursor or not j.get("trades"):
            break
    return rows


# -------------------------------------------------------------- Polymarket --
def pm_event(args):
    city, d = args
    slug = f"highest-temperature-in-{city}-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"
    j = get(f"{GAMMA}/events", {"slug": slug})
    if not j or not j[0].get("closed"):
        return []
    rows = []
    for m in j[0].get("markets", []):
        try:
            p, t = json.loads(m.get("outcomePrices") or "[]"), json.loads(m.get("clobTokenIds") or "[]")
        except ValueError:
            continue
        if len(p) == 2 and len(t) == 2 and p[0] in ("0", "1"):
            rows.append((city, d.isoformat(), slug, m.get("groupItemTitle"), t[0], t[1], int(p[0] == "1"),
                         m.get("startDate"), m.get("closedTime")))
    return rows


def pm_history(args):
    tok, start, end = args
    out = []
    for a in range(start, end, 7 * 86400):              # 1-minute data in weekly chunks
        j = get(CLOB, {"market": tok, "startTs": a, "endTs": min(a + 7 * 86400, end), "fidelity": 1})
        out += [(tok, h["t"], h["p"]) for h in (j or {}).get("history", [])]
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    since = int(time.time()) - (DAYS + 1) * 86400
    # Kalshi
    km = []
    for city, (series, same) in PAIRS.items():
        ms = k_markets(series, since)
        for m in ms:
            ev = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", m["event_ticker"])
            d = datetime.strptime(ev.group(0)[1:], "%y%b%d").date().isoformat() if ev else None
            km.append(dict(city=city, series=series, same_airport=same, ticker=m["ticker"], date=d,
                           strike_type=m.get("strike_type"), floor=m.get("floor_strike"), cap=m.get("cap_strike"),
                           result=m.get("result"), open_time=m["open_time"], close_time=m["close_time"],
                           title=m.get("yes_sub_title")))
        print(f"Kalshi {series}: {len(ms)} markets")
    km = pd.DataFrame(km)
    km.to_csv(os.path.join(OUT, "k_markets.csv"), index=False)
    with ThreadPoolExecutor(6) as ex:
        candles = [r for x in ex.map(k_candles, [(s, {"ticker": t, "open_time": o, "close_time": c})
                                                 for s, t, o, c in zip(km.series, km.ticker, km.open_time, km.close_time)])
                   for r in x]
        trades = [r for x in ex.map(k_trades, km.ticker) for r in x]
    pd.DataFrame(candles, columns=["ticker", "t", "yes_bid", "yes_ask", "last", "volume"]).to_parquet(
        os.path.join(OUT, "k_candles.parquet"), index=False)
    pd.DataFrame(trades, columns=["ticker", "t", "yes_price", "count", "taker_side"]).to_parquet(
        os.path.join(OUT, "k_trades.parquet"), index=False)
    print(f"Kalshi: {len(candles)} minute candles, {len(trades)} trades")
    # Polymarket
    today = datetime.now(timezone.utc).date()
    jobs = [(c, today - timedelta(days=i)) for c in PAIRS for i in range(1, DAYS + 2)]
    with ThreadPoolExecutor(8) as ex:
        pm = [r for x in ex.map(pm_event, jobs) for r in x]
    pm = pd.DataFrame(pm, columns=["city", "date", "event_slug", "label", "yes_token", "no_token", "won",
                                   "market_start", "closed_time"])
    pm.to_csv(os.path.join(OUT, "pm_markets.csv"), index=False)
    print(f"Polymarket: {len(pm)} brackets, {pm.event_slug.nunique()} events")
    jobs = [(t, ts(s) - 3600, (ts(c) if isinstance(c, str) else ts(s) + 4 * 86400) + 3600)
            for t, s, c in zip(pm.yes_token, pm.market_start, pm.closed_time)]
    with ThreadPoolExecutor(8) as ex:
        hist = [r for x in ex.map(pm_history, jobs) for r in x]
    pd.DataFrame(hist, columns=["yes_token", "t", "p"]).to_parquet(os.path.join(OUT, "pm_prices.parquet"), index=False)
    print(f"Polymarket: {len(hist)} 1-minute prices")


def pm_trades_event(slug):
    """All TAKER fills of one Polymarket event (every bracket), same format as backtest/trades.py."""
    j = get(f"{GAMMA}/events", {"slug": slug})
    if not j:
        return None
    rows = []
    for m in j[0].get("markets", []):
        cursor = None
        for _ in range(100):
            p = {"condition": m["conditionId"], "limit": 500, "taker_only": "true"}
            if cursor:
                p["cursor"] = cursor
            r = get("https://data-api.polymarket.com/v2/trades", p)
            if r is None:
                return None
            for t in r.get("data") or []:
                rows.append((slug, str(t["token_id"]), t["side"], float(t["size"]), float(t["price"]),
                             int(t["timestamp"]), t["proxy_wallet"]))
            pg = r.get("pagination") or {}
            cursor = pg.get("next_cursor")
            if not pg.get("has_more") or not cursor:
                break
    return rows


def pm_trades():
    """Taker fills for the same-airport cities (step 1 of the execution check)."""
    pm = pd.read_csv(os.path.join(OUT, "pm_markets.csv"))
    slugs = sorted(pm[pm.city.isin([c for c, (_, same) in PAIRS.items() if same])].event_slug.unique())
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(pm_trades_event, slugs))
    failed = sum(r is None for r in res)
    rows = [x for r in res if r for x in r]
    pd.DataFrame(rows, columns=["event_slug", "token_id", "side", "size", "price", "timestamp", "wallet"]).to_parquet(
        os.path.join(OUT, "pm_trades.parquet"), index=False)
    print(f"Polymarket taker fills: {len(rows)} in {len(slugs) - failed} events ({failed} failed)")


if __name__ == "__main__":
    import sys
    pm_trades() if len(sys.argv) > 1 and sys.argv[1] == "pmtrades" else main()
