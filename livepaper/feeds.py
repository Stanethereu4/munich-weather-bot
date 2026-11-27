"""
livepaper/feeds.py
------------------
Live public data (no account, no orders). Response shapes checked with the `live` probe, 27 Nov 2026:

Kalshi  GET /trade-api/v2/markets?series_ticker=KXHIGHDEN&status=open
          -> {"markets": [{"ticker", "event_ticker": "KXHIGHDEN-26NOV28", "strike_type", "floor_strike",
                           "cap_strike", "yes_bid_dollars", "yes_ask_dollars", "yes_bid_size_fp", ...}]}
        GET /trade-api/v2/markets/trades?ticker=..&min_ts=..
          -> {"trades": [{"created_time", "yes_price_dollars", "count_fp", "taker_side"}], "cursor"}
Polymarket
        Gamma GET /events?slug=highest-temperature-in-<city>-on-<month>-<day>-<year>
        CLOB  POST /books  [{"token_id": ..}, ..]
          -> [{"asset_id", "bids": [{"price","size"}..], "asks": [..], "timestamp"}]
             bids sorted cheapest first (best bid LAST), asks most expensive first (best ask LAST)
        Data  GET /v2/trades?condition=..&limit=..  (for the patient variant's fills)
aviationweather.gov  GET /api/data/metar?ids=KDEN&format=json&hours=30  (Polymarket's resolution source)
"""
import json
import re
import time
from datetime import datetime

import requests

from backtest.awc_obs import temp_c, to_f
from backtest.kalshi_test import pm_bounds

K = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
H = {"User-Agent": "munich-weather-bot paper trading (research, no orders)"}
S = requests.Session()


def get(url, params=None, tries=3, timeout=20):
    for k in range(tries):
        try:
            r = S.get(url, params=params, headers=H, timeout=timeout)
            if r.status_code == 429:
                time.sleep(2 * (k + 1)); continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(1 + k)
    return None


def k_bounds(m):
    """Kalshi bracket -> inclusive whole-°F range (between 62-63; greater 63 = 64+; less 56 = 55-)."""
    st = m.get("strike_type")
    if st == "between":
        return int(m["floor_strike"]), int(m["cap_strike"])
    if st == "greater":
        return int(m["floor_strike"]) + 1, 999
    if st == "less":
        return -999, int(m["cap_strike"]) - 1
    return None


def kalshi_quotes(series):
    """Open Kalshi brackets of a city, grouped by target date: {date: [dict(ticker, lo, hi, bid, ask)]}."""
    j = get(f"{K}/markets", {"series_ticker": series, "status": "open", "limit": 200})
    out = {}
    for m in (j or {}).get("markets", []):
        b = k_bounds(m)
        ev = re.search(r"-(\d{2}[A-Z]{3}\d{2})$", m["event_ticker"])
        if not b or not ev:
            continue
        d = datetime.strptime(ev.group(1), "%y%b%d").date().isoformat()
        f = lambda k: float(m.get(k) or "nan")
        out.setdefault(d, []).append(dict(ticker=m["ticker"], lo=b[0], hi=b[1], bid=f("yes_bid_dollars"),
                                          ask=f("yes_ask_dollars")))
    return out


def kalshi_trade_count(tickers, start_s, end_s):
    n = 0
    for t in tickers:
        j = get(f"{K}/markets/trades", {"ticker": t, "min_ts": start_s, "max_ts": end_s, "limit": 1000})
        n += len((j or {}).get("trades", []))
    return n


def pm_event(city, date):
    """Polymarket brackets for a city/date: list of dict(label, lo, hi, yes, no, condition) or None."""
    d = datetime.fromisoformat(date)
    slug = f"highest-temperature-in-{city}-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"
    j = get(f"{GAMMA}/events", {"slug": slug})
    if not j:
        return None
    out = []
    for m in j[0].get("markets", []):
        try:
            toks = json.loads(m.get("clobTokenIds") or "[]")
            lo, hi = pm_bounds(m.get("groupItemTitle") or "")
        except (ValueError, AttributeError, TypeError):
            continue
        if len(toks) == 2:
            out.append(dict(label=m.get("groupItemTitle"), lo=lo, hi=hi, yes=toks[0], no=toks[1],
                            condition=m.get("conditionId"), closed=bool(m.get("closed")),
                            prices=m.get("outcomePrices")))
    return dict(slug=slug, closed=bool(j[0].get("closed")), brackets=out)


def pm_books(tokens):
    """token -> dict(bid, ask, asks=[(price, size)] cheapest first). Empty side -> None."""
    out = {}
    for i in range(0, len(tokens), 40):
        chunk = tokens[i:i + 40]
        for k in range(3):
            try:
                r = S.post("https://clob.polymarket.com/books", json=[{"token_id": t} for t in chunk],
                           headers=H, timeout=20)
                r.raise_for_status()
                books = r.json()
                break
            except (requests.RequestException, ValueError):
                books = []
                time.sleep(1 + k)
        for b in books:
            asks = sorted((float(a["price"]), float(a["size"])) for a in b.get("asks", []))
            bids = sorted(((float(a["price"]), float(a["size"])) for a in b.get("bids", [])), reverse=True)
            out[b["asset_id"]] = dict(bid=bids[0][0] if bids else None, ask=asks[0][0] if asks else None,
                                      asks=asks, bids=bids)
    return out


def pm_sells_into_bids(condition, yes_token, no_token, since_s):
    """Real Polymarket trades since since_s where someone SOLD the YES (or bought the NO):
    list of (timestamp, yes_price, shares). Used to decide if a resting bid would have filled."""
    j = get("https://data-api.polymarket.com/v2/trades", {"condition": condition, "limit": 500})
    out = []
    for t in (j or {}).get("data") or []:
        ts = int(t["timestamp"])
        if ts < since_s:
            continue
        tok, side, p = str(t["token_id"]), t["side"], float(t["price"])
        if tok == yes_token and side == "SELL":
            out.append((ts, p, float(t["size"])))
        elif tok == no_token and side == "BUY":
            out.append((ts, 1 - p, float(t["size"])))
    return out


def floor_f(icao, tz, date):
    """Highest routine reading of the local date already published (°F, Polymarket's rules), or None."""
    from zoneinfo import ZoneInfo
    j = get("https://aviationweather.gov/api/data/metar", {"ids": icao, "format": "json", "hours": 30})
    best = None
    for x in j or []:
        raw = x.get("rawOb") or ""
        if raw.lstrip().startswith("SPECI"):
            continue
        if datetime.fromtimestamp(int(x["obsTime"]), ZoneInfo(tz)).date().isoformat() != date:
            continue
        c = temp_c(raw)
        if c is not None:
            best = max(best, to_f(c)) if best is not None else to_f(c)
    return best
