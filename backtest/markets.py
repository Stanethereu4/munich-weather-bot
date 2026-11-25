"""
backtest/markets.py  (Part C1)
------------------------------
Builds the list of every RESOLVED Munich temperature market (highest and
lowest) from Polymarket's Gamma API: one row per bracket.

Gamma's /events endpoint ignores free-text search, so events are found by
their exact slug, one date at a time, e.g.
    highest-temperature-in-munich-on-november-22-2026
    lowest-temperature-in-munich-on-november-22-2026

Output: backtest_data/markets.csv with columns
    kind, target_date, event_slug, bracket_label, yes_token, no_token,
    condition_id, market_start, market_end, closed_time, won, volume

Run:  python3 -m backtest.markets [first_date]     (default first_date: 2026-05-01)
Adding another city later = adding its slug prefixes to SERIES below
(airport-station cities only, see the research doc).
"""

import csv
import json
import os
import sys
import time
from datetime import date, timedelta

import requests

GAMMA = "https://gamma-api.polymarket.com/events"
HEADERS = {"User-Agent": "munich-weather-bot backtest (research)"}
OUT_DIR = "backtest_data"

# (city, kind) -> slug prefix. One line per series.
SERIES = {
    ("munich", "high"): "highest-temperature-in-munich-on",
    ("munich", "low"): "lowest-temperature-in-munich-on",
}

FIELDS = ["city", "kind", "target_date", "event_slug", "bracket_label", "yes_token", "no_token",
          "condition_id", "market_start", "market_end", "closed_time", "won", "volume"]


def slug_for(prefix, d):
    return f"{prefix}-{d.strftime('%B').lower()}-{d.day}-{d.year}"


def get_event(slug, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(GAMMA, params={"slug": slug}, headers=HEADERS, timeout=20)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            events = r.json()
            return events[0] if events else None
        except requests.RequestException as e:
            print(f"  {slug}: {e} (attempt {attempt + 1})")
            time.sleep(2)
    return None


def bracket_rows(city, kind, target_date, event):
    """One row per bracket; won = 1 for the bracket Polymarket resolved YES."""
    rows = []
    for m in event.get("markets", []):
        tokens = m.get("clobTokenIds")
        tokens = json.loads(tokens) if isinstance(tokens, str) else (tokens or [])
        prices = m.get("outcomePrices")
        prices = json.loads(prices) if isinstance(prices, str) else (prices or [])
        if len(tokens) < 2:
            continue
        rows.append({
            "city": city, "kind": kind, "target_date": target_date.isoformat(),
            "event_slug": event.get("slug"),
            "bracket_label": m.get("groupItemTitle") or m.get("question"),
            "yes_token": tokens[0], "no_token": tokens[1],
            "condition_id": m.get("conditionId"),
            "market_start": m.get("startDate"), "market_end": m.get("endDate"),
            "closed_time": m.get("closedTime"),
            "won": 1 if prices and str(prices[0]) == "1" else 0,
            "volume": m.get("volume"),
        })
    return rows


def main(first=date(2026, 5, 1)):
    os.makedirs(OUT_DIR, exist_ok=True)
    last = date.today() - timedelta(days=2)          # resolved for sure
    all_rows, summary = [], {}
    for (city, kind), prefix in SERIES.items():
        n_events = n_unresolved = 0
        first_found = None
        d = first
        while d <= last:
            ev = get_event(slug_for(prefix, d))
            if ev:
                rows = bracket_rows(city, kind, d, ev)
                if sum(r["won"] for r in rows) == 1:
                    all_rows += rows
                    n_events += 1
                    first_found = first_found or d
                else:
                    n_unresolved += 1
                    print(f"  {ev.get('slug')}: {sum(r['won'] for r in rows)} winners -- skipped")
            d += timedelta(days=1)
            time.sleep(0.05)                          # be polite to Gamma
        summary[(city, kind)] = (n_events, first_found, n_unresolved)

    path = os.path.join(OUT_DIR, "markets.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nwrote {len(all_rows)} bracket rows to {path}")
    for (city, kind), (n, ff, nu) in summary.items():
        print(f"  {city} {kind}: {n} resolved events, first {ff}, {nu} skipped (not cleanly resolved)")


if __name__ == "__main__":
    main(date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2026, 5, 1))
