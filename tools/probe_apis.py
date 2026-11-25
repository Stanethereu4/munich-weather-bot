"""
probe_apis.py -- one real call per external endpoint, printing the
response STRUCTURE (keys, types, a short sample value), so code is
written against what the APIs actually return, not what we assume.

Run in GitHub Actions:  workflow_dispatch, task = probe
"""
import json
import sys
from datetime import date, timedelta

import requests

H = {"User-Agent": "munich-weather-bot (research)"}
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def shape(obj, indent=2, depth=0, max_items=3):
    """Print keys / types / short samples of a JSON object."""
    pad = " " * (indent * depth)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v and depth < 3:
                print(f"{pad}{k}: {type(v).__name__}[{len(v)}]")
                shape(v, indent, depth + 1, max_items)
            else:
                print(f"{pad}{k}: {type(v).__name__} = {json.dumps(v)[:90]}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:max_items]):
            print(f"{pad}[{i}]")
            shape(v, indent, depth + 1, max_items)
        if len(obj) > max_items:
            print(f"{pad}... {len(obj) - max_items} more")
    else:
        print(f"{pad}{type(obj).__name__} = {json.dumps(obj)[:90]}")


def slug_for(d):
    return f"highest-temperature-in-munich-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"


def gamma_event(d):
    r = requests.get(f"{GAMMA}/events", params={"slug": slug_for(d)}, headers=H, timeout=20)
    r.raise_for_status()
    ev = r.json()
    return ev[0] if ev else None


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


probes = sys.argv[1:] or ["gamma-open", "gamma-resolved", "clob-book"]
today = date.today()

if "gamma-open" in probes or "clob-book" in probes:
    open_ev = gamma_event(today + timedelta(days=1))

if "gamma-open" in probes:
    section(f"GAMMA /events?slug={slug_for(today + timedelta(days=1))}  (open market) -- event level, markets truncated")
    ev = dict(open_ev)
    markets = ev.pop("markets", [])
    shape(ev)
    section("  ... one market (bracket) of that event")
    shape(markets[0])

if "gamma-resolved" in probes:
    d = today - timedelta(days=3)
    section(f"GAMMA /events?slug={slug_for(d)}  (resolved market) -- resolution-related fields per bracket")
    ev = gamma_event(d)
    print("event closed:", ev.get("closed"), "| active:", ev.get("active"))
    for m in ev["markets"]:
        keep = {k: m.get(k) for k in m
                if any(s in k.lower() for s in ("outcome", "resol", "closed", "winner", "uma", "groupitemtitle", "clobtoken"))}
        print(json.dumps(keep)[:400])

if "clob-book" in probes:
    m = sorted(open_ev["markets"], key=lambda m: -float(json.loads(m["outcomePrices"])[0]))[0]
    yes_tok, no_tok = json.loads(m["clobTokenIds"])
    print("\nbracket:", m.get("groupItemTitle"), "| outcomes:", m.get("outcomes"), "| outcomePrices:", m.get("outcomePrices"))
    for name, tok in (("YES", yes_tok), ("NO", no_tok)):
        section(f"CLOB /book?token_id=<{name} token>")
        r = requests.get(f"{CLOB}/book", params={"token_id": tok}, headers=H, timeout=20)
        r.raise_for_status()
        b = r.json()
        shape({k: v for k, v in b.items() if k not in ("bids", "asks")})
        for side in ("bids", "asks"):
            lv = b.get(side, [])
            print(f"{side}: {len(lv)} levels | first 3: {lv[:3]} | last 3: {lv[-3:]}")
    for ep in ("midpoint", "price"):
        params = {"token_id": yes_tok} if ep == "midpoint" else {"token_id": yes_tok, "side": "BUY"}
        r = requests.get(f"{CLOB}/{ep}", params=params, headers=H, timeout=20)
        print(f"\nCLOB /{ep} {params.get('side','')}: {r.status_code} {r.text[:200]}")
