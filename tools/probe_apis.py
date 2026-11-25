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

if "station" in probes:
    d = date.today() - timedelta(days=2)
    section(f"AWC /api/data/metar json, EDDM, ~30h ending {d + timedelta(days=1)}T02:00Z")
    r = requests.get("https://aviationweather.gov/api/data/metar",
                     params={"ids": "EDDM", "format": "json", "hours": 30,
                             "date": (d + timedelta(days=1)).strftime("%Y%m%d") + "_0200"}, headers=H, timeout=30)
    rows = r.json()
    print("n reports:", len(rows))
    shape(rows[0])
    types = {}
    for x in rows:
        types[x.get("metarType")] = types.get(x.get("metarType"), 0) + 1
    print("metarType counts:", types)
    specis = [x.get("rawOb") for x in rows if x.get("metarType") != "METAR"][:3]
    print("non-METAR examples:", specis)
    print("raw with T-group (RMK T....):", [x.get("rawOb") for x in rows if " T0" in x.get("rawOb","") or " T1" in x.get("rawOb","")][:2])
    section("IEM asos.py EDDM, report_type 3 (routine) vs 4 (specials), same day")
    for rt in (3, 4):
        r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
                         params={"station": "EDDM", "data": ["tmpc", "metar"], "tz": "Europe/Berlin",
                                 "year1": d.year, "month1": d.month, "day1": d.day,
                                 "year2": (d + timedelta(days=1)).year, "month2": (d + timedelta(days=1)).month,
                                 "day2": (d + timedelta(days=1)).day, "format": "onlycomma", "latlon": "no",
                                 "missing": "M", "direct": "no", "report_type": rt}, headers=H, timeout=60)
        lines = r.text.strip().splitlines()
        print(f"report_type={rt}: {len(lines)-1} rows | header: {lines[0]} | first: {lines[1] if len(lines)>1 else None}")

if "gamma-lowest" in probes:
    # Do "lowest temperature" markets exist for Munich? Try a few slug
    # patterns for recent and upcoming dates, then print rules + brackets.
    section("GAMMA lowest-temperature markets for Munich")
    found = None
    for d in [date.today() + timedelta(days=k) for k in (1, 0, -1, -3, -10)]:
        for pat in ("lowest-temperature-in-munich-on-{m}-{d}-{y}", "minimum-temperature-in-munich-on-{m}-{d}-{y}"):
            slug = pat.format(m=d.strftime('%B').lower(), d=d.day, y=d.year)
            r = requests.get(f"{GAMMA}/events", params={"slug": slug}, headers=H, timeout=20).json()
            if r:
                hi = requests.get(f"{GAMMA}/events", params={"slug": slug.replace("lowest", "highest")},
                                  headers=H, timeout=20).json()
                h = hi[0] if hi else {}
                print(f"{slug}: FOUND | volume {r[0].get('volume')} liquidity {r[0].get('liquidity')}"
                      f" || same-day HIGHEST market: volume {h.get('volume')} liquidity {h.get('liquidity')}")
            else:
                print(f"{slug}: -")
            if r and found is None:
                found = r[0]
    # also look at the series for Munich (the max markets are series 'munich-daily-weather')
    r = requests.get(f"{GAMMA}/series", params={"slug": "munich-daily-weather"}, headers=H, timeout=20)
    print("series munich-daily-weather:", r.status_code, r.text[:300])
    if found:
        print("\nTITLE:", found.get("title"), "| series:", [x.get("slug") for x in found.get("series", [])])
        print(found.get("description"))
        print("brackets:", [m.get("groupItemTitle") for m in found.get("markets", [])])
        print("volume:", found.get("volume"), "| liquidity:", found.get("liquidity"))

if "price-history" in probes:
    # C2: which public source gives the finest historical prices for a
    # RESOLVED market? One resolved high market (and its low twin), the
    # most-traded bracket, three candidate sources.
    import statistics
    from datetime import datetime, timezone
    d = date.today() - timedelta(days=3)
    for kind_slug in ("highest", "lowest"):
        slug = slug_for(d).replace("highest", kind_slug)
        ev = requests.get(f"{GAMMA}/events", params={"slug": slug}, headers=H, timeout=20).json()[0]
        m = max(ev["markets"], key=lambda m: float(m.get("volumeNum") or m.get("volume") or 0))
        yes_tok = json.loads(m["clobTokenIds"])[0]
        cond = m.get("conditionId")
        section(f"{slug} | bracket {m.get('groupItemTitle')} | volume {m.get('volume')} | conditionId {cond[:12]}...")
        print("market start:", m.get("startDate"), "| end:", m.get("endDate"), "| closedTime:", m.get("closedTime"))

        # 1) Data API trades feed (every fill)
        trades, offset = [], 0
        while offset < 5000:
            r = requests.get("https://data-api.polymarket.com/trades",
                             params={"market": cond, "limit": 500, "offset": offset}, headers=H, timeout=30)
            if r.status_code != 200:
                print("trades:", r.status_code, r.text[:200]); break
            batch = r.json()
            if not batch: break
            trades += batch; offset += len(batch)
            if len(batch) < 500: break
        print(f"\n[1] data-api /trades?market=<conditionId>: {len(trades)} trades (all brackets of this market? see 'asset')")
        if trades:
            shape(trades[0])
            mine = sorted([t for t in trades if str(t.get("asset")) == yes_tok], key=lambda t: t["timestamp"])
            print(f"   trades on this bracket's YES token: {len(mine)}")
            if len(mine) > 1:
                ts = [int(t["timestamp"]) for t in mine]
                gaps = [(b - a) / 60 for a, b in zip(ts, ts[1:])]
                print(f"   first {datetime.fromtimestamp(ts[0], timezone.utc):%Y-%m-%d %H:%M} | last {datetime.fromtimestamp(ts[-1], timezone.utc):%Y-%m-%d %H:%M} UTC")
                print(f"   gap between trades (min): median {statistics.median(gaps):.1f} | 90th pct {sorted(gaps)[int(len(gaps)*.9)]:.0f} | max {max(gaps):.0f}")
                hours = {datetime.fromtimestamp(t, timezone.utc).strftime('%m-%d %H') for t in ts}
                print(f"   distinct clock-hours with >=1 trade: {len(hours)}")

        # 2) CLOB prices-history, finest fidelity, whole life of the market
        for params in ({"market": yes_tok, "interval": "max", "fidelity": 1},
                       {"market": yes_tok, "startTs": int(datetime.fromisoformat(m["startDate"].replace("Z", "+00:00")).timestamp()),
                        "endTs": int(datetime.fromisoformat(m["startDate"].replace("Z", "+00:00")).timestamp()) + 4 * 86400,
                        "fidelity": 1}):
            r = requests.get(f"{CLOB}/prices-history", params=params, headers=H, timeout=30)
            hist = r.json().get("history", []) if r.status_code == 200 else []
            label = "interval=max" if "interval" in params else "startTs/endTs"
            if len(hist) > 1:
                gaps = [(b["t"] - a["t"]) / 60 for a, b in zip(hist, hist[1:])]
                print(f"\n[2] clob /prices-history ({label}, fidelity=1): {len(hist)} points, "
                      f"median spacing {statistics.median(gaps):.0f} min, sample {hist[:2]}")
            else:
                print(f"\n[2] clob /prices-history ({label}): status {r.status_code}, {len(hist)} points, {r.text[:150]}")

        # 3) 'v2 prices-history with bucket_seconds' -- try likely URLs
        for url in ("https://data-api.polymarket.com/v2/prices-history",
                    "https://data-api.polymarket.com/prices-history",
                    "https://clob.polymarket.com/v2/prices-history"):
            r = requests.get(url, params={"market": yes_tok, "bucket_seconds": 60}, headers=H, timeout=30)
            print(f"[3] {url}?bucket_seconds=60 -> {r.status_code} {r.text[:160]!r}")

if "price-history-2" in probes:
    # C2 follow-up: data retention by market age, Data API v2, and what a
    # CLOB 1-minute point actually is (forward-filled? trade-driven?).
    import statistics
    from datetime import datetime, timezone

    def iso_ts(x):
        return int(datetime.fromisoformat(x.replace("Z", "+00:00").replace(" ", "T")).timestamp())

    for d in (date.today() - timedelta(days=3), date(2026, 10, 15), date(2026, 7, 15)):
        slug = slug_for(d)
        evs = requests.get(f"{GAMMA}/events", params={"slug": slug}, headers=H, timeout=20).json()
        if not evs:
            print(f"\n{slug}: not found"); continue
        ev = evs[0]
        m = max(ev["markets"], key=lambda m: float(m.get("volumeNum") or m.get("volume") or 0))
        tok = json.loads(m["clobTokenIds"])[0]; cond = m["conditionId"]
        t0, t1 = iso_ts(m["startDate"]), iso_ts(m["startDate"]) + 4 * 86400
        section(f"{slug} | bracket {m.get('groupItemTitle')} | volume {float(m.get('volume') or 0):.0f}")

        # CLOB 1-min
        r = requests.get(f"{CLOB}/prices-history", params={"market": tok, "startTs": t0, "endTs": t1, "fidelity": 1},
                         headers=H, timeout=30)
        hist = r.json().get("history", []) if r.status_code == 200 else []
        if hist:
            gaps = [(b["t"] - a["t"]) / 60 for a, b in zip(hist, hist[1:])] or [0]
            changes = sum(1 for a, b in zip(hist, hist[1:]) if a["p"] != b["p"])
            print(f"[CLOB fidelity=1] {len(hist)} pts, median spacing {statistics.median(gaps):.0f} min, "
                  f"price changes between consecutive points: {changes}")
        else:
            print(f"[CLOB fidelity=1] status {r.status_code}, 0 pts {r.text[:120]}")

        # Data API v2 prices-history: omitted bucket (server picks densest), 300, 60
        for bs in (None, 300, 60):
            params = {"token_id": tok, "start": t0, "end": t1}
            if bs: params["bucket_seconds"] = bs
            r = requests.get("https://data-api.polymarket.com/v2/prices-history", params=params, headers=H, timeout=60)
            try:
                j = r.json()
            except Exception:
                j = {}
            rows = j.get("data") or j.get("history") or []
            keys = [k for k in j.keys()] if isinstance(j, dict) else type(j).__name__
            print(f"[v2 prices-history bucket={bs}] {r.status_code} keys={keys} rows={len(rows)} "
                  f"res={j.get('resolution_seconds') if isinstance(j, dict) else None} first={rows[:1]}")

        # Data API v2 trades for this market (one page) -> shape + compare with CLOB
        r = requests.get("https://data-api.polymarket.com/v2/trades", params={"condition": cond, "limit": 500},
                         headers=H, timeout=60)
        try:
            j = r.json()
        except Exception:
            j = {}
        rows = (j.get("data") or []) if isinstance(j, dict) else []
        print(f"[v2 trades condition=...] {r.status_code} rows on first page={len(rows)} "
              f"pagination={j.get('pagination') if isinstance(j, dict) else None}")
        if rows:
            shape(rows[0])
        if rows and d == date.today() - timedelta(days=3):
            # what is a CLOB point? check a few trade times against the CLOB minute series
            by_min = {h["t"] // 60: h["p"] for h in hist}
            mine = [x for x in rows if str(x.get("token_id") or x.get("asset") or x.get("asset_id")) == tok][:8]
            for x in mine:
                ts = int(x.get("timestamp") or x.get("ts") or 0)
                print(f"   trade {datetime.fromtimestamp(ts, timezone.utc):%m-%d %H:%M:%S} price {x.get('price')} "
                      f"side {x.get('side')} | CLOB same minute {by_min.get(ts // 60)} next minute {by_min.get(ts // 60 + 1)}")

if "om-meta" in probes:
    # B1/B2: when does each model's latest run become available on Open-Meteo?
    # Candidate per-model status files behind open-meteo.com/en/docs/model-updates.
    section("Open-Meteo per-model metadata (static/meta.json)")
    from datetime import datetime, timezone
    candidates = ["ecmwf_ifs025", "ecmwf_ifs", "dwd_icon", "dwd_icon_eu", "dwd_icon_d2", "cmc_gem_global", "gem_global", "cmc_gem_rdps", "cmc_gem_hrdps", "cmc_gem_gdps_15km", "meteofrance_arpege_world", "meteofrance_arpege_world025",
                  "ncep_gfs025", "ncep_gfs013", "cmc_gem_gdps", "meteofrance_arpege_europe",
                  "meteofrance_arome_france", "meteofrance_arome_france_hd"]
    first = True
    for name in candidates:
        url = f"https://api.open-meteo.com/data/{name}/static/meta.json"
        r = requests.get(url, headers=H, timeout=20)
        if r.status_code != 200:
            print(f"{name:<30} {r.status_code}")
            continue
        j = r.json()
        if first:
            print(f"full response for {name}:"); shape(j); first = False
        fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M") if isinstance(t, (int, float)) else t
        print(f"{name:<30} init {fmt(j.get('last_run_initialisation_time'))} | "
              f"available {fmt(j.get('last_run_availability_time'))} | "
              f"modified {fmt(j.get('last_run_modification_time'))} | "
              f"update every {j.get('update_interval_seconds')}s")

if "single-runs" in probes:
    # C3: which models does the Single Runs API keep, from when, what shape?
    section("Open-Meteo Single Runs API (single-runs-api.open-meteo.com/v1/forecast)")
    URL = "https://single-runs-api.open-meteo.com/v1/forecast"
    models = ["ecmwf_ifs025", "ecmwf_ifs", "icon_seamless", "icon_d2", "icon_eu", "icon_global",
              "gfs_seamless", "gfs025", "ncep_gfs025", "gem_global", "meteofrance_seamless",
              "meteofrance_arpege_europe", "meteofrance_arome_france_hd"]
    shown = False
    for run in ("2026-11-20T00:00", "2026-11-20T03:00", "2026-07-15T00:00", "2026-06-01T00:00",
                "2026-04-05T00:00", "2026-03-01T00:00"):
        print(f"\nrun={run}")
        for mdl in models:
            r = requests.get(URL, params={"latitude": 48.3537, "longitude": 11.786, "run": run,
                                          "models": mdl, "hourly": "temperature_2m",
                                          "timezone": "UTC"}, headers=H, timeout=60)
            try:
                j = r.json()
            except Exception:
                j = {}
            if r.status_code == 200 and "hourly" in j:
                t = j["hourly"].get("time", []); v = j["hourly"].get("temperature_2m", [])
                nn = [x for x in v if x is not None]
                print(f"  {mdl:<30} OK  {len(t)} hours {t[0] if t else ''} -> {t[-1] if t else ''} | non-null {len(nn)}")
                if not shown:
                    print("  response keys:", list(j.keys()), "| hourly keys:", list(j["hourly"].keys()),
                          "| hourly_units:", j.get("hourly_units")); shown = True
            else:
                print(f"  {mdl:<30} {r.status_code} {str(j.get('reason', r.text))[:110]}")

if "gamma-rules" in probes:
    d = date.today() - timedelta(days=3)
    ev = gamma_event(d)
    section(f"GAMMA rules text (description) for {slug_for(d)}")
    print(ev.get("description"))
    print("\nresolutionSource:", ev.get("resolutionSource"))
    m = ev["markets"][0]
    print("market keys containing 'uma'/'resol':",
          {k: m.get(k) for k in m if "uma" in k.lower() or "resol" in k.lower()})
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
