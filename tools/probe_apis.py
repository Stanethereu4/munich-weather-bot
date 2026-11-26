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
    models = ["ecmwf_ifs025", "icon_seamless", "gfs_seamless", "gem_global", "meteofrance_seamless",
              "meteofrance_arome_france_hd", "gfs_global"]
    shown = True

    def get_retry(params, tries=5):
        for k in range(tries):
            try:
                return requests.get(URL, params=params, headers=H, timeout=(10, 60))
            except requests.RequestException as e:
                print(f"    (retry {k + 1}: {type(e).__name__})"); time.sleep(5 * (k + 1))
        return None

    import time
    for run in ("2026-11-20T03:00", "2026-07-15T00:00", "2026-07-15T06:00", "2026-06-01T00:00",
                "2026-04-05T00:00", "2026-03-01T00:00"):
        print(f"\nrun={run}")
        for mdl in models:
            r = get_retry({"latitude": 48.3537, "longitude": 11.786, "run": run,
                           "models": mdl, "hourly": "temperature_2m", "timezone": "UTC"})
            if r is None:
                print(f"  {mdl:<30} gave up after retries"); continue
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

if "ens-mean" in probes:
    # Is there a usable HISTORY of ensemble mean + spread, and is it per run?
    import time
    section("Ensemble mean/spread history (ensemble-api) and per-run access (single-runs-api)")
    names = ["dwd_icon_eps_ensemble_mean_seamless", "ecmwf_ifs025_ensemble_mean", "ecmwf_ifs_ensemble_mean",
             "ncep_gefs_ensemble_mean_seamless", "ecmwf_aifs025_ensemble_mean"]

    def get(url, params, tries=4):
        for k in range(tries):
            try:
                return requests.get(url, params=params, headers=H, timeout=(10, 40))
            except requests.RequestException as e:
                time.sleep(4 * (k + 1))
        return None

    for name in names:
        base = {"latitude": 48.3537, "longitude": 11.786, "models": name,
                "hourly": "temperature_2m,temperature_2m_spread", "timezone": "UTC"}
        r = get("https://ensemble-api.open-meteo.com/v1/ensemble", {**base, "start_date": "2026-07-15", "end_date": "2026-07-15"})
        if r is None:
            print(f"{name}: ensemble-api no response"); continue
        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        h = j.get("hourly", {})
        print(f"{name}\n  ensemble-api 2026-07-15: {r.status_code} keys={list(h.keys())} "
              f"sample mean={h.get('temperature_2m', [None]*13)[12]} spread={h.get('temperature_2m_spread', [None]*13)[12]} "
              f"{'' if r.status_code == 200 else j.get('reason', '')[:100]}")
        for run in ("2026-07-14T00:00", "2026-07-15T00:00"):
            r = get("https://single-runs-api.open-meteo.com/v1/forecast", {**base, "run": run})
            if r is None:
                print(f"  single-runs run={run}: no response"); continue
            j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            h = j.get("hourly", {})
            t = h.get("time", [])
            idx = t.index("2026-07-15T12:00") if "2026-07-15T12:00" in t else None
            print(f"  single-runs run={run}: {r.status_code} "
                  + (f"15 Jul 12z mean={h['temperature_2m'][idx]} spread={h.get('temperature_2m_spread', [None]*999)[idx]}"
                     if idx is not None else str(j.get('reason', ''))[:100]))

if "ens-recent" in probes:
    # For the last few days: can we get ensemble forecasts AS ISSUED per run (no look-ahead)?
    import time
    section("Recent ensemble runs: per-run access?")
    def get(url, params, tries=3):
        for k in range(tries):
            try:
                return requests.get(url, params=params, headers=H, timeout=(10, 60))
            except requests.RequestException:
                time.sleep(4 * (k + 1))
        return None
    def show(tag, r, want="2026-11-24T12:00"):
        if r is None:
            print(f"  {tag}: no response"); return
        try:
            j = r.json()
        except Exception:
            print(f"  {tag}: {r.status_code} non-json"); return
        if r.status_code != 200:
            print(f"  {tag}: {r.status_code} {str(j.get('reason',''))[:110]}"); return
        h = j.get("hourly", {}); t = h.get("time", [])
        keys = [k for k in h if k != "time"]
        i = t.index(want) if want in t else None
        sample = {k: h[k][i] for k in keys[:3]} if i is not None else "target hour absent"
        print(f"  {tag}: 200 n_keys={len(keys)} first_keys={keys[:4]} time {t[0] if t else None}..{t[-1] if t else None} sample@{want}={sample}")
    loc = {"latitude": 48.3537, "longitude": 11.786, "timezone": "UTC"}
    for m in ["ecmwf_ifs025", "icon_seamless", "gfs_seamless", "ecmwf_aifs025", "ecmwf_ifs025_ensemble_mean"]:
        print(m)
        show("single-runs run=2026-11-23T00", get("https://single-runs-api.open-meteo.com/v1/forecast",
             {**loc, "models": m, "hourly": "temperature_2m", "run": "2026-11-23T00:00"}))
        show("ensemble-api run=2026-11-23T00", get("https://ensemble-api.open-meteo.com/v1/ensemble",
             {**loc, "models": m, "hourly": "temperature_2m", "run": "2026-11-23T00:00"}))
        show("ensemble-api start 22..25", get("https://ensemble-api.open-meteo.com/v1/ensemble",
             {**loc, "models": m, "hourly": "temperature_2m", "start_date": "2026-11-22", "end_date": "2026-11-25"}))
        show("ensemble-api previous_day1", get("https://ensemble-api.open-meteo.com/v1/ensemble",
             {**loc, "models": m, "hourly": "temperature_2m_previous_day1", "past_days": 3, "forecast_days": 1}))

if "ens-prev" in probes:
    # Ensemble MEAN with previous_dayN (forecast issued N days earlier): history depth and spread availability.
    import time
    section("Ensemble mean previous_dayN: depth and spread")
    loc = {"latitude": 48.3537, "longitude": 11.786, "timezone": "UTC"}
    vars_ = ",".join(f"temperature_2m_previous_day{n},temperature_2m_spread_previous_day{n}" for n in (1, 2, 3))
    for m in ["ecmwf_ifs025_ensemble_mean", "dwd_icon_eps_ensemble_mean_seamless", "ncep_gefs_ensemble_mean_seamless",
              "ecmwf_aifs025_ensemble_mean", "gem_global_ensemble_mean", "ukmo_global_ensemble_mean_20km"]:
        r = None
        for k in range(3):
            try:
                r = requests.get("https://ensemble-api.open-meteo.com/v1/ensemble", params={**loc, "models": m,
                    "hourly": vars_, "start_date": "2026-08-24", "end_date": "2026-11-25"}, headers=H, timeout=(10, 90)); break
            except requests.RequestException:
                time.sleep(5 * (k + 1))
        if r is None:
            print(f"{m}: no response"); continue
        j = r.json()
        if r.status_code != 200:
            print(f"{m}: {r.status_code} {str(j.get('reason',''))[:120]}"); continue
        h = j["hourly"]; t = h["time"]
        print(f"{m}: {len(t)} hours {t[0]}..{t[-1]}")
        for k, v in h.items():
            if k == "time": continue
            nn = [i for i, x in enumerate(v) if x is not None]
            print(f"   {k}: non-null {len(nn)}/{len(v)} first {t[nn[0]] if nn else None} sample@2026-10-15T12={v[t.index('2026-10-15T12:00')]}")

if "trades-shape" in probes:
    # Who trades Munich markets, and can we see maker vs taker per fill?
    section("Trades data: wallets and maker/taker")
    slug = "highest-temperature-in-munich-on-november-23-2026"
    ev = requests.get(f"{GAMMA}/events", params={"slug": slug}, headers=H, timeout=20).json()[0]
    m = max(ev["markets"], key=lambda m: float(m.get("volumeNum") or 0))
    cond, tok = m["conditionId"], json.loads(m["clobTokenIds"])[0]
    print(f"{slug} bracket {m.get('groupItemTitle')} volume {m.get('volumeNum')} cond {cond[:14]}.. token {tok[:14]}..")
    for url, params in (("https://data-api.polymarket.com/trades", {"market": cond, "limit": 3}),
                        ("https://data-api.polymarket.com/v2/trades", {"condition": cond, "limit": 3})):
        r = requests.get(url, params=params, headers=H, timeout=60)
        print(f"\n{url} -> {r.status_code}")
        try:
            j = r.json()
        except Exception:
            print(r.text[:200]); continue
        rows = j if isinstance(j, list) else (j.get("data") or [])
        if isinstance(j, dict):
            print("  top-level keys:", list(j.keys()))
        for row in rows[:2]:
            print("  ", json.dumps(row)[:700])
    gq = ("https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0bfh/"
          "subgraphs/orderbook-subgraph/0.0.1/gn")
    q = {"query": '{ orderFilledEvents(first: 3, orderBy: timestamp, orderDirection: desc, '
                  'where: {makerAssetId: "%s"}) { id timestamp maker taker makerAssetId takerAssetId '
                  'makerAmountFilled takerAmountFilled fee } }' % tok}
    try:
        r = requests.post(gq, json=q, headers=H, timeout=60)
        print(f"\ngoldsky orderbook subgraph -> {r.status_code}\n  ", r.text[:900])
    except requests.RequestException as e:
        print("goldsky:", type(e).__name__)
    w = None
    try:
        rows = requests.get("https://data-api.polymarket.com/trades", params={"market": cond, "limit": 1},
                            headers=H, timeout=60).json()
        w = rows[0].get("proxyWallet") if rows else None
    except Exception:
        pass
    if w:
        for url, params in (("https://data-api.polymarket.com/positions", {"user": w, "limit": 2}),
                            ("https://data-api.polymarket.com/closed-positions", {"user": w, "limit": 2}),
                            ("https://data-api.polymarket.com/activity", {"user": w, "limit": 2})):
            r = requests.get(url, params=params, headers=H, timeout=60)
            print(f"\n{url} (one wallet) -> {r.status_code}\n  ", r.text[:600])

if "trades-shape2" in probes:
    section("Trades v2: full keys, taker_only switch, per-wallet endpoints")
    cond = "0x054fa1234215343aae10e41fa312ef0d2d54676be0c4e98d2f0d874b40ff7c65"
    r = requests.get("https://data-api.polymarket.com/trades", params={"market": cond}, headers=H, timeout=60)
    print("v1 410 body:", r.text[:600])
    def page(params):
        r = requests.get("https://data-api.polymarket.com/v2/trades", params={"condition": cond, "limit": 500, **params},
                         headers=H, timeout=60)
        j = r.json() if r.status_code == 200 else {}
        return r.status_code, j.get("data") or [], j.get("pagination"), r.text[:200]
    st, rows, pg, _ = page({})
    print("default:", st, len(rows), "pagination:", pg)
    if rows:
        print("keys:", list(rows[0].keys()))
        print("non-profile fields:", {k: v for k, v in rows[0].items() if k not in ("bio", "icon", "profile_image", "profile_image_optimized", "title")})
    for extra in ({"taker_only": "false"}, {"takerOnly": "false"}, {"taker_only": "true"}, {"side": "maker"}):
        st, rows2, pg2, txt = page(extra)
        print(extra, "->", st, len(rows2), pg2 if st == 200 else txt)
    w = rows[0]["proxy_wallet"] if rows else None
    for base in ("https://data-api.polymarket.com/v2", "https://data-api.polymarket.com"):
        for ep, params in (("trades", {"user": w, "limit": 3}), ("positions", {"user": w, "limit": 2}),
                           ("closed-positions", {"user": w, "limit": 2}), ("activity", {"user": w, "limit": 2}),
                           ("leaderboard", {"limit": 2}), ("holders", {"market": cond, "limit": 2})):
            r = requests.get(f"{base}/{ep}", params=params, headers=H, timeout=60)
            print(f"\n{base}/{ep} -> {r.status_code}: {r.text[:450]}")

if "cities" in probes:
    # Low-volume cities: series -> event slugs, history depth, bracket labels, resolution station
    import re
    section("Cities: series, history, labels, station")
    cities = ["toronto", "panama-city", "istanbul", "chicago", "cape-town", "mexico-city", "sao-paulo",
              "seattle", "warsaw", "denver"]
    for c in cities:
        for kind, sslug in (("high", f"{c}-daily-weather"), ("low", f"{c}-daily-lowest-temperature")):
            try:
                r = requests.get(f"{GAMMA}/series", params={"slug": sslug}, headers=H, timeout=30)
                j = r.json()
            except Exception as e:
                print(f"{c} {kind}: series error {type(e).__name__}"); continue
            if not j:
                print(f"{c} {kind}: no series '{sslug}'"); continue
            evs = j[0].get("events") or []
            slugs = sorted((e.get("slug"), e.get("closed"), e.get("volume")) for e in evs)
            closed = [e for e in evs if e.get("closed")]
            dates = sorted(e.get("endDate", "")[:10] for e in evs)
            print(f"{c} {kind}: series ok, {len(evs)} events ({len(closed)} closed), dates {dates[0] if dates else None}..{dates[-1] if dates else None}")
            if evs:
                ex = sorted(closed or evs, key=lambda e: e.get("endDate", ""))[-1]
                print(f"   example slug: {ex.get('slug')}  volume {ex.get('volume')}")
                ev = requests.get(f"{GAMMA}/events", params={"slug": ex["slug"]}, headers=H, timeout=30).json()[0]
                labels = [m.get("groupItemTitle") for m in ev["markets"]]
                print(f"   labels: {labels}")
                desc = ev.get("description") or ""
                st = re.findall(r"(?:station|Station|airport|Airport)[^.]{0,140}", desc)
                url = re.findall(r"https?://\S+", desc)
                print(f"   rules: {st[:2]} | urls: {url[:2]}")
                print(f"   outcomePrices sample: {[m.get('outcomePrices') for m in ev['markets'][:2]]}")

if "metar-timing" in probes:
    # When does each EDDM report become available, and where? (latency test)
    section("METAR timing: AWC receipt time, NOAA tgftp file, IEM")
    r = requests.get("https://aviationweather.gov/api/data/metar", params={"ids": "EDDM", "format": "json", "hours": 3},
                     headers=H, timeout=30)
    j = r.json()
    print(f"AWC -> {r.status_code}, {len(j)} reports; keys: {list(j[0].keys()) if j else None}")
    for x in j[:6]:
        print(f"   type {x.get('metarType')} obsTime {x.get('obsTime')} reportTime {x.get('reportTime')} "
              f"receiptTime {x.get('receiptTime')} temp {x.get('temp')} raw {str(x.get('rawOb'))[:60]}")
    r = requests.get("https://tgftp.nws.noaa.gov/data/observations/metar/stations/EDDM.TXT", headers=H, timeout=30)
    print(f"\nNOAA tgftp EDDM.TXT -> {r.status_code}, Last-Modified {r.headers.get('Last-Modified')}\n   {r.text.strip()[:160]}")
    r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
                     params={"station": "EDDM", "data": "metar", "hours": 2, "tz": "UTC", "format": "onlycomma",
                             "report_type": "3,4"}, headers=H, timeout=30)
    print(f"\nIEM -> {r.status_code}\n   {r.text.strip()[:300]}")

if "rewards" in probes:
    # Do weather markets pay maker rewards / rebates? Fields per market + CLOB rewards endpoints.
    section("Maker rewards / rebates on weather markets")
    import datetime as _dt
    d = _dt.datetime.now(_dt.timezone.utc).date()
    for slug in (f"highest-temperature-in-munich-on-{d.strftime('%B').lower()}-{d.day}-{d.year}",
                 f"lowest-temperature-in-munich-on-{d.strftime('%B').lower()}-{d.day}-{d.year}",
                 f"highest-temperature-in-seoul-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"):
        ev = requests.get(f"{GAMMA}/events", params={"slug": slug}, headers=H, timeout=30).json()
        if not ev:
            print(slug, "not found"); continue
        ev = ev[0]
        evk = {k: v for k, v in ev.items() if any(w in k.lower() for w in ("reward", "rebate", "fee", "incentive", "negrisk"))}
        print(f"\n{slug}\n  event fields: {evk}")
        m = ev["markets"][len(ev["markets"]) // 2]
        mk = {k: v for k, v in m.items() if any(w in k.lower() for w in ("reward", "rebate", "fee", "incentive", "spread", "liquidity"))}
        print(f"  market {m.get('groupItemTitle')}: {json.dumps(mk)[:900]}")
        cond = m["conditionId"]
        for url in (f"https://clob.polymarket.com/rewards/markets/{cond}", f"https://clob.polymarket.com/markets/{cond}"):
            r = requests.get(url, headers=H, timeout=30)
            txt = r.text
            keep = txt if "rewards" in url else json.dumps({k: v for k, v in (r.json() if r.status_code == 200 else {}).items()
                                                           if any(w in k.lower() for w in ("reward", "fee", "rebate", "maker", "taker"))})
            print(f"  {url.split('.com')[1][:40]} -> {r.status_code}: {keep[:700]}")
    for url in ("https://clob.polymarket.com/rewards/markets/current?limit=5",
                "https://clob.polymarket.com/sampling-markets?limit=1"):
        r = requests.get(url, headers=H, timeout=30)
        print(f"\n{url} -> {r.status_code}: {r.text[:500]}")

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
