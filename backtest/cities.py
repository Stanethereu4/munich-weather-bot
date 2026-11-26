"""
backtest/cities.py
------------------
Downloads what the multi-city super-model test needs (backtest/cities_test.py):

  markets   every resolved daily high/low market per city since FIRST_DAY
            (Gamma /events?slug=highest-temperature-in-<city>-on-<month>-<day>-<year>,
            same pattern as Munich; checked with the `cities` probe, 26 Nov 2026)
            -> backtest_data/cities/markets.csv
  prices    10-minute YES price history per bracket (CLOB /prices-history, fidelity=10)
            -> backtest_data/cities/prices/<city>.parquet
  ensemble  ensemble mean + spread, issued 1-3 days earlier, at the resolution airport
            (same source as backtest/ensemble.py) -> backtest_data/cities/ens_<city>.parquet

Run:  python3 -m backtest.cities markets | prices | ensemble
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from backtest.timeconv import to_unix_seconds

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com/prices-history"
ENS = "https://ensemble-api.open-meteo.com/v1/ensemble"
H = {"User-Agent": "munich-weather-bot backtest (research)"}
OUT = os.path.join("backtest_data", "cities")
FIRST_DAY = date(2026, 8, 24)          # ensemble-mean history starts here

# city slug -> (ICAO, time zone, lat, lon); airports from each market's resolution rules
CITIES = {
    "munich":      ("EDDM", "Europe/Berlin",       48.3537,   11.7860),   # control
    "toronto":     ("CYYZ", "America/Toronto",     43.6772,  -79.6306),
    "istanbul":    ("LTFM", "Europe/Istanbul",     41.2753,   28.7519),
    "chicago":     ("KORD", "America/Chicago",     41.9786,  -87.9048),
    "cape-town":   ("FACT", "Africa/Johannesburg", -33.9649,  18.6017),
    "mexico-city": ("MMMX", "America/Mexico_City", 19.4363,  -99.0721),
    "sao-paulo":   ("SBGR", "America/Sao_Paulo",  -23.4356,  -46.4731),
    "seattle":     ("KSEA", "America/Los_Angeles", 47.4502, -122.3088),
    "warsaw":      ("EPWA", "Europe/Warsaw",       52.1657,   20.9671),
    "denver":      ("KDEN", "America/Denver",      39.8561, -104.6737),
}
KINDS = {"high": "highest-temperature-in", "low": "lowest-temperature-in"}
ENS_MODELS = ["ecmwf_ifs025_ensemble_mean", "dwd_icon_eps_ensemble_mean_seamless",
              "ncep_gefs_ensemble_mean_seamless", "ecmwf_aifs025_ensemble_mean"]


def get_json(url, params, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=H, timeout=(10, 180))
            if r.status_code == 429:
                time.sleep(5 * (k + 1)); continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            body = getattr(getattr(e, "response", None), "text", "") or ""
            print(f"    {url.split('/')[2]} attempt {k + 1}: {type(e).__name__} {body[:160]}")
            time.sleep(3 * (k + 1))
    return None


def slug_for(city, kind, d):
    return f"{KINDS[kind]}-{city}-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"


def one_event(args):
    city, kind, d = args
    j = get_json(f"{GAMMA}/events", {"slug": slug_for(city, kind, d)})
    if not j or not j[0].get("closed"):
        return []
    ev = j[0]
    rows = []
    for m in ev.get("markets", []):
        try:
            prices = json.loads(m.get("outcomePrices") or "[]")
            toks = json.loads(m.get("clobTokenIds") or "[]")
        except ValueError:
            continue
        if len(prices) != 2 or prices[0] not in ("0", "1") or len(toks) != 2:
            continue                       # unresolved / malformed
        rows.append((city, kind, d.isoformat(), ev["slug"], m.get("groupItemTitle"), toks[0], toks[1],
                     m.get("startDate"), m.get("closedTime"), int(prices[0] == "1"), float(m.get("volumeNum") or 0)))
    if sum(r[9] for r in rows) != 1:
        return []                          # need exactly one winning bracket
    return rows


def markets():
    os.makedirs(OUT, exist_ok=True)
    last = datetime.now(timezone.utc).date() - timedelta(days=1)
    jobs = [(c, k, FIRST_DAY + timedelta(days=i)) for c in CITIES for k in KINDS
            for i in range((last - FIRST_DAY).days + 1)]
    with ThreadPoolExecutor(8) as ex:
        rows = [r for res in ex.map(one_event, jobs) for r in res]
    df = pd.DataFrame(rows, columns=["city", "kind", "target_date", "event_slug", "bracket_label", "yes_token",
                                     "no_token", "market_start", "closed_time", "won", "volume"])
    df.to_csv(os.path.join(OUT, "markets.csv"), index=False)
    s = df.groupby(["city", "kind"]).agg(days=("target_date", "nunique"), first=("target_date", "min"),
                                         vol_per_day=("volume", "sum"))
    s["vol_per_day"] = (s["vol_per_day"] / s["days"]).round(0)
    print(s.to_string())


def one_history(args):
    tok, start, end = args
    j = get_json(CLOB, {"market": tok, "startTs": start, "endTs": end, "fidelity": 10})
    return tok, (j or {}).get("history")


def prices():
    mk = pd.read_csv(os.path.join(OUT, "markets.csv"), dtype={"yes_token": str})
    os.makedirs(os.path.join(OUT, "prices"), exist_ok=True)
    for city, g in mk.groupby("city"):
        path = os.path.join(OUT, "prices", f"{city}.parquet")
        if os.path.exists(path):
            continue
        st = to_unix_seconds(g["market_start"]) - 3600
        en = [int(pd.Timestamp(d).timestamp()) + 3 * 86400 for d in g["target_date"]]
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(one_history, zip(g["yes_token"], st, en)))
        parts = [pd.DataFrame({"yes_token": tok, "t": [h["t"] for h in hist], "p": [h["p"] for h in hist]})
                 for tok, hist in res if hist]
        missing = sum(1 for _, hist in res if hist is None)
        pd.concat(parts).to_parquet(path, index=False)
        print(f"{city}: {len(parts)} brackets with prices, {missing} failed")


def ensemble():
    today = datetime.now(timezone.utc).date()
    end = today.isoformat()
    # the archive is a rolling window of ~94 days, so start 92 days back (not a fixed date)
    start = max(FIRST_DAY, today - timedelta(days=92)).isoformat()
    hourly = ",".join(f"temperature_2m_previous_day{n},temperature_2m_spread_previous_day{n}" for n in (1, 2, 3))
    for city, (_, _, lat, lon) in CITIES.items():
        parts = []
        for m in ENS_MODELS:
            j = get_json(ENS, {"latitude": lat, "longitude": lon, "models": m, "hourly": hourly,
                               "start_date": start, "end_date": end, "timezone": "UTC"})
            if not j:
                print(f"{city} {m}: FAILED"); continue
            time.sleep(2)
            h = j["hourly"]
            t = to_unix_seconds(pd.Series(h["time"])).to_numpy()
            for n in (1, 2, 3):
                parts.append(pd.DataFrame({"model": m, "t": t, "lead_days": n,
                                           "mean": h[f"temperature_2m_previous_day{n}"],
                                           "spread": h[f"temperature_2m_spread_previous_day{n}"]}))
        df = pd.concat(parts).dropna(subset=["mean"])
        df.to_parquet(os.path.join(OUT, f"ens_{city}.parquet"), index=False)
        print(f"{city}: {df['model'].nunique()} models, {len(df)} rows")


if __name__ == "__main__":
    {"markets": markets, "prices": prices, "ensemble": ensemble}[sys.argv[1]]()
