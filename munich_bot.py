import json
import re
import argparse
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import norm
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry

from db import init_db, get_connection, latest_leadtime_sigma
from station_obs import fetch_daily_max as fetch_station_daily_max, observed_running_max
from daymax import hourly_times_local, daily_max_by_date, daily_max_for
from timeutil import utc_now_iso, berlin_now, berlin_today, hours_until_local_day_start, new_snapshot_id
from kelly_multi import Candidate, solve_joint_kelly
from http_session import TimeoutSession, TimeoutCachedSession

# --- CONFIGURATION ---
LATITUDE = 48.3537   # Munich Airport (EDDM)
LONGITUDE = 11.7860
ROLLING_WINDOW_DAYS = 30
MIN_EDGE_THRESHOLD = 0.03   # 3% minimum net probability edge after fees

# POLYMARKET TAKER FEE
# Formula (docs.polymarket.com/trading/fees, checked 25 Nov 2026):
#     fee = shares * feeRate * (p * (1 - p)) ** exponent      (exponent is 1 today)
# Only takers pay; makers pay 0. Weather's rate dropped from 0.05 to 0.01 on
# 20 Nov 2026 -- Polymarket says not to hardcode it, so the bot reads each
# market's own `feeSchedule` from Gamma (rate, exponent). This value is only
# the fallback if a market doesn't publish one.
DEFAULT_FEE_RATE = 0.01
DEFAULT_FEE_EXPONENT = 1.0

# CAPITAL & RISK PARAMETERS
BANKROLL = 1000.0
KELLY_FRACTION = 0.25

MODELS = {
    "ecmwf_ifs025": "ECMWF",
    "icon_seamless": "DWD ICON",
    "gfs_seamless": "NOAA GFS",
    "gem_global": "CMC GEM",
    "meteofrance_seamless": "Météo-France",
}

# Two Open-Meteo clients:
#  - openmeteo_live: NO cache. Live forecasts must be fresh: with a 1-hour
#    cache a new model run could be ignored for up to an hour, which would
#    wreck any timing analysis ("when did our model react to a new run?").
#  - openmeteo_hist: cached for 1 hour. Historical data (30-day blend
#    calibration) doesn't change between runs, so caching only saves calls.
# Both sessions apply a default timeout (http_session.py): without one, a
# stalled Open-Meteo call blocked the whole run indefinitely.
openmeteo_live = openmeteo_requests.Client(
    session=retry(TimeoutSession(), retries=3, backoff_factor=0.3))
openmeteo_hist = openmeteo_requests.Client(
    session=retry(TimeoutCachedSession('.cache_phase5_kelly', expire_after=3600),
                  retries=3, backoff_factor=0.3))


def get_horizon_sigma(lead_time_hours):
    """
    REPLACES the hardcoded {0.53, 0.51, 0.46, 0.43} dict with a live
    read from leadtime_calibration, populated by a REAL measurement in
    Lead_Time_Sensitivity.py (Previous Runs API) instead of the old
    fabricated linear scale factor. If that script hasn't been run
    yet, this prints a loud warning and falls back to a conservative
    default rather than silently trusting a stale number.
    """
    return latest_leadtime_sigma(lead_time_hours)


def parse_label_to_bounds(label):
    """Dynamically parses outcome bracket labels into numeric bounds.
    Examples:
        '16°C or below' -> (None, 16.5)
        '17°C'          -> (16.5, 17.5)
        '26°C or higher' -> (25.5, None)
    """
    # Normalise unicode minus/dashes so negative temps keep their sign.
    clean_label = (label.replace("°C", "").replace("°", "")
                   .replace("\u2212", "-").replace("\u2013", "-").strip())
    # Sign is optional on BOTH integers and decimals. (The old pattern
    # r"[-+]?\d*\.\d+|\d+" dropped the minus on integers, so
    # "-5°C or below" parsed as 5 -> ~100% model probability.)
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", clean_label)
    if not numbers:
        return None, None
    val = float(numbers[0])

    label_lower = label.lower()
    if any(w in label_lower for w in ["below", "lower", "less", "under"]):
        return None, val + 0.5
    elif any(w in label_lower for w in ["higher", "above", "more", "greater"]):
        return val - 0.5, None
    else:
        return val - 0.5, val + 0.5


def calculate_bracket_probability(mean_temp, sigma, bounds, observed_floor=None):
    """
    P(final daily max falls in the bracket), with the forecast max
    ~ Normal(mean_temp, sigma).

    observed_floor (same-day markets only): the max ALREADY reported today,
    in whole degrees. The final value is max(observed, rest-of-day max),
    so it can never end below the floor:
        P(final <= t) = 0                      if t < floor
                      = Phi((t - mean) / sigma) otherwise
    Example: 2°C already observed, rest of day forecast 1.0 +/- 0.6.
      bracket 1°C  [0.5, 1.5] -> 0 (below what's already happened)
      bracket 2°C  [1.5, 2.5] -> Phi((2.5-1.0)/0.6) - 0 = 0.994
      bracket 3°C  [2.5, 3.5] -> Phi(4.17) - Phi(2.5) = 0.006
    """
    def cdf(t):
        if t is None:
            return 1.0
        if observed_floor is not None and t < observed_floor:
            return 0.0
        if mean_temp is None:            # nothing left to forecast: only the observation counts
            return 1.0
        return norm.cdf((t - mean_temp) / sigma)

    t_min, t_max = bounds
    lower = 0.0 if t_min is None else cdf(t_min)
    return max(0.0, cdf(t_max) - lower)


def taker_fee_per_share(price, fee_rate=DEFAULT_FEE_RATE, exponent=DEFAULT_FEE_EXPONENT):
    """Fee per share when buying at `price` as a taker (formula above).
    Example: 0.01 * (0.5 * 0.5) = 0.0025, i.e. 0.25 cents per share at 50c."""
    return fee_rate * (price * (1.0 - price)) ** exponent


def decide_side_and_edge(model_prob, best_bid, best_ask,
                         fee_rate=DEFAULT_FEE_RATE, exponent=DEFAULT_FEE_EXPONENT):
    """
    Decides YES / NO / no trade for one bracket using prices we could
    ACTUALLY trade at, not the midpoint:
      - buying YES costs the best YES ask
      - buying NO costs 1 - best YES bid. For these linked (neg-risk)
        markets the NO book is an exact mirror of the YES book (checked
        live 25 Nov 2026: NO ask 0.45 = 1 - YES bid 0.55), so this equals
        the NO token's own best ask.
    Worked example: model says 40%, book is bid 0.30 / ask 0.34, fee rate 0.01.
      YES: cost 0.34 + fee 0.01*0.34*0.66 = 0.3422 -> edge 0.40-0.3422 = +5.8%
      NO:  cost 0.70 + fee 0.01*0.70*0.30 = 0.7021 -> edge 0.60-0.7021 = -10.2%
      -> BUY YES (edge above the 3% threshold).
    Returns (signal, net_edge, side, cost_per_share). Sizing happens
    jointly across the event in kelly_multi.solve_joint_kelly.
    """
    net_edge_yes = net_edge_no = float("-inf")
    cost_yes = cost_no = None

    if best_ask is not None and 0.0 < best_ask < 1.0:
        cost_yes = best_ask + taker_fee_per_share(best_ask, fee_rate, exponent)
        net_edge_yes = model_prob - cost_yes
    if best_bid is not None and 0.0 < best_bid < 1.0:
        price_no = 1.0 - best_bid
        cost_no = price_no + taker_fee_per_share(price_no, fee_rate, exponent)
        net_edge_no = (1.0 - model_prob) - cost_no

    if cost_yes is None and cost_no is None:
        return "NO TRADE", 0.0, None, None   # empty book on both sides

    if net_edge_yes >= MIN_EDGE_THRESHOLD and net_edge_yes >= net_edge_no:
        return "BUY YES", net_edge_yes, "YES", cost_yes
    elif net_edge_no >= MIN_EDGE_THRESHOLD:
        return "BUY NO", net_edge_no, "NO", cost_no

    best_edge = max(net_edge_yes, net_edge_no)
    return "NO TRADE", best_edge, None, None


def fetch_live_ensemble_forecast(target_date_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": target_date_str, "end_date": target_date_str,
        "hourly": "temperature_2m", "models": list(MODELS.keys()),
        "timezone": "Europe/Berlin",
    }
    responses = openmeteo_live.weather_api(url, params=params)
    live_maxes, remaining_maxes = [], []
    per_model_max = {}
    # Start of the current hour in Munich: hours from here on haven't been
    # observed yet. Only matters when the target day is today.
    hour_start = pd.Timestamp(berlin_now()).floor("h")

    for model_key, response in zip(MODELS.keys(), responses):
        hourly = response.Hourly()
        vals = hourly.Variables(0).ValuesAsNumpy()
        ts_local = hourly_times_local(hourly, len(vals))
        # Full local day (00:00-23:59), same as the market and the station truth.
        m = daily_max_for(target_date_str, ts_local, vals)
        live_maxes.append(m)
        per_model_max[model_key] = float(m)
        # Max over the hours of the target day still to come (NaN if none).
        on_day = pd.DatetimeIndex(ts_local).strftime("%Y-%m-%d") == target_date_str
        rest = vals[on_day & (pd.DatetimeIndex(ts_local) >= hour_start)]
        rest = rest[~np.isnan(rest)]
        remaining_maxes.append(float(rest.max()) if len(rest) else float("nan"))

    return np.array(live_maxes), per_model_max, np.array(remaining_maxes)


def log_forecast_snapshot(target_date_str, per_model_max):
    """Logs raw per-model DETERMINISTIC forecasts -- call this every
    3-6h, not every run, to match actual model-run cadence (see
    db.py docstring)."""
    conn = get_connection()
    cur = conn.cursor()
    ts = utc_now_iso()
    for model_key, val in per_model_max.items():
        cur.execute("""
            INSERT INTO forecast_snapshots (target_date, model_key, daily_max_temp_c, ts_utc)
            VALUES (?, ?, ?, ?)
        """, (target_date_str, model_key, val, ts))
    conn.commit()
    conn.close()


# Ensemble sources, corrected against Open-Meteo's actual documented
# parameter names (my earlier list had two wrong identifiers: GFS-ENS
# is "ncep_gefs025", not "gfs025"; ECMWF-ENS needs the "_ensemble"
# suffix -- "ecmwf_ifs025_ensemble", not "ecmwf_ifs025").
#
# meteoswiss_icon_ch1/ch2 and ukmo_uk_ensemble_2km were initially left
# out over domain-coverage uncertainty -- confirmed via direct check
# that both return real values for Munich's coordinates, so they're
# included now. Worth keeping in mind these are high-res regional
# models where Munich sits closer to the domain edge than their
# center, so treat them as legitimate extra signal, not necessarily
# as reliable as the models built around Germany specifically
# (icon_d2_eps, icon_eu_eps).
#
# Deliberately still left out: icon_seamless_eps, ncep_gefs_seamless,
# ncep_gefs05 -- these are lower-resolution or blended variants of
# models already covered at higher fidelity below (icon_d2_eps +
# icon_eu_eps + icon_global_eps already span ICON's seamless range;
# ncep_gefs025 is GEFS at its best available resolution). Adding the
# lower-res versions alongside the higher-res ones would mostly just
# duplicate signal, not add new information -- revisit if that
# assumption turns out wrong once you have real data to check it against.
#
# Member counts are NOT hardcoded anywhere in the fetch/reduction
# logic below (it reads hourly.VariablesLength() live), so this list
# can be edited freely without touching any other code.
ENSEMBLE_MODELS = {
    "icon_d2_eps": "ICON-D2-EPS",                        # DWD, Germany-domain, highest res
    "icon_eu_eps": "ICON-EU-EPS",                          # DWD, Europe-domain
    "icon_global_eps": "ICON-Global-EPS",                  # DWD, global
    "meteoswiss_icon_ch1_ensemble": "MeteoSwiss-ICON-CH1",  # confirmed covers Munich
    "meteoswiss_icon_ch2_ensemble": "MeteoSwiss-ICON-CH2",  # confirmed covers Munich
    "ncep_gefs025": "GFS-ENS",                              # NOAA, global, 0.25deg
    "ncep_aigefs025": "AI-GEFS",                            # NOAA, AI-based variant
    "ecmwf_ifs025_ensemble": "ECMWF-ENS",                  # global, 0.25deg
    "ecmwf_ifs_europe_ensemble": "ECMWF-ENS-Europe",       # higher-res Europe domain
    "ecmwf_aifs025_ensemble": "ECMWF-AIFS",                # ECMWF's AI-based ensemble
    "ecmwf_aifs_europe_ensemble": "ECMWF-AIFS-Europe",     # AI ensemble, Europe domain
    "gem_global_ensemble": "GEM-ENS",                       # Environment Canada, global
    "google_weathernext2_ensemble": "GraphCast-ENS",        # Google's AI ensemble
    "ukmo_uk_ensemble_2km": "UKMO-UK-2km",                  # confirmed covers Munich
    "ukmo_global_ensemble_20km": "UKMO-Global",             # UK Met Office, global
}


def fetch_and_log_all_ensemble_members(target_date_str):
    """
    Logs EVERY individual raw ensemble member (not just the blended
    point forecast) to forecast_snapshots, purely as insurance for
    future use. The live prediction pipeline
    (fetch_live_ensemble_forecast + SLSQP blend + Gaussian sigma) is
    UNCHANGED and does not read from this -- this is pure logging,
    additive, and safe to fail without affecting predictions.

    Why log it now even though nothing consumes it yet: this history
    cannot be reconstructed retroactively. If a future backtest wants
    to build an empirical (non-Gaussian) probability distribution
    instead of assuming a bell curve, it needs raw members from every
    day that's already passed -- logging starts the clock on that,
    the live Gaussian pipeline doesn't need to change to benefit later.
    """
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": target_date_str, "end_date": target_date_str,
        "hourly": "temperature_2m",
        "models": list(ENSEMBLE_MODELS.keys()),
        "timezone": "Europe/Berlin",
    }
    try:
        responses = openmeteo_live.weather_api(url, params=params)
    except Exception as e:
        print(f"[ensemble-log] fetch failed for {target_date_str}: {e}")
        return 0

    conn = get_connection()
    cur = conn.cursor()
    total_logged = 0
    ts = utc_now_iso()

    for model_key, response in zip(ENSEMBLE_MODELS.keys(), responses):
        hourly = response.Hourly()
        num_members = hourly.VariablesLength()
        model_logged = 0

        for i in range(num_members):
            vals = hourly.Variables(i).ValuesAsNumpy()
            # Full local day; NaN if the member doesn't cover enough of it.
            member_max = daily_max_for(target_date_str, hourly_times_local(hourly, len(vals)), vals)
            if np.isnan(member_max):
                continue
            cur.execute("""
                INSERT INTO forecast_snapshots (target_date, model_key, daily_max_temp_c, ts_utc)
                VALUES (?, ?, ?, ?)
            """, (target_date_str, f"{model_key}_member{i:02d}", member_max, ts))
            total_logged += 1
            model_logged += 1

        print(f"[ensemble-log]   {model_key} ({ENSEMBLE_MODELS[model_key]}): "
              f"{model_logged}/{num_members} members logged")

    conn.commit()
    conn.close()
    print(f"[ensemble-log] logged {total_logged} raw member rows for {target_date_str} "
          f"across {len(ENSEMBLE_MODELS)} ensemble sources "
          f"(per-source breakdown printed below if you need to sanity-check counts)")
    return total_logged


def fetch_calibration_data(target_date_str):
    now = berlin_now().replace(tzinfo=None)   # Munich calendar, naive for date maths below
    safe_end_dt = now - timedelta(days=2)

    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    end_dt = min(target_dt - timedelta(days=1), safe_end_dt)
    start_dt = end_dt - timedelta(days=ROLLING_WINDOW_DAYS)

    start_str, end_str = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    # Fit the blend against the EDDM station max (what markets resolve
    # on), so the bias term absorbs the grid-vs-station offset.
    actuals = fetch_station_daily_max(start_str, end_str)

    url_m = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params_m = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": start_str, "end_date": end_str,
        "hourly": "temperature_2m", "models": list(MODELS.keys()),
        "timezone": "Europe/Berlin",
    }
    resp_m = openmeteo_hist.weather_api(url_m, params=params_m)

    model_daily_max = {m: {} for m in MODELS.keys()}
    for model_key, response in zip(MODELS.keys(), resp_m):
        hourly = response.Hourly()
        vals = hourly.Variables(0).ValuesAsNumpy()
        # Full local day per date; incomplete days (NaN) are left out.
        per_day = daily_max_by_date(hourly_times_local(hourly, len(vals)), vals)
        model_daily_max[model_key] = {d: v for d, v in per_day.items() if not np.isnan(v)}

    valid_dates = [d for d in sorted(actuals.keys()) if all(d in model_daily_max[m] for m in MODELS.keys())]
    y_tr = np.array([actuals[d] for d in valid_dates])
    X_tr = np.column_stack([[model_daily_max[m][d] for d in valid_dates] for m in MODELS.keys()])
    return X_tr, y_tr


def discover_open_munich_events():
    """
    Replaces the old approach of computing ONE guessed target_date
    (now + 2 days) and building a single slug for it. That silently
    missed the fact that Polymarket weather markets for different
    target dates run IN PARALLEL -- a new market opens at 00:56 ET,
    exactly 2 days before its target date, while the previous day's
    market is often still open too.

    Instead, this searches Gamma for every currently OPEN
    ("closed": false) Munich temperature event, regardless of how many
    there are. Running this every 15 minutes means a newly-opened
    market gets picked up within 15 minutes of going live, without
    needing to hardcode the exact 00:56 ET opening timestamp anywhere
    -- discovery-based polling naturally handles it.

    Returns a list of dicts: {event_slug, title, target_date (date obj
    or None if unparseable), brackets: {label: {token_id, question}}}
    """
    url = "https://gamma-api.polymarket.com/events"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    # Gamma's /events endpoint IGNORES a "search" param (it silently
    # returns unrelated open events), so look events up by their exact
    # slug instead, for every date a market could currently be open:
    # today through +3 days (markets open ~2 days ahead, and the
    # previous day's market can still be open in parallel).
    events = []
    today = berlin_today()   # markets are named after Munich local dates
    for offset in range(0, 4):
        d = today + timedelta(days=offset)
        slug = f"highest-temperature-in-munich-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"
        try:
            response = requests.get(url, headers=headers, params={"slug": slug}, timeout=15)
            response.raise_for_status()
            found = response.json()
        except Exception as e:
            print(f"[discovery] Failed to fetch {slug}: {e}")
            continue
        for ev in found or []:
            if not ev.get("closed", False):
                events.append(ev)
    print(f"[discovery] {len(events)} open Munich event(s) found")

    results = []
    for event in events:
        title = event.get("title", "")
        slug = event.get("slug", "")
        if "munich" not in title.lower():
            continue  # Gamma search is fuzzy, filter defensively

        target_date = _parse_target_date_from_slug(slug)

        brackets = {}
        for m in event.get("markets", []):
            label = m.get("groupItemTitle") or m.get("question")
            question = m.get("question", label)
            token_ids = m.get("clobTokenIds")
            prices_raw = m.get("outcomePrices")
            if not token_ids:
                continue
            tokens = json.loads(token_ids) if isinstance(token_ids, str) else token_ids
            token_id = tokens[0]                      # YES token (outcomes are ["Yes", "No"])
            no_token_id = tokens[1] if len(tokens) > 1 else None
            price = None
            if prices_raw:
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                if len(prices) > 0:
                    price = float(prices[0])
            # Fee parameters published on the market itself (checked live:
            # feeSchedule = {"rate": 0.01, "exponent": 1, "takerOnly": true, ...}).
            fs = m.get("feeSchedule") or {}
            fees_on = m.get("feesEnabled", True)
            fee_rate = float(fs.get("rate", DEFAULT_FEE_RATE)) if fees_on else 0.0
            fee_exp = float(fs.get("exponent", DEFAULT_FEE_EXPONENT))
            brackets[label] = {"token_id": token_id, "no_token_id": no_token_id,
                               "question": question, "price": price,
                               "fee_rate": fee_rate, "fee_exponent": fee_exp}

        if brackets:
            results.append({
                "event_slug": slug, "title": title,
                "target_date": target_date, "brackets": brackets,
            })

    return results


def _parse_target_date_from_slug(slug: str):
    """Extracts the target date from slugs like
    'highest-temperature-in-munich-on-august-24-2026'. Returns a
    datetime.date, or None if the slug doesn't match the expected
    pattern (logged, not silently swallowed, since a parsing miss
    means lead-time-hours can't be computed for that event)."""
    match = re.search(r"on-([a-z]+)-(\d{1,2})-(\d{4})$", slug.lower())
    if not match:
        print(f"[discovery] WARNING: could not parse target date from slug '{slug}'")
        return None
    month_str, day_str, year_str = match.groups()
    try:
        return datetime.strptime(f"{month_str} {day_str} {year_str}", "%B %d %Y").date()
    except ValueError:
        print(f"[discovery] WARNING: date parse failed for slug '{slug}'")
        return None


def fetch_orderbook_depth(token_id: str):
    """
    Fetches full L2 order book depth (all visible price levels, not
    just best bid/ask) for one outcome token via Polymarket's CLOB API.
    This is the piece needed to actually test 'could we have gotten
    filled at this size' in a backtest, instead of assuming infinite
    liquidity at the last-trade price.

    Endpoint per Polymarket's CLOB API docs -- verify against a live
    response on first run, response shape can vary.
    """
    url = "https://clob.polymarket.com/book"
    try:
        response = requests.get(url, params={"token_id": token_id}, timeout=10)
        response.raise_for_status()
        book = response.json()
        return book.get("bids", []), book.get("asks", [])
    except Exception as e:
        print(f"[orderbook] Failed to fetch depth for token {token_id}: {e}")
        return [], []


def top_of_book(bids, asks, depth_window=0.05):
    """
    Best bid / best ask / midpoint from a raw CLOB book, plus how many
    shares sit within `depth_window` (5 cents) of each best price.
    The CLOB returns bids sorted ascending and asks descending (best
    price LAST), but we take max/min explicitly rather than rely on order.
    Returns a dict; values are None when a side is empty.
    """
    bid_px = [(float(l["price"]), float(l["size"])) for l in bids]
    ask_px = [(float(l["price"]), float(l["size"])) for l in asks]
    best_bid = max((p for p, _ in bid_px), default=None)
    best_ask = min((p for p, _ in ask_px), default=None)
    mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
    bid_depth = sum(sz for p, sz in bid_px if best_bid is not None and p >= best_bid - depth_window - 1e-9)
    ask_depth = sum(sz for p, sz in ask_px if best_ask is not None and p <= best_ask + depth_window + 1e-9)
    return {"best_bid": best_bid, "best_ask": best_ask, "mid": mid,
            "bid_depth_5c": bid_depth if best_bid is not None else None,
            "ask_depth_5c": ask_depth if best_ask is not None else None}


def log_orderbook_depth(event_slug, question, bids, asks, token_id=None, bracket_label=None,
                        snapshot_id=None, ts_utc=None):
    """Logs every visible price level, not just top-of-book. Cadence:
    call this from the 15-min prices-only path -- see db.py docstring,
    this was upgraded from the original 'hourly to start' plan since
    it now piggybacks on the same discovery pass as price logging at
    no extra API-call cost."""
    conn = get_connection()
    cur = conn.cursor()
    rows = [(event_slug, question, side, float(level.get("price", 0)), float(level.get("size", 0)),
             token_id, bracket_label, snapshot_id, ts_utc)
            for side, levels in (("bid", bids), ("ask", asks)) for level in levels]
    if rows:
        cur.executemany("""
            INSERT INTO orderbook_depth (event_slug, market_question, side, price_level, size,
                                         token_id, bracket_label, snapshot_id, ts_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    conn.commit()
    conn.close()


def log_price_snapshot(event_slug, brackets, snapshot_id, ts_utc, books=None):
    """--prices-only path: one row per bracket, all sharing the same
    snapshot_id and UTC timestamp. All prices are probabilities 0-1:
    gamma_price = Gamma's displayed price (a midpoint), best_bid /
    best_ask / mid = CLOB top of book (what you could actually trade at)."""
    books = books or {}
    conn = get_connection()
    cur = conn.cursor()
    rows = []
    for label, b in brackets.items():
        tob = books.get(label) or {}
        if b["price"] is None and not tob:
            continue
        rows.append((event_slug, b["question"], label, b["token_id"], b["price"],
                     tob.get("best_bid"), tob.get("best_ask"), tob.get("mid"),
                     tob.get("bid_depth_5c"), tob.get("ask_depth_5c"), snapshot_id, ts_utc))
    if rows:
        cur.executemany("""
            INSERT INTO price_snapshots (event_slug, market_question, bracket_label, token_id,
                                         gamma_price, best_bid, best_ask, mid,
                                         bid_depth_5c, ask_depth_5c, snapshot_id, ts_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    conn.commit()
    conn.close()
    return len(rows)


def run_prices_only_all_open_events():
    """
    NEW default for --prices-only: instead of computing one guessed
    target_date and fetching just that market, discovers and logs
    EVERY currently open Munich event -- so markets running in
    parallel (today's + tomorrow's newly-opened one) both get logged
    automatically, and order book depth is captured alongside price
    at no extra discovery cost.
    """
    events = discover_open_munich_events()
    if not events:
        print("[prices-only] no open Munich events found this run.")
        return

    for event in events:
        # Everything logged for this event in this pass shares one id + time.
        snapshot_id, ts = new_snapshot_id(), utc_now_iso()

        books = {}
        for label, b in event["brackets"].items():
            bids, asks = fetch_orderbook_depth(b["token_id"])
            if bids or asks:
                books[label] = top_of_book(bids, asks)
                log_orderbook_depth(event["event_slug"], b["question"], bids, asks,
                                    token_id=b["token_id"], bracket_label=label,
                                    snapshot_id=snapshot_id, ts_utc=ts)

        n_prices = log_price_snapshot(event["event_slug"], event["brackets"], snapshot_id, ts, books)

        print(f"[prices-only] {event['event_slug']} (target_date={event['target_date']}): "
              f"logged {n_prices} price rows + depth for {len(event['brackets'])} brackets "
              f"| snapshot {snapshot_id[:8]} at {ts}")


def log_predictions(target_date_str, event_slug, questions_by_label, rows,
                     calibrated_mean, sigma, lead_time_hours, tokens_by_label=None):
    conn = get_connection()
    cur = conn.cursor()
    snapshot_id, ts = new_snapshot_id(), utc_now_iso()   # one id/time for this event's rows
    tokens_by_label = tokens_by_label or {}
    for r in rows:
        cur.execute("""
            INSERT INTO predictions (
                target_date, event_slug, market_question, bracket_label,
                model_prob, market_price, net_edge, signal, kelly_fraction,
                bet_amount, calibrated_mean_temp, sigma_used, lead_time_hours,
                token_id, snapshot_id, ts_utc, best_bid, best_ask, entry_cost, fee_rate,
                observed_max_so_far
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            target_date_str, event_slug, questions_by_label.get(r["label"], r["label"]), r["label"],
            r["model_prob"], r["market_price"], r["net_edge"], r["signal"], r["kelly_fraction"],
            r["bet_amount"], calibrated_mean, sigma, lead_time_hours,
            tokens_by_label.get(r["label"]), snapshot_id, ts,
            r.get("best_bid"), r.get("best_ask"), r.get("entry_cost"), r.get("fee_rate"),
            r.get("observed_max_so_far"),
        ))
    conn.commit()
    conn.close()
    print(f"[DB] Logged {len(rows)} prediction rows for {target_date_str}")


def _is_valid_whole_degree(value):
    """True only for a real whole-number temperature (not None / NaN / float junk)."""
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    return isinstance(value, float) and not np.isnan(value) and value.is_integer()


def _bracket_outcome(label, actual_temp):
    """1 if the whole-degree actual_temp falls in this bracket, else 0."""
    t_min, t_max = parse_label_to_bounds(label)
    if t_min is None:
        return 1 if actual_temp <= t_max else 0
    if t_max is None:
        return 1 if actual_temp >= t_min else 0
    return 1 if t_min <= actual_temp <= t_max else 0


def settle_unsettled_markets():
    """
    Settles every past target date in two independent ways:
      1. Station outcome (for calibration / Brier): EDDM routine-METAR max,
         exactly per Polymarket's rules (station_obs.py). A date is only
         settled if a valid whole-degree max exists (A3: never settle on
         a missing or NaN observation).
      2. Polymarket outcome (for P&L): the winning bracket as Polymarket
         actually resolved it (A4), read from Gamma once resolved.
    Then flags any disagreement between the two and audits old rows.
    """
    settle_from_station()
    settle_from_polymarket()
    audit_settlement()


def settle_from_station():
    conn = get_connection()
    cur = conn.cursor()
    today_str = berlin_today().strftime("%Y-%m-%d")   # only settle Munich days that are over
    cur.execute("""
        SELECT DISTINCT target_date FROM predictions
        WHERE settled = 0 AND target_date < ?
    """, (today_str,))
    dates = sorted(d for (d,) in cur.fetchall())
    if not dates:
        print("[Settlement] No past unsettled predictions found.")
        conn.close()
        return

    station_max = fetch_station_daily_max(dates[0], dates[-1], verbose=True)

    for t_date in dates:
        actual_temp = station_max.get(t_date)
        if not _is_valid_whole_degree(actual_temp):
            print(f"[Settlement] No valid EDDM max for {t_date} (got {actual_temp!r}) -- leaving unsettled")
            continue
        actual_temp = int(actual_temp)
        print(f"[Settlement] EDDM routine-METAR max on {t_date}: {actual_temp}°C")

        cur.execute("""
            SELECT id, bracket_label, model_prob FROM predictions
            WHERE target_date = ? AND settled = 0
        """, (t_date,))
        for row_id, label, model_prob in cur.fetchall():
            outcome = _bracket_outcome(label, actual_temp)
            brier = (model_prob - outcome) ** 2
            cur.execute("""
                UPDATE predictions
                SET actual_max_temp = ?, resolved_outcome = ?, brier_score = ?, settled = 1
                WHERE id = ?
            """, (actual_temp, outcome, round(brier, 4), row_id))
        conn.commit()
        print(f"[Settlement] Settled all rows for {t_date}")
    conn.close()


def fetch_polymarket_resolution(event_slug):
    """
    Returns the winning bracket label as Polymarket resolved it, or None
    if the event isn't fully resolved yet. Gamma shape (checked live):
    each bracket market has outcomes '["Yes","No"]' and, once resolved,
    outcomePrices '["1","0"]' for the winner and '["0","1"]' for the rest,
    plus closed=true and umaResolutionStatus='resolved'.
    """
    r = requests.get("https://gamma-api.polymarket.com/events",
                     params={"slug": event_slug}, timeout=15)
    r.raise_for_status()
    events = r.json()
    if not events:
        return None
    markets = events[0].get("markets", [])
    if not markets or not all(m.get("closed") for m in markets):
        return None
    winners = []
    for m in markets:
        prices = m.get("outcomePrices")
        prices = json.loads(prices) if isinstance(prices, str) else (prices or [])
        if prices and str(prices[0]) == "1":
            winners.append(m.get("groupItemTitle") or m.get("question"))
    if len(winners) != 1:
        print(f"[PM-settle] {event_slug}: expected 1 winner, found {winners} -- skipping")
        return None
    return winners[0]


def settle_from_polymarket():
    conn = get_connection()
    cur = conn.cursor()
    today_str = berlin_today().strftime("%Y-%m-%d")
    cur.execute("""
        SELECT DISTINCT event_slug FROM predictions
        WHERE pm_outcome IS NULL AND target_date < ? AND event_slug IS NOT NULL
    """, (today_str,))
    slugs = [s for (s,) in cur.fetchall()]
    for slug in slugs:
        try:
            winner = fetch_polymarket_resolution(slug)
        except Exception as e:
            print(f"[PM-settle] {slug}: Gamma fetch failed: {e}")
            continue
        if winner is None:
            print(f"[PM-settle] {slug}: not resolved yet")
            continue
        cur.execute("""
            UPDATE predictions
            SET pm_outcome = CASE WHEN bracket_label = ? THEN 1 ELSE 0 END,
                pm_winning_bracket = ?
            WHERE event_slug = ? AND pm_outcome IS NULL
        """, (winner, winner, slug))
        conn.commit()
        print(f"[PM-settle] {slug}: Polymarket winner = {winner}")

    # Flag rows where the station-derived outcome disagrees with Polymarket.
    cur.execute("""
        UPDATE predictions
        SET outcome_mismatch = CASE WHEN resolved_outcome = pm_outcome THEN 0 ELSE 1 END
        WHERE settled = 1 AND pm_outcome IS NOT NULL AND outcome_mismatch IS NULL
    """)
    conn.commit()
    cur.execute("""
        SELECT DISTINCT target_date, actual_max_temp, pm_winning_bracket FROM predictions
        WHERE outcome_mismatch = 1
    """)
    for t_date, actual, winner in cur.fetchall():
        print(f"[PM-settle] MISMATCH {t_date}: station max {actual}°C but Polymarket winner '{winner}'")
    conn.close()


def audit_settlement():
    """A3: report already-settled rows that have no valid observation."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT target_date, actual_max_temp FROM predictions WHERE settled = 1")
    bad = {}
    for t_date, actual in cur.fetchall():
        if actual is None or (isinstance(actual, float) and np.isnan(actual)):
            bad[t_date] = bad.get(t_date, 0) + 1
    conn.close()
    if bad:
        print(f"[audit] WARNING: {sum(bad.values())} settled row(s) with missing/NaN actual_max_temp: {bad}")
    else:
        print("[audit] OK: no settled rows with a missing/NaN observation.")


def run_scanner_for_all_open_events():
    """
    Replaces the old single-market call. Discovers every currently
    open Munich event, and runs the full calibration+scan+Kelly+log
    flow separately for EACH one -- calibration/live-forecast/sigma
    all depend on that specific event's target_date, and Kelly sizing
    must stay scoped to one event's mutually-exclusive brackets, not
    mixed across events.
    """
    events = discover_open_munich_events()
    if not events:
        print("[full-run] no open Munich events found this run.")
        return

    for event in events:
        if event["target_date"] is None:
            print(f"[full-run] skipping {event['event_slug']} -- couldn't parse target date")
            continue

        target_date_str = event["target_date"].strftime("%Y-%m-%d")
        # Real lead time to this specific event's target date, instead
        # of a hardcoded 48h assumption -- matters most right after a
        # new market opens (near 62h out) vs one about to resolve.
        lead_time_hours = hours_until_local_day_start(event["target_date"])

        # One market failing (e.g. an API timeout) must not stop the others.
        try:
            run_phase5_scanner(target_date_str=target_date_str, lead_time_hours=lead_time_hours,
                                pre_discovered_event=event)
        except Exception as e:
            print(f"[full-run] {event['event_slug']} FAILED, skipping to the next market: "
                  f"{type(e).__name__}: {str(e)[:200]}")


def run_phase5_scanner(target_date_str, lead_time_hours, pre_discovered_event=None):
    print("\n" + "=" * 105)
    print(f" PHASE 5: POLYMARKET EDGE & KELLY POSITION SCANNER | TARGET DATE: {target_date_str}")
    print("=" * 105)
    print(f"[*] Trading Parameters: Capital = ${BANKROLL:,.2f} | Risk = Quarter-Kelly ({KELLY_FRACTION:.2f}x) "
          f"| Taker fee: per-market feeSchedule (fallback rate {DEFAULT_FEE_RATE})")

    print(f"[*] Calibrating weights over prior {ROLLING_WINDOW_DAYS} days...")
    X_tr, y_tr = fetch_calibration_data(target_date_str)

    num_models = X_tr.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w[:num_models]) - 1.0})
    bounds = [(0.0, 1.0)] * num_models + [(-2.0, 2.0)]
    initial_guess = [1.0 / num_models] * num_models + [0.0]

    def loss_func(params):
        w, b = params[:num_models], params[num_models]
        return np.mean((y_tr - (np.dot(X_tr, w) + b)) ** 2)

    res = minimize(loss_func, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_w, opt_b = res.x[:num_models], res.x[num_models]
    print(f"[*] Calibration Complete | ECMWF Weight: {opt_w[0]*100:.1f}% | Station Bias: {opt_b:+.2f} °C")

    live_X, per_model_max, remaining_X = fetch_live_ensemble_forecast(target_date_str)
    log_forecast_snapshot(target_date_str, per_model_max)  # every run here; cron this at 3-6h cadence, not every scan
    fetch_and_log_all_ensemble_members(target_date_str)     # raw member logging -- additive, doesn't feed live prediction

    if np.isnan(live_X).any():
        missing = [m for m, v in per_model_max.items() if np.isnan(v)]
        print(f"[!] SKIPPING {target_date_str}: no full-day forecast from {missing} "
              f"(target day not fully inside their forecast range yet).")
        return

    calibrated_mean = np.dot(live_X, opt_w) + opt_b
    sigma = get_horizon_sigma(lead_time_hours)  # measured, from leadtime_calibration
    print(f"[*] Live Forecast Peak: {calibrated_mean:.2f} °C (Horizon: {lead_time_hours:.1f}h, "
          f"Measured Sigma: {sigma:.2f} °C)")

    # Same-day markets: part of the day has already happened. The final
    # max = max(observed so far, max of the remaining hours), so we use the
    # official routine-METAR running max as a floor and forecast only the
    # hours still to come (see calculate_bracket_probability).
    observed_floor = None
    forecast_mean_for_probs = calibrated_mean
    if target_date_str == berlin_today().strftime("%Y-%m-%d"):
        try:
            obs = observed_running_max(berlin_today())
        except Exception as e:
            obs = None
            print(f"[same-day] could not read today's METARs ({e}) -- using full-day forecast only")
        if obs is not None:
            observed_floor, n_reports, last_time = obs
            if np.isnan(remaining_X).any():
                forecast_mean_for_probs = None   # no hours left: the observation decides
                print(f"[same-day] observed max so far {observed_floor}°C ({n_reports} reports, "
                      f"last {last_time:%H:%M}); no forecast hours left")
            else:
                forecast_mean_for_probs = float(np.dot(remaining_X, opt_w) + opt_b)
                print(f"[same-day] observed max so far {observed_floor}°C ({n_reports} reports, "
                      f"last {last_time:%H:%M}); rest-of-day forecast max {forecast_mean_for_probs:.2f}°C")

    print("\n[*] Fetching Polymarket order book...")
    if pre_discovered_event is not None:
        # Reuse the event already fetched by discover_open_munich_events()
        # in run_scanner_for_all_open_events() -- avoids a redundant
        # Gamma call per event.
        event_slug = pre_discovered_event["event_slug"]
        market_prices = {label: b["price"] for label, b in pre_discovered_event["brackets"].items()
                          if b["price"] is not None}
        questions_by_label = {label: b["question"] for label, b in pre_discovered_event["brackets"].items()}
        tokens_by_label = {label: b["token_id"] for label, b in pre_discovered_event["brackets"].items()}
        fees_by_label = {label: (b["fee_rate"], b["fee_exponent"])
                         for label, b in pre_discovered_event["brackets"].items()}
        error_msg = None if market_prices else "No priced brackets in pre-discovered event."
    else:
        # Fallback path if called directly without discovery (e.g. manual testing).
        matches = [e for e in discover_open_munich_events()
                   if e["target_date"] is not None
                   and e["target_date"].strftime("%Y-%m-%d") == target_date_str]
        if not matches:
            market_prices, questions_by_label, event_slug, error_msg = None, None, None, \
                f"No open event found for target_date {target_date_str}"
        else:
            event = matches[0]
            event_slug = event["event_slug"]
            market_prices = {label: b["price"] for label, b in event["brackets"].items() if b["price"] is not None}
            questions_by_label = {label: b["question"] for label, b in event["brackets"].items()}
            tokens_by_label = {label: b["token_id"] for label, b in event["brackets"].items()}
            fees_by_label = {label: (b["fee_rate"], b["fee_exponent"]) for label, b in event["brackets"].items()}
            error_msg = None if market_prices else "No priced brackets found."

    if not market_prices:
        print(f"\n[!] ABORTING SCAN: Market is offline or unreadable ({error_msg}).")
        print("=" * 105)
        return

    # Tradable prices: fetch each bracket's CLOB book (YES token). The
    # "market probability" shown/logged is the book midpoint; the cost of
    # a trade is the ask (YES) or 1 - bid (NO) -- see decide_side_and_edge.
    books = {}
    for label, tok in tokens_by_label.items():
        bids, asks = fetch_orderbook_depth(tok)
        if bids or asks:
            books[label] = top_of_book(bids, asks)

    model_probs = {}
    for label in market_prices.keys():
        bounds_tuple = parse_label_to_bounds(label)
        if bounds_tuple[0] is not None or bounds_tuple[1] is not None:
            model_probs[label] = calculate_bracket_probability(
                forecast_mean_for_probs, sigma, bounds_tuple, observed_floor)

    print("\n" + "-" * 105)
    print(f" {'Bracket':<18} | {'Model P':<8} | {'Mid':<7} | {'Bid/Ask':<11} | {'Net Edge':<9} | {'Signal':<15} | {'Kelly Rec ($)':<14}")
    print("-" * 105)

    # PASS 1: decide side (YES/NO/no-trade) and edge per bracket, no sizing yet.
    all_labels = list(model_probs.keys())
    all_probs = [model_probs[lbl] for lbl in all_labels]
    decisions = {}
    for label, gamma_price in market_prices.items():
        if label not in model_probs:
            continue
        mod_p = model_probs[label]
        tob = books.get(label, {})
        if tob.get("best_bid") is None and tob.get("best_ask") is None:
            print(f" {label:<18} | no order book -- skipped")
            continue
        fee_rate, fee_exp = fees_by_label.get(label, (DEFAULT_FEE_RATE, DEFAULT_FEE_EXPONENT))
        signal, net_edge, side, cost = decide_side_and_edge(
            mod_p, tob.get("best_bid"), tob.get("best_ask"), fee_rate, fee_exp)
        # Midpoint only exists if BOTH sides have orders. With one side empty
        # Gamma still shows a "price" (e.g. half of a lone 0.13 ask = 0.065),
        # which is not a real market probability -- log None instead.
        decisions[label] = {"model_prob": mod_p, "market_price": tob.get("mid"),
                             "best_bid": tob.get("best_bid"), "best_ask": tob.get("best_ask"),
                             "signal": signal, "net_edge": net_edge, "side": side, "cost": cost,
                             "fee_rate": fee_rate}

    # PASS 2: joint sizing across every tradeable bracket in this event,
    # sharing ONE bankroll constraint instead of each bracket getting
    # its own independent Kelly allocation.
    candidates = [
        Candidate(label=lbl, side=d["side"], model_prob=d["model_prob"], cost=d["cost"])
        for lbl, d in decisions.items() if d["side"] is not None
    ]
    kelly_results = solve_joint_kelly(candidates, all_labels, all_probs, KELLY_FRACTION, BANKROLL)
    kelly_by_label = {r.label: r for r in kelly_results}

    log_rows = []
    for label, d in decisions.items():
        kr = kelly_by_label.get(label)
        bet_amount = kr.stake_dollars if kr else 0.0
        kelly_frac = kr.fraction_after_multiplier if kr else 0.0
        pos_str = f"${bet_amount:>6.2f} ({kelly_frac*100:>4.1f}%)" if bet_amount > 0 else "$0.00 (0.0%)"
        ba = (f"{d['best_bid'] if d['best_bid'] is not None else '-'}/"
              f"{d['best_ask'] if d['best_ask'] is not None else '-'}")
        mid_str = f"{d['market_price']*100:>5.1f}%" if d["market_price"] is not None else f"{'-':>6}"
        print(f" {label:<18} | {d['model_prob']*100:>6.1f}% | {mid_str} | {ba:<11} | "
              f"{d['net_edge']*100:>+7.1f}% | {d['signal']:<15} | {pos_str}")

        log_rows.append({
            "label": label, "model_prob": d["model_prob"], "market_price": d["market_price"],
            "net_edge": d["net_edge"], "signal": d["signal"], "kelly_fraction": kelly_frac,
            "bet_amount": bet_amount, "best_bid": d["best_bid"], "best_ask": d["best_ask"],
            "entry_cost": d["cost"], "fee_rate": d["fee_rate"],
        })
    print("=" * 105)
    total_staked = sum(r["bet_amount"] for r in log_rows)
    print(f" Total staked across event: ${total_staked:.2f} of ${BANKROLL:.2f} bankroll "
          f"({total_staked/BANKROLL*100:.1f}%) -- now correctly bounded by the joint solver.")

    # THIS is the restored piece -- every run now persists, instead of
    # printing to a terminal and discarding the only evidence that
    # could ever answer "is there a real edge here".
    for r in log_rows:
        r["observed_max_so_far"] = observed_floor
    log_predictions(target_date_str, event_slug, questions_by_label, log_rows,
                     calibrated_mean, sigma, lead_time_hours, tokens_by_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices-only", action="store_true",
                         help="Lightweight 15-min price + order book depth snapshot "
                              "across ALL currently open Munich markets.")
    args = parser.parse_args()

    init_db()

    if args.prices_only:
        run_prices_only_all_open_events()
    else:
        settle_unsettled_markets()
        run_scanner_for_all_open_events()
