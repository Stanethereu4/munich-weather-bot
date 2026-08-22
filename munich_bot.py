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
from kelly_multi import Candidate, solve_joint_kelly

# --- CONFIGURATION ---
LATITUDE = 48.3537   # Munich Airport (EDDM)
LONGITUDE = 11.7860
ROLLING_WINDOW_DAYS = 30
MIN_EDGE_THRESHOLD = 0.03   # 3% minimum net probability edge after fees

# POLYMARKET TAKER FEE PARAMETER
# Formula: Fee = shares * THETA * p * (1 - p)
# Category: Economics / Culture / Weather / Other -> THETA = 0.05 (Max 1.25% at p=0.50)
# Re-verify against https://help.polymarket.com/en/articles/13364478-trading-fees
# periodically -- this schedule has changed more than once this year.
POLYMARKET_THETA = 0.05

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

cache_session = requests_cache.CachedSession('.cache_phase5_kelly', expire_after=3600)
retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
openmeteo = openmeteo_requests.Client(session=retry_session)


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
    clean_label = label.replace("°C", "").replace("°", "").strip()
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", clean_label)
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


def calculate_bracket_probability(mean_temp, sigma, bounds):
    t_min, t_max = bounds
    if t_min is None:
        return norm.cdf((t_max - mean_temp) / sigma)
    elif t_max is None:
        return 1.0 - norm.cdf((t_min - mean_temp) / sigma)
    else:
        return norm.cdf((t_max - mean_temp) / sigma) - norm.cdf((t_min - mean_temp) / sigma)


def decide_side_and_edge(model_prob, market_price, theta=0.05):
    """
    Determines which side (YES/NO) has positive net edge, and the
    effective cost basis for that side (price + fee) -- but no longer
    does per-bracket Kelly sizing. Sizing is now handled jointly
    across the whole event by kelly_multi.solve_joint_kelly, which
    respects a single shared bankroll constraint across mutually
    exclusive/overlapping bets instead of treating each bracket as if
    it had its own bankroll.
    """
    if market_price <= 0.001 or market_price >= 0.999:
        return "NO TRADE", 0.0, None, None

    fee_per_share = theta * market_price * (1.0 - market_price)

    cost_yes = market_price + fee_per_share
    net_edge_yes = model_prob - cost_yes

    price_no = 1.0 - market_price
    cost_no = price_no + fee_per_share
    net_edge_no = (1.0 - model_prob) - cost_no

    if net_edge_yes >= MIN_EDGE_THRESHOLD and net_edge_yes >= net_edge_no:
        return "BUY YES", net_edge_yes, "YES", cost_yes
    elif net_edge_no >= MIN_EDGE_THRESHOLD:
        return "BUY NO", net_edge_no, "NO", cost_no

    best_edge = net_edge_yes if net_edge_yes > net_edge_no else net_edge_no
    return "NO TRADE", best_edge, None, None


def fetch_live_ensemble_forecast(target_date_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": target_date_str, "end_date": target_date_str,
        "hourly": "temperature_2m", "models": list(MODELS.keys()),
        "timezone": "Europe/Berlin",
    }
    responses = openmeteo.weather_api(url, params=params)
    live_maxes = []
    per_model_max = {}

    for model_key, response in zip(MODELS.keys(), responses):
        hourly = response.Hourly()
        vals = hourly.Variables(0).ValuesAsNumpy()
        times = hourly.Time()
        step = hourly.Interval()
        ts_utc = pd.to_datetime(np.arange(times, times + len(vals) * step, step), unit="s", utc=True)
        ts_local = ts_utc.tz_convert("Europe/Berlin")

        df = pd.DataFrame({"dt": ts_local, "temp": vals})
        daytime = df[(df["dt"].dt.hour >= 10) & (df["dt"].dt.hour <= 19)]
        m = daytime["temp"].max()
        live_maxes.append(m)
        per_model_max[model_key] = float(m)

    return np.array(live_maxes), per_model_max


def log_forecast_snapshot(target_date_str, per_model_max):
    """Logs raw per-model forecasts -- call this every 3-6h, not every
    run, to match actual model-run cadence (see db.py docstring)."""
    conn = get_connection()
    cur = conn.cursor()
    for model_key, val in per_model_max.items():
        cur.execute("""
            INSERT INTO forecast_snapshots (target_date, model_key, daily_max_temp_c)
            VALUES (?, ?, ?)
        """, (target_date_str, model_key, val))
    conn.commit()
    conn.close()


def fetch_calibration_data(target_date_str):
    now = datetime.now()
    safe_end_dt = now - timedelta(days=2)

    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    end_dt = min(target_dt - timedelta(days=1), safe_end_dt)
    start_dt = end_dt - timedelta(days=ROLLING_WINDOW_DAYS)

    start_str, end_str = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    url_gt = "https://archive-api.open-meteo.com/v1/archive"
    params_gt = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": start_str, "end_date": end_str,
        "hourly": "temperature_2m", "timezone": "Europe/Berlin",
    }
    resp_gt = openmeteo.weather_api(url_gt, params=params_gt)[0].Hourly()
    vals_gt = resp_gt.Variables(0).ValuesAsNumpy()
    ts_gt = pd.to_datetime(np.arange(resp_gt.Time(), resp_gt.Time() + len(vals_gt) * resp_gt.Interval(),
                                      resp_gt.Interval()), unit="s", utc=True).tz_convert("Europe/Berlin")
    df_gt = pd.DataFrame({"dt": ts_gt, "temp": vals_gt})
    df_gt["date"] = df_gt["dt"].dt.strftime("%Y-%m-%d")

    actuals = {}
    for d, grp in df_gt.groupby("date"):
        daytime = grp[(grp["dt"].dt.hour >= 10) & (grp["dt"].dt.hour <= 19)]
        if len(daytime) > 0:
            actuals[d] = daytime["temp"].max()

    url_m = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params_m = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": start_str, "end_date": end_str,
        "hourly": "temperature_2m", "models": list(MODELS.keys()),
        "timezone": "Europe/Berlin",
    }
    resp_m = openmeteo.weather_api(url_m, params=params_m)

    model_daily_max = {m: {} for m in MODELS.keys()}
    for model_key, response in zip(MODELS.keys(), resp_m):
        hourly = response.Hourly()
        vals = hourly.Variables(0).ValuesAsNumpy()
        ts = pd.to_datetime(np.arange(hourly.Time(), hourly.Time() + len(vals) * hourly.Interval(),
                                       hourly.Interval()), unit="s", utc=True).tz_convert("Europe/Berlin")
        df_mod = pd.DataFrame({"dt": ts, "temp": vals})
        df_mod["date"] = df_mod["dt"].dt.strftime("%Y-%m-%d")
        for d, grp in df_mod.groupby("date"):
            daytime = grp[(grp["dt"].dt.hour >= 10) & (grp["dt"].dt.hour <= 19)]
            if len(daytime) > 0:
                model_daily_max[model_key][d] = daytime["temp"].max()

    valid_dates = [d for d in sorted(actuals.keys()) if all(d in model_daily_max[m] for m in MODELS.keys())]
    y_tr = np.array([actuals[d] for d in valid_dates])
    X_tr = np.column_stack([[model_daily_max[m][d] for d in valid_dates] for m in MODELS.keys()])
    return X_tr, y_tr


def fetch_polymarket_orderbook_snapshot(target_date_str):
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    slug_date = f"{target_dt.strftime('%B').lower()}-{target_dt.day}-{target_dt.year}"
    slug = f"highest-temperature-in-munich-on-{slug_date}"

    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        res = response.json()

        if res and isinstance(res, list) and len(res) > 0:
            title = res[0].get("title", slug)
            markets = res[0].get("markets", [])
            parsed_brackets = {}
            questions_by_label = {}

            for m in markets:
                label = m.get("groupItemTitle") or m.get("question")
                prices_raw = m.get("outcomePrices")
                if prices_raw:
                    prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                    if len(prices) > 0:
                        parsed_brackets[label] = float(prices[0])
                        questions_by_label[label] = m.get("question", label)

            if parsed_brackets:
                print(f"[*] Market Source : LIVE - Polymarket Event: '{title}'")
                print(f"[*] Market URL    : https://polymarket.com/event/{slug}")
                return parsed_brackets, questions_by_label, slug, None

        return None, None, slug, f"API Response Empty for slug '{slug}'."

    except requests.exceptions.HTTPError as http_err:
        return None, None, slug, f"HTTP Error: {http_err} (Status Code: {response.status_code})"
    except Exception as e:
        return None, None, slug, f"Parsing/Connection Error: {e}"


def log_price_snapshot(event_slug, market_prices, questions_by_label):
    """--prices-only path: append-only 15-min price log, no model run."""
    conn = get_connection()
    cur = conn.cursor()
    for label, price in market_prices.items():
        cur.execute("""
            INSERT INTO price_snapshots (event_slug, market_question, polymarket_prob)
            VALUES (?, ?, ?)
        """, (event_slug, questions_by_label.get(label, label), price * 100.0))
    conn.commit()
    conn.close()
    print(f"[prices-only] logged {len(market_prices)} rows for {event_slug} "
          f"at {datetime.now().isoformat()}")


def log_predictions(target_date_str, event_slug, questions_by_label, rows,
                     calibrated_mean, sigma, lead_time_hours):
    conn = get_connection()
    cur = conn.cursor()
    for r in rows:
        cur.execute("""
            INSERT INTO predictions (
                target_date, event_slug, market_question, bracket_label,
                model_prob, market_price, net_edge, signal, kelly_fraction,
                bet_amount, calibrated_mean_temp, sigma_used, lead_time_hours
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            target_date_str, event_slug, questions_by_label.get(r["label"], r["label"]), r["label"],
            r["model_prob"], r["market_price"], r["net_edge"], r["signal"], r["kelly_fraction"],
            r["bet_amount"], calibrated_mean, sigma, lead_time_hours,
        ))
    conn.commit()
    conn.close()
    print(f"[DB] Logged {len(rows)} prediction rows for {target_date_str}")


def settle_unsettled_markets():
    """Ported over from the earlier version -- was missing from the
    Phase 5 scanner entirely, which meant nothing was ever settled or
    Brier-scored. This is the actual mechanism that answers 'is there
    an edge', so it can't be optional."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT target_date FROM predictions
        WHERE settled = 0 AND target_date < DATE('now', 'localtime')
    """)
    unsettled_dates = cur.fetchall()
    if not unsettled_dates:
        print("[DB] No past unsettled predictions found.")
        conn.close()
        return

    for (t_date,) in unsettled_dates:
        print(f"[Settlement] Fetching actual station weather for {t_date}...")
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "start_date": t_date, "end_date": t_date,
            "hourly": "temperature_2m", "timezone": "Europe/Berlin",
        }
        try:
            resp = openmeteo.weather_api(url, params=params)[0].Hourly()
            vals = resp.Variables(0).ValuesAsNumpy()
            ts = pd.to_datetime(np.arange(resp.Time(), resp.Time() + len(vals) * resp.Interval(),
                                           resp.Interval()), unit="s", utc=True).tz_convert("Europe/Berlin")
            df = pd.DataFrame({"dt": ts, "temp": vals})
            daytime = df[(df["dt"].dt.hour >= 10) & (df["dt"].dt.hour <= 19)]
            actual_temp = float(round(daytime["temp"].max(), 2))
            print(f"[Settlement] Observed Peak Temp on {t_date}: {actual_temp}°C")

            cur.execute("""
                SELECT id, bracket_label, model_prob FROM predictions
                WHERE target_date = ? AND settled = 0
            """, (t_date,))
            rows = cur.fetchall()

            for row_id, label, model_prob in rows:
                bounds = parse_label_to_bounds(label)
                t_min, t_max = bounds
                if t_min is None:
                    outcome = 1 if actual_temp <= t_max else 0
                elif t_max is None:
                    outcome = 1 if actual_temp >= t_min else 0
                else:
                    outcome = 1 if t_min <= actual_temp <= t_max else 0

                brier = (model_prob - outcome) ** 2
                cur.execute("""
                    UPDATE predictions
                    SET actual_max_temp = ?, resolved_outcome = ?, brier_score = ?, settled = 1
                    WHERE id = ?
                """, (actual_temp, outcome, round(brier, 4), row_id))

            conn.commit()
            print(f"[Settlement] Settled all rows for {t_date}")
        except Exception as e:
            print(f"[Settlement Error] Could not settle {t_date}: {e}")

    conn.close()


def run_phase5_scanner(target_date_str, lead_time_hours):
    print("\n" + "=" * 105)
    print(f" PHASE 5: POLYMARKET EDGE & KELLY POSITION SCANNER | TARGET DATE: {target_date_str}")
    print("=" * 105)
    print(f"[*] Trading Parameters: Capital = ${BANKROLL:,.2f} | Risk = Quarter-Kelly ({KELLY_FRACTION:.2f}x) "
          f"| Taker Fee Theta = {POLYMARKET_THETA:.2f}")

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

    live_X, per_model_max = fetch_live_ensemble_forecast(target_date_str)
    log_forecast_snapshot(target_date_str, per_model_max)  # every run here; cron this at 3-6h cadence, not every scan

    calibrated_mean = np.dot(live_X, opt_w) + opt_b
    sigma = get_horizon_sigma(lead_time_hours)  # NOW REAL, from leadtime_calibration
    print(f"[*] Live Forecast Peak: {calibrated_mean:.2f} °C (Horizon: {lead_time_hours}h, "
          f"Measured Sigma: {sigma:.2f} °C)")

    print("\n[*] Fetching Polymarket order book...")
    market_prices, questions_by_label, event_slug, error_msg = fetch_polymarket_orderbook_snapshot(target_date_str)

    if market_prices is None:
        print(f"\n[!] ABORTING SCAN: Market is offline or unreadable ({error_msg}).")
        print("=" * 105)
        return

    model_probs = {}
    for label in market_prices.keys():
        bounds_tuple = parse_label_to_bounds(label)
        if bounds_tuple[0] is not None or bounds_tuple[1] is not None:
            model_probs[label] = calculate_bracket_probability(calibrated_mean, sigma, bounds_tuple)

    print("\n" + "-" * 105)
    print(f" {'Bracket':<18} | {'Model P':<8} | {'Mkt P':<7} | {'Net Edge':<9} | {'Signal':<15} | {'Kelly Rec ($)':<14}")
    print("-" * 105)

    # PASS 1: decide side (YES/NO/no-trade) and edge per bracket, no sizing yet.
    all_labels = list(model_probs.keys())
    all_probs = [model_probs[lbl] for lbl in all_labels]
    decisions = {}
    for label, mkt_price in market_prices.items():
        if label not in model_probs:
            continue
        mod_p = model_probs[label]
        signal, net_edge, side, cost = decide_side_and_edge(mod_p, mkt_price, POLYMARKET_THETA)
        decisions[label] = {"model_prob": mod_p, "market_price": mkt_price,
                             "signal": signal, "net_edge": net_edge, "side": side, "cost": cost}

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
        print(f" {label:<18} | {d['model_prob']*100:>6.1f}% | {d['market_price']*100:>5.1f}% | "
              f"{d['net_edge']*100:>+7.1f}% | {d['signal']:<15} | {pos_str}")

        log_rows.append({
            "label": label, "model_prob": d["model_prob"], "market_price": d["market_price"],
            "net_edge": d["net_edge"], "signal": d["signal"], "kelly_fraction": kelly_frac,
            "bet_amount": bet_amount,
        })
    print("=" * 105)
    total_staked = sum(r["bet_amount"] for r in log_rows)
    print(f" Total staked across event: ${total_staked:.2f} of ${BANKROLL:.2f} bankroll "
          f"({total_staked/BANKROLL*100:.1f}%) -- now correctly bounded by the joint solver.")

    # THIS is the restored piece -- every run now persists, instead of
    # printing to a terminal and discarding the only evidence that
    # could ever answer "is there a real edge here".
    log_predictions(target_date_str, event_slug, questions_by_label, log_rows,
                     calibrated_mean, sigma, lead_time_hours)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices-only", action="store_true",
                         help="Lightweight 15-min price snapshot, skips the full model run.")
    args = parser.parse_args()

    init_db()

    if args.prices_only:
        target_date = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        market_prices, questions_by_label, event_slug, error_msg = fetch_polymarket_orderbook_snapshot(target_date)
        if market_prices is None:
            print(f"[prices-only] no market found: {error_msg}")
        else:
            log_price_snapshot(event_slug, market_prices, questions_by_label)
    else:
        settle_unsettled_markets()
        target_date = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        run_phase5_scanner(target_date_str=target_date, lead_time_hours=48)
