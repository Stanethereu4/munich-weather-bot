"""
Lead_Time_Sensitivity.py
--------------------------
REPLACES the fabricated `horiz_scale = 1.0 + (horizon_h - 6) * 0.0045`
formula with an actually-measured lead-time degradation curve, using
Open-Meteo's Previous Runs API.

What changed and why: the old version pulled ONE forecast series
(historical-forecast-api's seamless "freshest data available" series)
and multiplied its error by a hand-picked linear factor. That factor
wasn't fit to anything -- it was an assumption labeled as a "physics
degradation factor," and every downstream probability in munich_bot.py
depended on it.

The Previous Runs API (previous-runs-api.open-meteo.com) instead
returns REAL archived forecasts at a fixed lead time: a
`temperature_2m_previous_day2` series contains only values that were
actually forecast 48 hours before their valid time, for real, as
archived at the time. That's what lets you measure a genuine
degradation curve instead of assuming one.

Real limitation to be upfront about: this API's granularity is in
whole days (previous_day1 = 24h, previous_day2 = 48h, ..., up to
previous_day7), not the arbitrary-hour resolution (62h, 36h, 12h...)
the old script pretended to have. That's a real loss of precision,
but a measured 24h-granularity curve is worth more than a fabricated
hourly one -- you're trading false precision for honest precision.
Archive coverage starts January 2024 for most models.

Writes results to leadtime_calibration in db.py's schema so
munich_bot.py can read a real, current sigma instead of a hardcoded
dict of numbers someone has to remember to update by hand.

NOT executed against the live endpoint in this sandbox -- run it
directly and inspect the printed table before trusting the DB writes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize
import openmeteo_requests
import requests_cache
from retry_requests import retry

from db import init_db, get_connection
from station_obs import fetch_daily_max as fetch_station_daily_max
from daymax import hourly_times_local, daily_max_by_date

# --- CONFIGURATION ---
LATITUDE = 48.3537
LONGITUDE = 11.7860
DAYS_TO_EVALUATE = 365          # capped by actual archive availability, see note below
HORIZON_DAYS = [0, 1, 2, 3]     # 0 = same-day run (the plain temperature_2m series
                                 # in the Previous Runs API), needed for same-day markets.
                                 # Polymarket weather markets open ~62h (2.58 days) before
                                 # resolution -- there's no tradeable market at a 4-day
                                 # horizon at all, so measuring sigma there was wasted
                                 # effort (and was also the thinnest-coverage horizon).
                                 # Day 2 (48h) and day 3 (72h) already bracket the real
                                 # 62h market-open point, and latest_leadtime_sigma()
                                 # interpolates between them.

MODELS = {
    "ecmwf_ifs025": "ECMWF",
    "icon_seamless": "DWD ICON",
    "gfs_seamless": "NOAA GFS",
    "gem_global": "CMC GEM",
    "meteofrance_seamless": "Météo-France",
}

cache_session = requests_cache.CachedSession('.cache_leadtime_real', expire_after=86400)
retry_session = retry(cache_session, retries=5, backoff_factor=0.3)
openmeteo = openmeteo_requests.Client(session=retry_session)


def fetch_previous_run_forecasts(start_str, end_str, horizon_day):
    """
    Fetches REAL archived forecasts at a fixed lead time via the
    Previous Runs API -- e.g. horizon_day=2 returns, for every valid
    date in range, the value that was actually forecast 48h ahead of
    that date, as archived at the time (not reconstructed after the
    fact from the freshest available data).

    Endpoint and per-model `_previous_dayN` suffix convention per
    Open-Meteo's Previous Runs API docs
    (https://open-meteo.com/en/docs/previous-runs-api). Confirm this
    against the live docs on first run -- suffix conventions can shift.
    """
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    hourly_vars = ["temperature_2m" if horizon_day == 0
                   else f"temperature_2m_previous_day{horizon_day}"]
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": start_str, "end_date": end_str,
        "hourly": hourly_vars,
        "models": list(MODELS.keys()),
        "timezone": "Europe/Berlin",
    }
    return openmeteo.weather_api(url, params=params)


def compute_model_daily_max_from_previous_run(responses, var_index=0):
    """Reduces the previous-run hourly series to a full-day max per model."""
    model_daily_max = {m: {} for m in MODELS.keys()}
    for model_key, response in zip(MODELS.keys(), responses):
        hourly = response.Hourly()
        vals = hourly.Variables(var_index).ValuesAsNumpy()
        # Full local day (00:00-23:59), same as the market and station truth.
        per_day = daily_max_by_date(hourly_times_local(hourly, len(vals)), vals)
        model_daily_max[model_key] = {d: v for d, v in per_day.items() if not np.isnan(v)}
    return model_daily_max


def fit_slsqp_ensemble(X, y_true):
    """Same SLSQP weighting approach as backtest.py, applied here
    per-horizon so the reported sigma reflects the ensemble you'd
    actually be using at that lead time, not a single model."""
    num_models = X.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w[:num_models]) - 1.0})
    bounds = [(0.0, 1.0)] * num_models + [(-2.0, 2.0)]
    initial_guess = [1.0 / num_models] * num_models + [0.0]

    def loss_func(params):
        w, b = params[:num_models], params[num_models]
        return np.mean((y_true - (np.dot(X, w) + b)) ** 2)

    res = minimize(loss_func, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x[:num_models], res.x[num_models]


def run_real_leadtime_measurement():
    end_date = datetime.now() - timedelta(days=3)  # leave margin for API archival lag
    start_date = end_date - timedelta(days=DAYS_TO_EVALUATE)
    start_str, end_str = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    print(f"--- MEASURING REAL LEAD-TIME DEGRADATION ({start_str} to {end_str}) ---")
    print("(Previous Runs API archive starts Jan 2024 for most models -- if your "
          "DAYS_TO_EVALUATE window predates that, results will be shorter than requested.)\n")

    # Ground truth = EDDM station (METAR) daily max in whole °C -- what
    # Polymarket resolves on -- not the Open-Meteo grid. Sigma measured
    # against the grid understates the real error versus resolution.
    actuals = fetch_station_daily_max(start_str, end_str, verbose=True)

    print(f"{'Horizon':<12} | {'Ensemble MAE':<15} | {'Measured Sigma':<16} | {'ECMWF Wgt':<10} | {'N days':<8}")
    print("-" * 75)

    results = []
    for horizon_day in HORIZON_DAYS:
        responses = fetch_previous_run_forecasts(start_str, end_str, horizon_day)
        model_daily_max = compute_model_daily_max_from_previous_run(responses)

        # Diagnostic: show per-model coverage BEFORE intersecting, so a
        # single short-archive model doesn't silently zero out the
        # whole horizon without explanation (this is what happened to
        # day 4 in the first real run -- one model's Previous Runs
        # archive likely doesn't extend that far).
        coverage = {m: len(model_daily_max[m]) for m in MODELS.keys()}
        print(f"  [coverage] day{horizon_day} per-model day counts: {coverage}")

        MIN_DAYS_PER_MODEL = 30
        usable_models = [m for m, n in coverage.items() if n >= MIN_DAYS_PER_MODEL]
        dropped_models = [m for m in MODELS.keys() if m not in usable_models]
        if dropped_models:
            print(f"  [coverage] dropping {dropped_models} for day{horizon_day} "
                  f"(< {MIN_DAYS_PER_MODEL} days available) -- refitting on remaining "
                  f"{len(usable_models)} model(s) instead of failing the whole horizon.")

        if len(usable_models) < 2:
            print(f"{horizon_day} day(s)   | fewer than 2 usable models after coverage "
                  f"filtering -- skipping")
            continue

        valid_dates = [d for d in sorted(actuals.keys())
                       if all(d in model_daily_max[m] for m in usable_models)]
        if len(valid_dates) < 30:
            print(f"{horizon_day} day(s)   | insufficient overlapping data "
                  f"({len(valid_dates)} days) -- skipping")
            continue

        y_true = np.array([actuals[d] for d in valid_dates])
        X = np.column_stack([[model_daily_max[m][d] for d in valid_dates] for m in usable_models])

        opt_w, opt_b = fit_slsqp_ensemble(X, y_true)
        pred = np.dot(X, opt_w) + opt_b
        mae = float(np.mean(np.abs(pred - y_true)))
        sigma = mae * 1.2533  # MAE -> sigma conversion for a normal distribution, same as before

        # ecmwf_weight is only meaningful if ECMWF was among the usable
        # models for this horizon -- report None otherwise rather than
        # silently mislabeling a different model's weight as ECMWF's.
        ecmwf_idx = usable_models.index("ecmwf_ifs025") if "ecmwf_ifs025" in usable_models else None
        ecmwf_w = float(opt_w[ecmwf_idx]) if ecmwf_idx is not None else None

        results.append({
            "horizon_days": horizon_day, "mae_c": mae, "sigma_c": sigma,
            "n_days_evaluated": len(valid_dates), "ecmwf_weight": ecmwf_w,
        })
        ecmwf_w_str = f"{ecmwf_w*100:.1f}%" if ecmwf_w is not None else "N/A (dropped)"
        print(f"{horizon_day} day(s)   | {mae:<15.3f} | {sigma:<16.3f} | "
              f"{ecmwf_w_str:<10} | {len(valid_dates):<8}")

    if not results:
        print("\nNo horizons produced enough data -- nothing written to DB.")
        return

    init_db()
    conn = get_connection()
    cur = conn.cursor()
    for r in results:
        cur.execute("""
            INSERT INTO leadtime_calibration (horizon_days, mae_c, sigma_c, n_days_evaluated, ecmwf_weight)
            VALUES (?, ?, ?, ?, ?)
        """, (r["horizon_days"], r["mae_c"], r["sigma_c"], r["n_days_evaluated"], r["ecmwf_weight"]))
    conn.commit()
    conn.close()
    print(f"\nWrote {len(results)} REAL measured horizon rows to leadtime_calibration.")
    print("munich_bot.py's get_horizon_sigma() should now read from this table "
          "instead of a hardcoded dict.")


if __name__ == "__main__":
    run_real_leadtime_measurement()
