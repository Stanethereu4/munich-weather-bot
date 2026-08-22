"""
db.py
-----
Shared schema for munich_bot.py and Lead_Time_Sensitivity.py.

Table-by-table intended write cadence (this is the answer to "what do
I log and how often" -- encode it here once, not as a scattered
convention across scripts):

  forecast_snapshots   -- every 3-6h (matches actual model run cadence;
                           polling hourly just re-fetches the same run)
  price_snapshots       -- every 15 min (--prices-only mode). This is
                           genuinely fast-moving data, worth the cadence.
  orderbook_depth        -- hourly to start. Heavier to store/query;
                           tighten later once you've seen a real
                           dislocation and want finer resolution.
  predictions            -- once per full model run (a few times/day),
                           one row per bracket.
  settlements             -- once per resolved market/day.
  leadtime_calibration    -- once per backtest re-run (daily is fine).
"""

import sqlite3

DB_FILE = "weather_bot.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecast_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetch_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            target_date TEXT,
            model_key TEXT,
            daily_max_temp_c REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_slug TEXT,
            market_question TEXT,
            polymarket_prob REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS orderbook_depth (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_slug TEXT,
            market_question TEXT,
            side TEXT,              -- 'bid' or 'ask'
            price_level REAL,
            size REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            target_date TEXT,
            event_slug TEXT,
            market_question TEXT,
            bracket_label TEXT,
            model_prob REAL,
            market_price REAL,
            net_edge REAL,
            signal TEXT,
            kelly_fraction REAL,
            bet_amount REAL,
            calibrated_mean_temp REAL,
            sigma_used REAL,
            lead_time_hours REAL,
            settled INTEGER DEFAULT 0,
            actual_max_temp REAL,
            resolved_outcome INTEGER,
            brier_score REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS leadtime_calibration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            horizon_days INTEGER,          -- 1-7, from Previous Runs API granularity
            mae_c REAL,
            sigma_c REAL,
            n_days_evaluated INTEGER,
            ecmwf_weight REAL              -- SLSQP-fit weight at this horizon, for inspection
        )
    """)

    conn.commit()
    conn.close()


def get_connection():
    return sqlite3.connect(DB_FILE)


def latest_leadtime_sigma(lead_time_hours: float) -> float:
    """
    Reads the most recently computed sigma for the horizon closest to
    lead_time_hours, interpolating between the two nearest measured
    day-level horizons rather than snapping to whichever fabricated
    bucket it happened to fall into.

    Falls back to a conservative default (widest measured sigma, or
    0.6C if no calibration has been run yet) if the table is empty --
    loud enough to notice (prints a warning) rather than silently
    reusing stale hardcoded numbers.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT horizon_days, sigma_c FROM leadtime_calibration
        WHERE computed_at = (SELECT MAX(computed_at) FROM leadtime_calibration)
        ORDER BY horizon_days
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("[db] WARNING: no leadtime_calibration rows found -- run "
              "Lead_Time_Sensitivity.py first. Falling back to sigma=0.60C.")
        return 0.60

    target_days = lead_time_hours / 24.0
    horizons = [r[0] for r in rows]
    sigmas = [r[1] for r in rows]

    if target_days <= horizons[0]:
        return sigmas[0]
    if target_days >= horizons[-1]:
        return sigmas[-1]

    for i in range(len(horizons) - 1):
        if horizons[i] <= target_days <= horizons[i + 1]:
            span = horizons[i + 1] - horizons[i]
            weight = (target_days - horizons[i]) / span if span > 0 else 0
            return sigmas[i] + weight * (sigmas[i + 1] - sigmas[i])

    return sigmas[-1]
