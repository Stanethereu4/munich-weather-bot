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

"""
db.py
-----
DUAL-MODE: local SQLite (original behavior, zero setup) OR a remote
Postgres database, switched automatically by whether DATABASE_URL is
set in the environment. Every other file (munich_bot.py,
Lead_Time_Sensitivity.py, app.py, scheduler.py) is UNCHANGED -- they
all just call get_connection() and use '?' placeholders like before;
in Postgres mode a thin compatibility layer translates '?' to '%s'
transparently, so nothing else needed to be rewritten.

Why this matters for "free hosted database": SQLite is a FILE on your
disk. It has no concept of a remote server -- there's no way to make
a local .db file reachable from GitHub Actions' cloud runners (which
get a fresh, empty filesystem on every single scheduled run) or from
any other machine. A remote Postgres database is what actually solves
"log this without my computer running 24/7", because it's a server
your script talks to over the network, independent of which machine
runs the script.

Setting DATABASE_URL:
  Get a free Postgres instance from Supabase (supabase.com) or Neon
  (neon.tech) -- both have permanently-free tiers, no credit card
  needed for the free tier. Copy the connection string they give you
  (looks like postgresql://user:pass@host:5432/dbname) and either:
    - export DATABASE_URL="postgresql://..." before running locally, or
    - add it as a GitHub Actions repo secret (see .github/workflows/)
  If DATABASE_URL is unset, this falls back to local weather_bot.db
  exactly as before -- nothing breaks if you don't set it.

Table-by-table intended write cadence (unchanged from before):
  forecast_snapshots   -- every 3-6h
  price_snapshots       -- every 15 min
  orderbook_depth        -- hourly to start
  predictions            -- once per full model run
  settlements             -- once per resolved market/day
  leadtime_calibration    -- once per backtest re-run (daily)
"""

import os
import sqlite3

DB_FILE = "weather_bot.db"
DATABASE_URL = os.environ.get("DATABASE_URL")

# (horizon_days, sigma_c) measured from Open-Meteo Previous Runs API.
FALLBACK_SIGMAS = [(1, 0.77), (2, 0.99), (3, 1.14)]


# ---------------------------------------------------------------------
# Postgres compatibility layer: lets every other file keep using '?'
# placeholders and sqlite3-style cursor.description / fetchall(),
# without needing psycopg2-specific code anywhere else.
# ---------------------------------------------------------------------
class _PgCompatCursor:
    def __init__(self, real_cursor):
        self._cur = real_cursor

    def execute(self, query, params=None):
        pg_query = query.replace("?", "%s")
        if params is None:
            self._cur.execute(pg_query)
        else:
            self._cur.execute(pg_query, params)

    def fetchall(self):
        return self._cur.fetchall()

    def fetchone(self):
        return self._cur.fetchone()

    @property
    def description(self):
        return self._cur.description


class _PgCompatConnection:
    def __init__(self, real_conn):
        self._conn = real_conn

    def cursor(self):
        return _PgCompatCursor(self._conn.cursor())

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()


def get_connection():
    if DATABASE_URL:
        import psycopg2
        real_conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        return _PgCompatConnection(real_conn)
    else:
        return sqlite3.connect(DB_FILE)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    is_postgres = DATABASE_URL is not None

    # Postgres and SQLite differ on auto-increment PK and default-timestamp
    # syntax -- everything else in these DDL statements is identical.
    pk = "SERIAL PRIMARY KEY" if is_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
    ts_default = "TIMESTAMP DEFAULT NOW()" if is_postgres else "DATETIME DEFAULT CURRENT_TIMESTAMP"

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS forecast_snapshots (
            id {pk},
            fetch_ts {ts_default},
            target_date TEXT,
            model_key TEXT,
            daily_max_temp_c REAL
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS price_snapshots (
            id {pk},
            timestamp {ts_default},
            event_slug TEXT,
            market_question TEXT,
            polymarket_prob REAL
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS orderbook_depth (
            id {pk},
            timestamp {ts_default},
            event_slug TEXT,
            market_question TEXT,
            side TEXT,
            price_level REAL,
            size REAL
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS predictions (
            id {pk},
            timestamp {ts_default},
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

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS leadtime_calibration (
            id {pk},
            computed_at {ts_default},
            horizon_days INTEGER,
            mae_c REAL,
            sigma_c REAL,
            n_days_evaluated INTEGER,
            ecmwf_weight REAL
        )
    """)

    conn.commit()
    conn.close()
    print(f"[db] Initialized in {'POSTGRES (remote)' if is_postgres else 'SQLITE (local file)'} mode.")


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
        # Fallback = sigmas MEASURED via the Previous Runs API (Sept 2026),
        # not an invented number. The old 0.60C fallback was ~40% too
        # narrow at 48h and made the model systematically overconfident.
        print("[db] WARNING: no leadtime_calibration rows found -- using "
              "measured fallback sigmas (24h 0.77, 48h 0.99, 72h 1.14 C).")
        rows = FALLBACK_SIGMAS

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
