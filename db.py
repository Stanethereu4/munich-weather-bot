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

# (horizon_days, sigma_c) measured with Lead_Time_Sensitivity.py on
# 25 Nov 2026: Previous Runs API forecasts vs the EDDM station max
# (whole °C, 366 days). Only used if leadtime_calibration is empty.
FALLBACK_SIGMAS = [(0, 0.63), (1, 1.17), (2, 1.36), (3, 1.49)]


# ---------------------------------------------------------------------
# Postgres compatibility layer: lets every other file keep using '?'
# placeholders and sqlite3-style cursor.description / fetchall(),
# without needing psycopg2-specific code anywhere else.
# ---------------------------------------------------------------------
def _to_python(params):
    """psycopg2 can't adapt numpy scalars (np.float64 gets rendered as
    the SQL text 'np.float64(...)' -> 'schema "np" does not exist').
    Convert anything with .item() to the plain Python equivalent."""
    return tuple(p.item() if hasattr(p, "item") else p for p in params)


class _PgCompatCursor:
    def __init__(self, real_cursor):
        self._cur = real_cursor

    def execute(self, query, params=None):
        pg_query = query.replace("?", "%s")
        if params is None:
            self._cur.execute(pg_query)
        else:
            self._cur.execute(pg_query, _to_python(params))

    def executemany(self, query, seq_of_params):
        # One network round trip per page of rows instead of per row --
        # matters a lot when GitHub's runner talks to a remote database.
        from psycopg2.extras import execute_batch
        pg_query = query.replace("?", "%s")
        execute_batch(self._cur, pg_query,
                      [_to_python(p) for p in seq_of_params], page_size=500)

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
            polymarket_prob REAL      -- LEGACY: price x 100, no longer written (see gamma_price)
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

    _apply_migrations(cur, is_postgres)

    conn.commit()
    conn.close()
    print(f"[db] Initialized in {'POSTGRES (remote)' if is_postgres else 'SQLITE (local file)'} mode.")


# ---------------------------------------------------------------------
# Schema migrations
# ---------------------------------------------------------------------
# CREATE TABLE IF NOT EXISTS never changes a table that already exists,
# so columns added later are listed here and added on startup if
# missing. Safe to run every time (it only adds what isn't there yet).
NEW_COLUMNS = [
    # A7 -- market price stored as a probability 0-1 (the old
    # price_snapshots.polymarket_prob column is price x 100; kept for
    # old rows, no longer written).
    ("price_snapshots", "gamma_price", "REAL"),
    # A8 -- explicit UTC timestamp written by Python (ISO-8601 text, sorts
    # correctly) + one snapshot_id shared by all rows of the same snapshot.
    # The old per-row `timestamp` DEFAULT columns are kept but not relied on.
    # A9 -- rows keyed by token, not only by question text.
    ("price_snapshots", "ts_utc", "TEXT"),
    ("price_snapshots", "snapshot_id", "TEXT"),
    ("price_snapshots", "token_id", "TEXT"),
    ("price_snapshots", "bracket_label", "TEXT"),
    ("orderbook_depth", "ts_utc", "TEXT"),
    ("orderbook_depth", "snapshot_id", "TEXT"),
    ("orderbook_depth", "token_id", "TEXT"),
    ("orderbook_depth", "bracket_label", "TEXT"),
    ("predictions", "ts_utc", "TEXT"),
    ("predictions", "snapshot_id", "TEXT"),
    ("predictions", "token_id", "TEXT"),
    ("forecast_snapshots", "ts_utc", "TEXT"),
    # A5 -- tradable prices from the CLOB order book (YES token), 0-1.
    # depth_5c = shares available within 5 cents of the best price.
    ("price_snapshots", "best_bid", "REAL"),
    ("price_snapshots", "best_ask", "REAL"),
    ("price_snapshots", "mid", "REAL"),
    ("price_snapshots", "bid_depth_5c", "REAL"),
    ("price_snapshots", "ask_depth_5c", "REAL"),
    ("predictions", "best_bid", "REAL"),
    ("predictions", "best_ask", "REAL"),
    ("predictions", "entry_cost", "REAL"),   # price + fee actually paid for the chosen side
    # A4 -- Polymarket's own resolution (exact truth for P&L), stored next
    # to the station-derived outcome (resolved_outcome, used for calibration).
    ("predictions", "pm_outcome", "INTEGER"),         # 1 if this bracket won on Polymarket
    ("predictions", "pm_winning_bracket", "TEXT"),
    ("predictions", "outcome_mismatch", "INTEGER"),   # 1 if station outcome != Polymarket outcome
    ("predictions", "fee_rate", "REAL"),              # taker fee rate used (market's feeSchedule)
    ("predictions", "observed_max_so_far", "REAL"),   # same-day only: EDDM routine-METAR max so far
]


def _existing_columns(cur, table, is_postgres):
    if is_postgres:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = ?", (table,))
        return {r[0] for r in cur.fetchall()}
    cur.execute(f"PRAGMA table_info({table})")
    return {r[1] for r in cur.fetchall()}


def _apply_migrations(cur, is_postgres):
    for table, column, col_type in NEW_COLUMNS:
        if column not in _existing_columns(cur, table, is_postgres):
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            print(f"[db] migration: added {table}.{column}")

    # A7 backfill: convert old x100 prices into the 0-1 column, once.
    cur.execute("""
        UPDATE price_snapshots SET gamma_price = polymarket_prob / 100.0
        WHERE gamma_price IS NULL AND polymarket_prob IS NOT NULL
    """)


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
              "measured fallback sigmas (0h 0.63, 24h 1.17, 48h 1.36, 72h 1.49 C).")
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
