"""
livepaper/store.py
------------------
Paper bets in the same database as the logger (Supabase Postgres in the cloud,
SQLite locally). Two new tables, created if missing; nothing else is touched.

lp_bets       one row per (signal, variant): the moment, the signal, the bracket, and the
              full Polymarket ask ladder at that moment (so any bet size can be replayed later)
              "net" bets that were sold early also get exit_ts, exit_price (best bid) and exit_bids
              (the bid ladder at that moment, best first) and status 'sold'
lp_heartbeat  one row every few minutes: is the loop alive, how many cities/thresholds it saw
"""
import json

import db

COLS = ["created_utc", "ts", "variant", "city", "target_date", "x", "direction", "k_now", "k_move", "p_now",
        "p_move", "bracket_label", "yes_token", "no_token", "condition_id", "best_bid", "best_ask", "mid",
        "ladder", "floor_f", "kalshi_trades", "maker", "maker_price", "maker_status", "maker_fill_ts",
        "maker_traded_shares", "status", "won", "settled_utc"]


def init():
    conn = db.get_connection()
    cur = conn.cursor()
    pk = "SERIAL PRIMARY KEY" if db.DATABASE_URL else "INTEGER PRIMARY KEY AUTOINCREMENT"
    cur.execute(f"""CREATE TABLE IF NOT EXISTS lp_bets (
        id {pk}, created_utc TEXT, ts BIGINT, variant TEXT, city TEXT, target_date TEXT, x INTEGER,
        direction INTEGER, k_now REAL, k_move REAL, p_now REAL, p_move REAL, bracket_label TEXT,
        yes_token TEXT, no_token TEXT, condition_id TEXT, best_bid REAL, best_ask REAL, mid REAL,
        ladder TEXT, floor_f INTEGER, kalshi_trades INTEGER, maker INTEGER, maker_price REAL,
        maker_status TEXT, maker_fill_ts BIGINT, maker_traded_shares REAL, status TEXT, won INTEGER,
        settled_utc TEXT)""")
    conn.commit()
    # columns added on 28 Nov 2026 for the "net" variant (selling early); harmless if they exist
    for col, typ in (("exit_ts", "BIGINT"), ("exit_price", "REAL"), ("exit_bids", "TEXT")):
        if db.DATABASE_URL:
            cur.execute(f"ALTER TABLE lp_bets ADD COLUMN IF NOT EXISTS {col} {typ}")
        else:
            try:
                cur.execute(f"ALTER TABLE lp_bets ADD COLUMN {col} {typ}")
            except Exception:
                pass                                   # SQLite: column already there
    cur.execute(f"""CREATE TABLE IF NOT EXISTS lp_heartbeat (
        id {pk}, ts BIGINT, created_utc TEXT, cities_ok INTEGER, thresholds INTEGER, note TEXT)""")
    conn.commit(); conn.close()


def add_bet(row):
    row = dict(row)
    row["ladder"] = json.dumps(row.get("ladder") or [])
    conn = db.get_connection(); cur = conn.cursor()
    cur.execute(f"INSERT INTO lp_bets ({', '.join(COLS)}) VALUES ({', '.join('?' * len(COLS))})",
                [row.get(c) for c in COLS])
    conn.commit(); conn.close()


def query(sql, params=()):
    conn = db.get_connection(); cur = conn.cursor()
    cur.execute(sql, params)
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


def update(sql, params=()):
    conn = db.get_connection(); cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit(); conn.close()


def heartbeat(ts, created, cities_ok, thresholds, note=""):
    update("INSERT INTO lp_heartbeat (ts, created_utc, cities_ok, thresholds, note) VALUES (?, ?, ?, ?, ?)",
           (ts, created, cities_ok, thresholds, note))
