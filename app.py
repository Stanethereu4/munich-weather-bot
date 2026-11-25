"""
app.py
-------
Minimal local dashboard: Start/Stop button controlling scheduler.py's
background loop, plus three auto-refreshing tables showing what's
actually landing in the database. Read-only queries against db.py --
doesn't duplicate any logic from munich_bot.py.

Run with:
    python3 app.py
Then open http://127.0.0.1:5000 in a browser.
"""

from flask import Flask, jsonify, render_template, request

from db import get_connection, init_db
from scheduler import scheduler

app = Flask(__name__)
init_db()


def rows_as_dicts(cur):
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "running": scheduler.is_running(),
        "last_price_run": scheduler.last_price_run.isoformat() if scheduler.last_price_run else None,
        "last_full_run": scheduler.last_full_run.isoformat() if scheduler.last_full_run else None,
        "run_count_prices": scheduler.run_count_prices,
        "run_count_full": scheduler.run_count_full,
        "last_error": scheduler.last_error,
    })


@app.route("/api/start", methods=["POST"])
def api_start():
    started = scheduler.start()
    return jsonify({"started": started, "already_running": not started})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    scheduler.stop()
    return jsonify({"stopped": True})


@app.route("/api/predictions")
def api_predictions():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 50")
    data = rows_as_dicts(cur)
    conn.close()
    return jsonify(data)


@app.route("/api/prices")
def api_prices():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM price_snapshots ORDER BY id DESC LIMIT 50")
    data = rows_as_dicts(cur)
    conn.close()
    return jsonify(data)


@app.route("/api/calibration")
def api_calibration():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM leadtime_calibration ORDER BY id DESC LIMIT 20")
    data = rows_as_dicts(cur)
    conn.close()
    return jsonify(data)


@app.route("/api/orderbook")
def api_orderbook():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM orderbook_depth ORDER BY id DESC LIMIT 100")
    data = rows_as_dicts(cur)
    conn.close()
    return jsonify(data)


@app.route("/api/summary")
def api_summary():
    """Quick counts so the dashboard can show 'how much data do I actually have' at a glance."""
    conn = get_connection()
    cur = conn.cursor()
    out = {}
    for table in ["predictions", "price_snapshots", "forecast_snapshots", "leadtime_calibration", "orderbook_depth"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        out[table] = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM predictions WHERE settled = 1")
    out["predictions_settled"] = cur.fetchone()[0]
    cur.execute("SELECT AVG(brier_score) FROM predictions WHERE settled = 1")
    row = cur.fetchone()
    out["avg_brier_score"] = row[0] if row and row[0] is not None else None
    conn.close()
    return jsonify(out)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
