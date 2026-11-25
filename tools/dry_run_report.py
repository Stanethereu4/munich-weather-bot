"""Print a short summary of the throwaway SQLite DB after a dry run."""
import sqlite3

c = sqlite3.connect("weather_bot.db")
q = lambda sql: c.execute(sql).fetchall()
print("by kind -- price rows:", q("SELECT market_kind, COUNT(*), COUNT(DISTINCT event_slug) FROM price_snapshots GROUP BY market_kind"))
print("by kind -- predictions:", q("SELECT market_kind, COUNT(*) FROM predictions GROUP BY market_kind"))
print("by kind -- forecast rows:", q("SELECT market_kind, COUNT(*) FROM forecast_snapshots GROUP BY market_kind"))
print("calibration:", q("SELECT market_kind, horizon_days, ROUND(sigma_c, 3) FROM leadtime_calibration ORDER BY market_kind, horizon_days"))
print("price_snapshots:", q("SELECT COUNT(*), COUNT(DISTINCT snapshot_id), COUNT(token_id), COUNT(best_ask) FROM price_snapshots"),
      "(rows, snapshots, rows with token, rows with best_ask)")
print("units check (all must be 0-1):", q("SELECT MIN(gamma_price), MAX(gamma_price), MIN(mid), MAX(mid) FROM price_snapshots"))
for r in q("""SELECT bracket_label, gamma_price, best_bid, best_ask, mid, bid_depth_5c, ask_depth_5c, ts_utc
              FROM price_snapshots ORDER BY id LIMIT 12"""):
    print("  ", r)
print("orderbook_depth:", q("SELECT COUNT(*), COUNT(token_id), COUNT(DISTINCT snapshot_id) FROM orderbook_depth"))
print("predictions:", q("SELECT COUNT(*), COUNT(token_id), COUNT(entry_cost) FROM predictions"))
for r in q("""SELECT target_date, bracket_label, ROUND(model_prob,3), market_price, best_bid, best_ask,
                     ROUND(entry_cost,4), signal, ROUND(lead_time_hours,1), ts_utc
              FROM predictions ORDER BY id"""):
    print("  ", r)
print("forecast_snapshots:", q("SELECT COUNT(*), COUNT(ts_utc) FROM forecast_snapshots"))

print("blend_weights:", q("SELECT fit_date, market_kind, bias_c, n_days, window_start, window_end FROM blend_weights"))
print("forecast rows with run tags:", q("SELECT market_kind, model_key, init_time FROM forecast_snapshots WHERE init_time IS NOT NULL ORDER BY id LIMIT 10"))
