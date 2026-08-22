"""
scheduler.py
-------------
Background thread that replaces the two cron entries with an
in-process loop, controllable via the dashboard's Start/Stop button.
Reuses the existing functions in munich_bot.py directly rather than
reimplementing anything -- this is just a scheduling wrapper around
the same logic you already ran manually from the command line.

If you'd rather keep using real cron (more robust for something that
needs to survive your laptop sleeping/restarting), that's still fine
and unaffected by this -- this is an alternative for convenience while
you're actively watching the dashboard, not a replacement for cron in
production. Cron keeps running even if this Flask process isn't up;
this scheduler only runs while app.py is running.
"""

import threading
import time
import traceback
from datetime import datetime, timedelta

import munich_bot as bot


class BotScheduler:
    def __init__(self, price_interval_sec=15 * 60, full_run_interval_sec=4 * 3600):
        self.price_interval = price_interval_sec
        self.full_run_interval = full_run_interval_sec
        self._thread = None
        self._stop_event = threading.Event()
        self.last_price_run = None
        self.last_full_run = None
        self.last_error = None
        self.run_count_prices = 0
        self.run_count_full = 0

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.is_running():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        bot.init_db()
        # Run both immediately on start, so the dashboard shows data
        # right away instead of waiting a full interval for the first row.
        next_price_run = datetime.now()
        next_full_run = datetime.now()

        while not self._stop_event.is_set():
            now = datetime.now()
            if now >= next_price_run:
                self._safe_run(self._do_price_run, "prices-only")
                next_price_run = datetime.now() + timedelta(seconds=self.price_interval)
            if now >= next_full_run:
                self._safe_run(self._do_full_run, "full run")
                next_full_run = datetime.now() + timedelta(seconds=self.full_run_interval)

            # Check at a cadence fine enough to be responsive to Stop
            # and to short intervals, but no finer than 15s to avoid
            # busy-looping when both intervals are long (the normal case).
            check_interval = min(15, max(1, self.price_interval // 2))
            self._stop_event.wait(timeout=check_interval)

    def _safe_run(self, fn, label):
        try:
            fn()
            self.last_error = None
        except Exception:
            self.last_error = f"[{label}] {traceback.format_exc(limit=3)}"
            print(f"[scheduler] ERROR during {label}: {self.last_error}")

    def _do_price_run(self):
        target_date = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        market_prices, questions_by_label, event_slug, error_msg = bot.fetch_polymarket_orderbook_snapshot(target_date)
        if market_prices is None:
            print(f"[scheduler] prices-only: no market found: {error_msg}")
        else:
            bot.log_price_snapshot(event_slug, market_prices, questions_by_label)
        self.last_price_run = datetime.now()
        self.run_count_prices += 1

    def _do_full_run(self):
        bot.settle_unsettled_markets()
        target_date = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        bot.run_phase5_scanner(target_date_str=target_date, lead_time_hours=48)
        self.last_full_run = datetime.now()
        self.run_count_full += 1


scheduler = BotScheduler()
