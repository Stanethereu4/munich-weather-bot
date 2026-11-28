"""
livepaper/engine.py
-------------------
The live PAPER trader. Every minute, for the 7 same-airport cities:
  1. Kalshi: every open bracket's best bid/ask (today and tomorrow).
  2. Polymarket: the order book of every bracket of the same days.
  3. For every Kalshi threshold X: Kalshi "at least X" (sum of mids from X up) and Polymarket's
     matching number (average of its thresholds X-1 and X+1), exactly as in the backtest.
  4. If Kalshi moved >= the variant's threshold in the last 15 minutes and Polymarket moved less
     than half of that: record a paper bet on the Polymarket bracket next to X, with the whole
     ask ladder of that moment.
Every 5 minutes: check fills of the patient variant's resting bids, settle finished days from
Polymarket's official outcome, write a heartbeat; every 10 minutes publish the dashboard.

Nothing here can place an order: it only reads public data and writes to our own database.
Run:  python3 -m livepaper.engine <minutes>      (e.g. 340 in the cloud loop, 5 for a test)
"""
import json
import sys
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from livepaper import feeds, store
from livepaper.config import (CITIES, COHERENT, DEDUPE_S, LADDER_LEVELS, LOOKBACK_S, MAKER_WAIT_S, UNDECIDED,
                              VARIANTS)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Engine:
    def __init__(self):
        self.hist = defaultdict(lambda: deque(maxlen=40))     # (city, date, X) -> (t, k, p, ok)
        self.last_bet = {}                                    # (variant, city, date, X) -> t
        self.first_done = set()                               # (variant, city, date)
        self.pm_cache = {}                                    # (city, date) -> (fetched_t, event)
        self.floor_cache = {}                                 # (city, date) -> (fetched_t, °F)
        self.thresholds_seen = 0
        self.cities_ok = 0
        # do not repeat "first per city-day" bets after a restart
        for r in store.query("SELECT DISTINCT variant, city, target_date FROM lp_bets WHERE variant = 'first'"):
            self.first_done.add((r["variant"], r["city"], r["target_date"]))

    # ---------------------------------------------------------------- data --
    def pm_event(self, city, date, t):
        c = self.pm_cache.get((city, date))
        if c is None or t - c[0] > 1800:
            ev = feeds.pm_event(city, date)
            self.pm_cache[(city, date)] = (t, ev)
            return ev
        return c[1]

    def floor(self, city, date, t):
        c = self.floor_cache.get((city, date))
        if c is None or t - c[0] > 300:
            _, icao, tz = CITIES[city]
            self.floor_cache[(city, date)] = (t, feeds.floor_f(icao, tz, date))
        return self.floor_cache[(city, date)][1]

    # ---------------------------------------------------------------- tick --
    def tick(self):
        t = int(time.time())
        days = []
        for city, (series, _, _) in CITIES.items():
            kq = feeds.kalshi_quotes(series)
            for date, kb in kq.items():
                ev = self.pm_event(city, date, t)
                if ev and not ev["closed"] and ev["brackets"]:
                    days.append((city, date, kb, ev["brackets"]))
        books = feeds.pm_books([b["yes"] for _, _, _, pb in days for b in pb])
        self.cities_ok = len({c for c, *_ in days})
        n_thr = 0
        for city, date, kb, pb in days:
            n_thr += self.evaluate(t, city, date, kb, pb, books)
        self.thresholds_seen = n_thr

    def evaluate(self, t, city, date, kb, pb, books):
        k_mid = {b["ticker"]: ((b["bid"] if b["bid"] == b["bid"] else 0) + (b["ask"] if b["ask"] == b["ask"] else 1)) / 2
                 for b in kb}
        p_mid = {}
        for b in pb:
            bk = books.get(b["yes"])
            if bk is None:
                return 0
            p_mid[b["yes"]] = ((bk["bid"] or 0.0) + (bk["ask"] if bk["ask"] is not None else 1.0)) / 2
        ok = abs(sum(k_mid.values()) - 1) <= COHERENT and abs(sum(p_mid.values()) - 1) <= COHERENT
        n = 0
        for X in sorted({b["lo"] for b in kb if b["lo"] > -500}):
            kx = sum(k_mid[b["ticker"]] for b in kb if b["lo"] >= X)
            up = [b for b in pb if b["lo"] >= X + 1]
            dn = [b for b in pb if b["lo"] >= X - 1]
            if not up or len(up) == len(pb) or not dn or len(dn) == len(pb):
                continue
            px = (sum(p_mid[b["yes"]] for b in up) + sum(p_mid[b["yes"]] for b in dn)) / 2
            h = self.hist[(city, date, X)]
            h.append((t, kx, px, ok))
            n += 1
            past = [e for e in h if abs((t - e[0]) - LOOKBACK_S) <= 90 and e[3]]
            if not ok or not past or not (UNDECIDED[0] <= kx <= UNDECIDED[1] and UNDECIDED[0] <= px <= UNDECIDED[1]):
                continue
            t0, k0, p0, _ = min(past, key=lambda e: abs((t - e[0]) - LOOKBACK_S))
            dK, dP = kx - k0, px - p0
            for name, v in VARIANTS.items():
                if abs(dK) >= v["thr"] and abs(dP) < v["thr"] / 2:
                    self.maybe_bet(name, v, t, city, date, X, dK, dP, kx, px, kb, pb, books)
        return n

    def maybe_bet(self, name, v, t, city, date, X, dK, dP, kx, px, kb, pb, books):
        key = (name, city, date, X)
        if key in self.last_bet and t - self.last_bet[key] < DEDUPE_S:
            return
        if v.get("first_per_city_day") and (name, city, date) in self.first_done:
            return
        direction = 1 if dK > 0 else -1
        if v.get("net") and self.close_opposite(name, t, city, date, direction, books):
            self.last_bet[key] = t                         # a sale counts as this threshold's action
            return
        target = [b for b in pb if (b["lo"] == X + 1 if direction > 0 else b["hi"] == X - 2)]
        if len(target) != 1:
            return
        b = target[0]
        bk = books.get(b["yes"])
        if not bk or bk["ask"] is None:
            return
        flo = None
        if v.get("skip_dead") or v.get("price_band"):
            flo = self.floor(city, date, t)
            if v.get("skip_dead") and flo is not None and b["hi"] < flo:
                return
            lo_p, hi_p = v["price_band"]
            if not lo_p <= bk["ask"] <= hi_p:
                return
        kt = None
        if v.get("min_kalshi_trades"):
            ticks = [k["ticker"] for k in kb if k["lo"] >= X]
            kt = feeds.kalshi_trade_count(ticks, t - LOOKBACK_S, t)
            if kt < v["min_kalshi_trades"]:
                return
        maker = bool(v.get("maker"))
        if maker and bk["bid"] is None:
            return
        self.last_bet[key] = t
        if v.get("first_per_city_day"):
            self.first_done.add((name, city, date))
        store.add_bet(dict(
            created_utc=now_iso(), ts=t, variant=name, city=city, target_date=date, x=int(X), direction=direction,
            k_now=kx, k_move=dK, p_now=px, p_move=dP, bracket_label=b["label"], yes_token=b["yes"], no_token=b["no"],
            condition_id=b["condition"], best_bid=bk["bid"], best_ask=bk["ask"],
            mid=((bk["bid"] or 0) + bk["ask"]) / 2, ladder=bk["asks"][:LADDER_LEVELS], floor_f=flo, kalshi_trades=kt,
            maker=int(maker), maker_price=bk["bid"] if maker else None, maker_status="pending" if maker else "n/a",
            maker_fill_ts=None, maker_traded_shares=None, status="open", won=None, settled_utc=None))
        print(f"[bet] {name:9} {city} {date} X={X} Kalshi {dK*100:+.0f}c Polymarket {dP*100:+.1f}c -> "
              f"{b['label']} ask {bk['ask']:.3f} ({bk['asks'][0][1]:.0f} shares)")

    def close_opposite(self, name, t, city, date, direction, books):
        """"net" variant: sell the open bets of this city-day that point the other way, at the best bid.
        Returns True if there were any (then no new bet is placed, even if a book had no bid)."""
        opp = store.query("SELECT id, yes_token, bracket_label FROM lp_bets WHERE variant = ? AND city = ? "
                          "AND target_date = ? AND direction = ? AND status = 'open'", (name, city, date, -direction))
        for r in opp:
            bk = books.get(r["yes_token"])
            if not bk or bk["bid"] is None:
                print(f"[sell]  {name:9} {city} {date} {r['bracket_label']}: no bid, keep holding")
                continue
            store.update("UPDATE lp_bets SET status = 'sold', exit_ts = ?, exit_price = ?, exit_bids = ? WHERE id = ?",
                         (t, bk["bid"], json.dumps(bk["bids"][:LADDER_LEVELS]), r["id"]))
            print(f"[sell]  {name:9} {city} {date} {r['bracket_label']} at bid {bk['bid']:.3f}")
        return bool(opp)

    # ------------------------------------------------------- housekeeping --
    def maker_fills(self):
        t = int(time.time())
        for r in store.query("SELECT * FROM lp_bets WHERE maker = 1 AND maker_status = 'pending'"):
            sells = feeds.pm_sells_into_bids(r["condition_id"], r["yes_token"], r["no_token"], r["ts"])
            hits = [s for s in sells if s[1] <= r["maker_price"] + 1e-9 and s[0] <= r["ts"] + MAKER_WAIT_S]
            if hits:
                store.update("UPDATE lp_bets SET maker_status = 'filled', maker_fill_ts = ?, maker_traded_shares = ? "
                             "WHERE id = ?", (min(h[0] for h in hits), sum(h[2] for h in hits), r["id"]))
            elif t - r["ts"] > MAKER_WAIT_S:
                store.update("UPDATE lp_bets SET maker_status = 'unfilled' WHERE id = ?", (r["id"],))

    def settle(self):
        # 'sold' bets ("net" variant) also get their result, for shares the bids could not absorb
        open_ = store.query("SELECT DISTINCT city, target_date FROM lp_bets "
                            "WHERE status = 'open' OR (status = 'sold' AND won IS NULL)")
        for r in open_:
            tz = ZoneInfo(CITIES[r["city"]][2])
            if datetime.now(tz).date().isoformat() <= r["target_date"]:
                continue                                   # the day has not ended yet
            ev = feeds.pm_event(r["city"], r["target_date"])
            if not ev or not ev["closed"]:
                continue
            for b in ev["brackets"]:
                try:
                    pr = json.loads(b["prices"]) if b["prices"] else None
                    won = int(pr[0] == "1") if pr and pr[0] in ("0", "1") else None
                except (ValueError, TypeError):
                    won = None
                if won is None:
                    continue
                store.update("UPDATE lp_bets SET status = 'settled', won = ?, settled_utc = ? "
                             "WHERE status = 'open' AND yes_token = ?", (won, now_iso(), b["yes"]))
                store.update("UPDATE lp_bets SET won = ?, settled_utc = ? "
                             "WHERE status = 'sold' AND won IS NULL AND yes_token = ?", (won, now_iso(), b["yes"]))
            print(f"[settle] {r['city']} {r['target_date']}")


def run(minutes):
    store.init()
    e = Engine()
    stop = time.time() + minutes * 60
    last_house = last_pub = 0
    while time.time() < stop:
        t0 = time.time()
        try:
            e.tick()
        except Exception:
            traceback.print_exc()
        if time.time() - last_house >= 300:
            for f in (e.maker_fills, e.settle):
                try:
                    f()
                except Exception:
                    traceback.print_exc()
            store.heartbeat(int(time.time()), now_iso(), e.cities_ok, e.thresholds_seen)
            print(f"[heartbeat] {now_iso()} cities {e.cities_ok}/7, thresholds {e.thresholds_seen}")
            last_house = time.time()
        if time.time() - last_pub >= 600:
            try:
                from livepaper import publish
                publish.publish()
            except Exception:
                traceback.print_exc()
            last_pub = time.time()
        time.sleep(max(5, 60 - (time.time() - t0)))


if __name__ == "__main__":
    run(float(sys.argv[1]) if len(sys.argv) > 1 else 5)
