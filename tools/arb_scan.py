"""
tools/arb_scan.py
-----------------
READ-ONLY scan of our own logged Munich order books for riskless combinations
("free money"), after the taker fee (0.01 * p * (1 - p) per share).

Three checks, per snapshot (all books logged in the same 15-min pass):

  1. BUY-ALL   one YES of every bracket of an event pays exactly $1
               (exactly one bracket wins).  Free money if the sum of best asks
               + fees < 1.
               Example: 11 brackets whose asks add up to 0.97 -> +3c per set.
  2. NO-ALL    one NO of every bracket pays N-1 dollars (all but the winner).
               NO costs 1 - (YES best bid). Free money if sum of YES bids > 1 + fees.
  3. LOW<=HIGH For the same day, the low cannot be above the high. For any
               threshold X:  YES on every high bracket >= X  +  YES on every low
               bracket < X  pays at least $1 in every outcome.
               Free money if those asks + fees < 1.

Signals come from top of book (price_snapshots); the full logged book
(orderbook_depth) then tells how many sets could really have been bought
before the price became unprofitable, i.e. the dollars actually available.
Caveat: the brackets of one snapshot are fetched a few seconds apart, so a
signal can be a timing artefact; persistence across snapshots is reported.

Run (needs DATABASE_URL; only SELECT queries):  python3 tools/arb_scan.py
"""
import os
import re
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import get_connection                      # noqa: E402
from munich_bot import parse_label_to_bounds, taker_fee_per_share  # noqa: E402


def q(sql, params=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, params or ())
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=cols)


def target_date(slug):
    m = re.search(r"-on-([a-z]+)-(\d+)-(\d{4})$", slug)
    return pd.to_datetime(f"{m.group(1)} {m.group(2)} {m.group(3)}").strftime("%Y-%m-%d") if m else None


def fee(p):
    return taker_fee_per_share(p)


def fill(legs, payout):
    """legs: list of [(price, size), ...] best first (already in the price we PAY).
    Buy one unit of every leg at a time while it still costs less than `payout`.
    Returns (sets bought, profit $, cost per set of the first set)."""
    idx = [0] * len(legs)
    left = [l[0][1] if l else 0 for l in legs]
    sets = profit = 0.0
    first = None
    while all(i < len(l) for i, l in zip(idx, legs)):
        prices = [l[i][0] for i, l in zip(idx, legs)]
        cost = sum(p + fee(p) for p in prices)
        if first is None:
            first = cost
        if cost >= payout:
            break
        qty = min(left)
        sets += qty
        profit += qty * (payout - cost)
        for k in range(len(legs)):
            left[k] -= qty
            if left[k] <= 1e-9:
                idx[k] += 1
                if idx[k] < len(legs[k]):
                    left[k] = legs[k][idx[k]][1]
    return sets, profit, first


def main():
    ps = q("""SELECT snapshot_id, ts_utc, event_slug, market_kind, bracket_label, token_id, best_bid, best_ask
              FROM price_snapshots WHERE snapshot_id IS NOT NULL AND best_ask IS NOT NULL""")
    ps["date"] = ps["event_slug"].map(target_date)
    print(f"price snapshots: {len(ps):,} rows, {ps['snapshot_id'].nunique():,} snapshots, "
          f"{ps['ts_utc'].min()} .. {ps['ts_utc'].max()}")

    # brackets per event (an event's full bracket list = the most labels ever seen in one snapshot)
    n_br = ps.groupby(["snapshot_id", "event_slug"])["bracket_label"].nunique().groupby("event_slug").max()

    sig = []        # (check, snapshot, ts, kind/date, key, top-of-book profit per set)
    for (sid, slug), g in ps.groupby(["snapshot_id", "event_slug"]):
        if g["bracket_label"].nunique() < n_br[slug]:
            continue                                   # incomplete snapshot
        ts, kind = g["ts_utc"].iloc[0], g["market_kind"].iloc[0]
        a = g["best_ask"].to_numpy(float)
        sig.append(("buy-all", sid, ts, kind, slug, 1 - sum(x + fee(x) for x in a)))
        b = g["best_bid"].dropna().to_numpy(float)
        if len(b) == len(g):
            n = len(g)
            sig.append(("no-all", sid, ts, kind, slug, (n - 1) - sum((1 - x) + fee(1 - x) for x in b)))
    # check 3
    # each event has its own snapshot_id; one 15-min logging pass = same 10-minute bucket
    ps["pass"] = pd.to_datetime(ps["ts_utc"], utc=True, format="ISO8601").dt.floor("10min").astype(str)
    for (sid, d), g in ps.groupby(["pass", "date"]):
        g = g[g.snapshot_id.isin(g.groupby("event_slug")["snapshot_id"].first())]   # one snapshot per event
        hi, lo = g[g.market_kind == "high"], g[g.market_kind == "low"]
        if hi.empty or lo.empty or hi.bracket_label.nunique() < n_br[hi.event_slug.iloc[0]] \
                or lo.bracket_label.nunique() < n_br[lo.event_slug.iloc[0]]:
            continue
        hb = [(parse_label_to_bounds(l), a) for l, a in zip(hi.bracket_label, hi.best_ask)]
        lb = [(parse_label_to_bounds(l), a) for l, a in zip(lo.bracket_label, lo.best_ask)]
        cuts = sorted({b[0] for (b, _) in hb + lb if b[0] is not None})
        best = None
        for x in cuts:
            legs = [a for (b, a) in hb if b[0] is not None and b[0] >= x] + \
                   [a for (b, a) in lb if b[1] is not None and b[1] <= x]
            if not legs:
                continue
            pr = 1 - sum(p + fee(p) for p in legs)
            if best is None or pr > best[0]:
                best = (pr, x)
        if best:
            sig.append(("low<=high", sid, g["ts_utc"].iloc[0], "both", f"{d} X={best[1]}", best[0]))
    s = pd.DataFrame(sig, columns=["check", "snapshot", "ts", "kind", "key", "profit_per_set"])

    print("\nTOP OF BOOK: how close do prices come to free money? (profit per $1 set after fees; >0 = free money)")
    summ = s.groupby(["check", "kind"])["profit_per_set"].describe(percentiles=[.5, .9, .99])[
        ["count", "50%", "90%", "99%", "max"]]
    summ["share > 0"] = s.groupby(["check", "kind"])["profit_per_set"].apply(lambda x: (x > 0).mean())
    print(summ.round(4).to_string())

    hits = s[s["profit_per_set"] > 0].sort_values("ts")
    print(f"\nsnapshots with free money at top of book: {len(hits)}")
    if hits.empty:
        return
    # persistence: same check + key in consecutive snapshots
    first_ts = ps.groupby("snapshot_id")["ts_utc"].min().sort_values()
    order = {sid: i for i, sid in enumerate(first_ts.index)}
    runs = defaultdict(list)
    for r in hits.itertuples():
        runs[(r.check, r.key)].append(order[r.snapshot])
    lengths = []
    for v in runs.values():
        v = sorted(v); cur = 1
        for a, b in zip(v, v[1:]):
            if b == a + 1:
                cur += 1
            else:
                lengths.append(cur); cur = 1
        lengths.append(cur)
    print(f"episodes: {len(lengths)}; lasting 1 snapshot (<15 min): {sum(l == 1 for l in lengths)}, "
          f"2+ snapshots: {sum(l >= 2 for l in lengths)}, longest {max(lengths)} snapshots")

    # depth: how many dollars were really available (checks 1 and 2, from the full logged book)
    ids = tuple(hits.loc[hits.check != "low<=high", "snapshot"].unique().tolist())
    out = []
    if ids:
        parts = []
        for k in range(0, len(ids), 400):          # chunked IN (...) lists
            chunk = ids[k:k + 400]
            parts.append(q(f"""SELECT snapshot_id, event_slug, token_id, side, price_level, size
                               FROM orderbook_depth WHERE snapshot_id IN ({','.join('?' * len(chunk))})""", chunk))
        ob = pd.concat(parts)
        for r in hits[hits.check != "low<=high"].itertuples():
            b = ob[(ob.snapshot_id == r.snapshot) & (ob.event_slug == r.key)]
            toks = ps[(ps.snapshot_id == r.snapshot) & (ps.event_slug == r.key)]["token_id"]
            legs = []
            for t in toks:
                bt = b[b.token_id == t]
                if r.check == "buy-all":
                    lv = bt[bt.side == "ask"].sort_values("price_level")
                    legs.append(list(zip(lv.price_level.astype(float), lv["size"].astype(float))))
                else:   # buy NO = sell into YES bids; NO price = 1 - bid
                    lv = bt[bt.side == "bid"].sort_values("price_level", ascending=False)
                    legs.append(list(zip(1 - lv.price_level.astype(float), lv["size"].astype(float))))
            if any(len(l) == 0 for l in legs):
                out.append((r.check, r.ts, r.key, r.profit_per_set, None, None)); continue
            payout = 1.0 if r.check == "buy-all" else len(legs) - 1.0
            sets, profit, _ = fill(legs, payout)
            out.append((r.check, r.ts, r.key, r.profit_per_set, sets, profit))
    d = pd.DataFrame(out, columns=["check", "ts", "event", "top_profit_per_set", "sets_fillable", "dollar_profit"])
    pd.set_option("display.width", 220)
    print("\nFREE-MONEY SNAPSHOTS (checks 1-2) with what the full book allowed:")
    print(d.sort_values("dollar_profit", ascending=False).head(25).to_string(index=False))
    print(f"\ntotal $ available across all such snapshots (if each was caught): {d['dollar_profit'].sum():.2f}")
    x3 = hits[hits.check == "low<=high"]
    if len(x3):
        print("\nLOW<=HIGH hits (top of book only):")
        print(x3[["ts", "key", "profit_per_set"]].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
