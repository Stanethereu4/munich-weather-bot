"""
backtest/kalshi_test.py
-----------------------
Paper-trades the "safe" Kalshi / Polymarket trade on the last 30 days of US
daily highs (data from backtest/kalshi.py). Nothing is traded for real.

Why it is safe (same airport on both platforms only):
  * Kalshi settles on the official daily climate report (every reading of the day),
    Polymarket on the hourly reports only -> Kalshi's high >= Polymarket's high.
  * Brackets are offset by 1°F (Polymarket 80-81, Kalshi 79-80 / 81-82), so we
    compare "high is at least X" ladders instead of single brackets.
  * If Polymarket's high is >= Y, Kalshi's high is >= Y >= X. So for X <= Y:
        P_PM(high >= Y)  must be  <=  P_K(high >= X)
    THE TRADE when it is broken: buy Kalshi YES on every bracket >= X, and buy Polymarket
    YES on every bracket below Y. Every outcome pays at least $1:
        PM high >= Y -> Kalshi leg pays $1 ;  PM high < Y -> Polymarket leg pays $1.
    Example: Kalshi ">= 81" asks sum to 0.40 (+ fees 0.017), Polymarket brackets below 82
             sum to 0.55 (+1c each + fees) -> pair costs ~0.99 -> +1c, guaranteed.

Costs: Kalshi = real ask + 0.07 * p * (1 - p) per contract (taker fee);
       Polymarket = mid + 1c (we have no historical asks) + 0.01 * p * (1 - p).

Outputs:
  1) settlement check: was Kalshi's winning bracket ever entirely BELOW Polymarket's?
     (that would break the "safe" logic)
  2) every minute the trade was available, by edge size, and what the first signal per
     opportunity really paid at settlement.

Run:  python3 -m backtest.kalshi_test
"""
import os
import re

import numpy as np
import pandas as pd

# data folder; KALSHI_DIR=backtest_data/kalshi_hist switches to the long history (May 2026 on)
D = os.environ.get("KALSHI_DIR", os.path.join("backtest_data", "kalshi"))
PM_HALF_SPREAD, PM_FEE, K_FEE = 0.01, 0.01, 0.07
FFILL_MIN = 10            # a price is used for at most 10 minutes after it was seen


def pm_bounds(label):
    s = label.replace("°F", "").replace("°", "").strip()
    m = re.match(r"^(-?\d+)\s*-\s*(-?\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    n = int(re.match(r"^(-?\d+)", s).group(1))
    return (-999, n) if any(w in s.lower() for w in ("below", "lower", "less")) else (n, 999)


def k_bounds(r):
    if r.strike_type == "between":
        return int(r.floor), int(r.cap)
    if r.strike_type == "greater":
        return int(r.floor) + 1, 999
    return -999, int(r.cap) - 1


def load():
    km = pd.read_csv(os.path.join(D, "k_markets.csv"))
    km = km[km.date.notna() & km.strike_type.isin(["between", "greater", "less"])].copy()
    km[["lo", "hi"]] = [k_bounds(r) for r in km.itertuples()]
    km["k_won"] = (km.result == "yes").astype(int)
    pm = pd.read_csv(os.path.join(D, "pm_markets.csv"), dtype={"yes_token": str, "no_token": str})
    pm[["lo", "hi"]] = [pm_bounds(l) for l in pm.label]
    return km, pm


def settlement(km, pm):
    kw = km[km.k_won == 1].groupby(["city", "date"]).first()[["lo", "hi", "same_airport"]]
    pw = pm[pm.won == 1].groupby(["city", "date"]).first()[["lo", "hi"]]
    j = kw.join(pw, lsuffix="_k", rsuffix="_p", how="inner")
    j["k_below"] = j.hi_k < j.lo_p          # Kalshi's whole winning bracket below Polymarket's: "safe" logic broken
    j["k_above"] = j.lo_k > j.hi_p          # Kalshi clearly higher
    j["overlap"] = ~(j.k_below | j.k_above)
    out = j.groupby(["same_airport", "city"]).agg(days=("overlap", "size"), kalshi_clearly_higher=("k_above", "mean"),
                                                   overlapping=("overlap", "mean"), kalshi_clearly_LOWER=("k_below", "sum"))
    print("1) SETTLEMENT: Kalshi's winning bracket vs Polymarket's (brackets are offset by 1°F)")
    print(out.round(2).to_string())
    bad = j[j.same_airport & j.k_below]
    if len(bad):
        print("   same-airport days where Kalshi settled LOWER (would have broken the safe trade):")
        print(bad[["lo_k", "hi_k", "lo_p", "hi_p"]].to_string())
    return j


def ladders(km, pm):
    """Per city-day: minute grid of Kalshi asks (per bracket) and Polymarket mids (per bracket)."""
    kc = pd.read_parquet(os.path.join(D, "k_candles.parquet"))
    pp = pd.read_parquet(os.path.join(D, "pm_prices.parquet"))
    kc["t"] = kc.t // 60 * 60
    pp["t"] = pp.t // 60 * 60
    for (city, date), kb in km[km.same_airport].groupby(["city", "date"]):
        pb = pm[(pm.city == city) & (pm.date == date)]
        if pb.empty or pb.won.sum() != 1 or kb.k_won.sum() != 1:
            continue
        ka = kc[kc.ticker.isin(kb.ticker)].pivot_table(index="t", columns="ticker", values="yes_ask", aggfunc="last")
        pmid = pp[pp.yes_token.isin(pb.yes_token)].pivot_table(index="t", columns="yes_token", values="p", aggfunc="last")
        if ka.empty or pmid.empty:
            continue
        grid = np.arange(max(ka.index.min(), pmid.index.min()), min(ka.index.max(), pmid.index.max()) + 60, 60)
        if len(grid) == 0:
            continue
        ka = ka.reindex(grid).ffill(limit=FFILL_MIN).reindex(columns=kb.ticker)
        pmid = pmid.reindex(grid).ffill(limit=FFILL_MIN).reindex(columns=pb.yes_token)
        yield city, date, kb.set_index("ticker"), pb.set_index("yes_token"), ka, pmid


def scan(km, pm):
    sig = []
    for city, date, kb, pb, ka, pmid in ladders(km, pm):
        A = ka.to_numpy(float)                                   # minutes x Kalshi brackets (asks)
        P = pmid.to_numpy(float)                                 # minutes x Polymarket brackets (mids)
        ok = ~np.isnan(A).any(1) & ~np.isnan(P).any(1) & (np.abs(P.sum(1) - 1) <= 0.1)
        if not ok.any():
            continue
        A, P, t = A[ok], P[ok], ka.index.to_numpy()[ok]
        ka_cost = np.where(A >= 1, np.inf, A + K_FEE * A * (1 - A))           # no seller -> not buyable
        pc = np.minimum(P + PM_HALF_SPREAD, 0.999); pm_cost = pc + PM_FEE * pc * (1 - pc)
        k_lo, k_won = kb.lo.to_numpy(), kb.k_won.to_numpy()
        p_hi, p_lo, p_won = pb.hi.to_numpy(), pb.lo.to_numpy(), pb.won.to_numpy()
        for X in sorted(set(k_lo[k_lo > -500])):
            cand = p_lo[(p_lo >= X) & (p_lo > -500)]
            if len(cand) == 0:
                continue
            Y = cand.min()                                       # tightest Polymarket threshold >= X
            kset, pset = k_lo >= X, p_hi < Y
            if not kset.any() or not pset.any():
                continue
            cost = ka_cost[:, kset].sum(1) + pm_cost[:, pset].sum(1)
            edge = 1 - cost
            payoff = k_won[kset].sum() + p_won[pset].sum()
            for thr in (0.0, 0.01, 0.02, 0.05):
                hit = np.where(edge > thr)[0]
                if len(hit):
                    i = hit[0]
                    sig.append((city, date, X, Y, thr, len(hit), t[i], edge[i], edge[hit].max(), payoff, payoff - cost[i],
                                kset.sum() + pset.sum()))
    return pd.DataFrame(sig, columns=["city", "date", "X", "Y", "threshold", "minutes", "first_ts", "edge_first",
                                      "edge_max", "payoff", "profit_first", "legs"])


def report(s):
    print("\n2) THE SAFE TRADE: buy Kalshi '>= X' + Polymarket brackets below Y (same airport only), last 30 days")
    if s.empty:
        print("   never available after costs"); return
    g = s.groupby("threshold").agg(opportunities=("edge_first", "size"), city_days=("date", lambda d: d.nunique()),
                                   avg_minutes_open=("minutes", "mean"), avg_edge=("edge_first", "mean"),
                                   avg_profit=("profit_first", "mean"), total_per_pair=("profit_first", "sum"),
                                   payoff_below_1=("payoff", lambda p: (p < 1).sum()))
    g.index = [f"edge > {x:.0%}" for x in g.index]
    print(g.round(3).to_string())
    s0 = s[s.threshold == 0.01]
    if len(s0):
        print("\n   by city (edge > 1c):")
        print(s0.groupby("city").agg(opportunities=("edge_first", "size"), avg_edge=("edge_first", "mean"),
                                     avg_minutes=("minutes", "mean"), profit=("profit_first", "sum")).round(3).to_string())
        print("\n   biggest ones:")
        x = s0.sort_values("edge_first", ascending=False).head(10).copy()
        x["first_utc"] = pd.to_datetime(x.first_ts, unit="s").dt.strftime("%m-%d %H:%M")
        print(x[["city", "date", "X", "Y", "first_utc", "minutes", "edge_first", "payoff", "profit_first", "legs"]]
              .round(3).to_string(index=False))


def main():
    km, pm = load()
    print(f"Kalshi {km.ticker.nunique()} markets, Polymarket {pm.yes_token.nunique()} brackets, "
          f"{km.date.min()} .. {km.date.max()}\n")
    settlement(km, pm)
    report(scan(km, pm))


if __name__ == "__main__":
    main()
