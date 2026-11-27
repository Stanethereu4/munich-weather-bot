"""
backtest/leadlag.py
-------------------
Does one platform's price predict where the other is heading? (Kalshi vs Polymarket,
US daily highs, last 30 days; data from backtest/kalshi.py.)

Comparable numbers: "high is at least X" on each side (sum of the brackets from X up).
Brackets are offset by 1°F, so Polymarket's "at least X" is interpolated as the average of
its two neighbouring thresholds (X-1 and X+1). Kalshi prices = mid of its real bid/ask.
Polymarket prices = its 1-minute price (mid). Minutes are kept only when both ladders add
up to about 1 (0.9-1.1) and the threshold is undecided (both between 5% and 95%).

1) Prediction test, every minute: next move of Polymarket over h minutes
       dPM(t -> t+h) = a + b * dK(t-15 -> t) + c * dPM(t-15 -> t)
   b > 0 means Polymarket follows Kalshi. Same the other way round (does Kalshi follow
   Polymarket?). Standard errors clustered by city-day.
2) Trading test: when Kalshi moved >= 5c / 10c in 15 minutes and Polymarket moved less
   than half of that, buy the Polymarket side in Kalshi's direction and
     (a) sell it again 15 / 30 minutes later, or (b) hold it to resolution.
   Costs: buying "at least X" means buying several brackets; each costs half the
   typical spread on the way in and on the way out (1.75c, measured on the Telonex books)
   plus the 0.01 * p * (1 - p) fee. One trade per threshold per signal, no overlap.
   Example: Kalshi ">= 81" 40c -> 55c; Polymarket ">= 82" still 30c; buy it (2 brackets
   at +1.75c each), 15 min later it is 38c, sell (2 x -1.75c): +8 - 7 = +1c per set.

Run:  python3 -m backtest.leadlag
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest.kalshi_test import D, load

warnings.filterwarnings("ignore")
HALF_SPREAD = 0.0175
FEE = 0.01
LOOK = 15


def series(km, pm):
    kc = pd.read_parquet(os.path.join(D, "k_candles.parquet"))
    pp = pd.read_parquet(os.path.join(D, "pm_prices.parquet"))
    kc["t"] = kc.t // 60 * 60; pp["t"] = pp.t // 60 * 60
    kc["mid"] = (kc.yes_bid + kc.yes_ask) / 2
    out = []
    for (city, date), kb in km.groupby(["city", "date"]):
        pb = pm[(pm.city == city) & (pm.date == date)]
        if pb.empty or kb.k_won.sum() != 1 or pb.won.sum() != 1:
            continue
        K = kc[kc.ticker.isin(kb.ticker)].pivot_table(index="t", columns="ticker", values="mid", aggfunc="last")
        P = pp[pp.yes_token.isin(pb.yes_token)].pivot_table(index="t", columns="yes_token", values="p", aggfunc="last")
        if K.empty or P.empty:
            continue
        grid = np.arange(max(K.index.min(), P.index.min()), min(K.index.max(), P.index.max()) + 60, 60)
        if len(grid) < 120:
            continue
        K = K.reindex(grid).ffill(limit=10).reindex(columns=kb.ticker)
        P = P.reindex(grid).ffill(limit=10).reindex(columns=pb.yes_token)
        ok = ~K.isna().any(axis=1) & ~P.isna().any(axis=1) & (K.sum(1).sub(1).abs() <= .1) & (P.sum(1).sub(1).abs() <= .1)
        k_lo = kb.set_index("ticker").lo.reindex(K.columns).to_numpy()
        p_lo = pb.set_index("yes_token").lo.reindex(P.columns).to_numpy()
        p_won = pb.set_index("yes_token").won.reindex(P.columns).to_numpy()
        for X in sorted(set(k_lo[k_lo > -500])):
            kx = K.loc[:, k_lo >= X].sum(1)
            up, dn = (p_lo >= X + 1), (p_lo >= X - 1)
            if not up.any() or up.all() or not dn.any() or dn.all():
                continue
            px = (P.loc[:, up].sum(1) + P.loc[:, dn].sum(1)) / 2
            # brackets you would buy to bet on "higher" (>= X+1) or "lower" (< X-1 side), and outcome
            n_up, n_dn = int(up.sum()), int((~dn).sum())
            won_up = int(p_won[up].sum()); won_dn = int(p_won[~dn].sum())
            df = pd.DataFrame({"t": grid, "k": kx.to_numpy(), "p": px.to_numpy(), "ok": ok.to_numpy(),
                               "p_up": P.loc[:, up].sum(1).to_numpy(),          # price of the set ">= X+1"
                               "p_lo": 1 - P.loc[:, dn].sum(1).to_numpy()})     # price of the set "< X-1"
            df["city"], df["date"], df["X"] = city, date, X
            df["same_airport"] = bool(kb.same_airport.iloc[0])
            df["n_up"], df["n_dn"], df["won_up"], df["won_dn"] = n_up, n_dn, won_up, won_dn
            out.append(df)
    return pd.concat(out)


def features(s):
    s = s.sort_values(["city", "date", "X", "t"]).copy()
    g = s.groupby(["city", "date", "X"])
    s["dK"] = s.k - g.k.shift(LOOK)
    s["dP"] = s.p - g.p.shift(LOOK)
    for h in (5, 15, 30):
        s[f"fP{h}"] = g.p.shift(-h) - s.p
        s[f"fK{h}"] = g.k.shift(-h) - s.k
        s[f"fUp{h}"] = g.p_up.shift(-h) - s.p_up        # change in the price of the set actually bought
        s[f"fLo{h}"] = g.p_lo.shift(-h) - s.p_lo
    s["okall"] = s.ok & g.ok.shift(LOOK).fillna(False).astype(bool) & g.ok.shift(-30).fillna(False).astype(bool)
    s = s[s.okall & s.k.between(.05, .95) & s.p.between(.05, .95)].dropna(subset=["dK", "dP", "fP30", "fK30"])
    return s


def regressions(s):
    print("1) PREDICTION TEST (coefficients; t-stats clustered by city-day)")
    print("   next Polymarket move  = b * Kalshi's last-15-min move + c * Polymarket's own last-15-min move")
    print("   next Kalshi move      = b * Polymarket's last-15-min move + c * Kalshi's own last-15-min move\n")
    rows = []
    for subset, d in (("all 10 cities", s), ("same airport", s[s.same_airport])):
        cl = pd.factorize(d.city + d.date.astype(str))[0]
        for h in (5, 15, 30):
            for tgt, lead, own, name in ((f"fP{h}", "dK", "dP", "Polymarket follows Kalshi"),
                                         (f"fK{h}", "dP", "dK", "Kalshi follows Polymarket")):
                X = sm.add_constant(d[[lead, own]])
                m = sm.OLS(d[tgt], X).fit(cov_type="cluster", cov_kwds={"groups": cl})
                rows.append(dict(sample=subset, horizon_min=h, test=name, b_other=m.params[lead], t_other=m.tvalues[lead],
                                 c_own=m.params[own], t_own=m.tvalues[own], minutes=len(d)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def trades(s):
    print("\n2) TRADING TEST: buy Polymarket in Kalshi's direction after a Kalshi move it has not followed")
    res = []
    for thr in (0.05, 0.10):
        sig = s[(s.dK.abs() >= thr) & (s.dP.abs() < thr / 2)].copy()
        # no overlap: at most one signal per threshold every 30 minutes
        sig = sig.sort_values("t")
        keep, last = [], {}
        for r in sig.itertuples():
            k = (r.city, r.date, r.X)
            if k in last and r.t - last[k] < 1800:
                continue
            last[k] = r.t; keep.append(r.Index)
        sig = sig.loc[keep]
        up = sig.dK > 0
        n = np.where(up, sig.n_up, sig.n_dn)                  # brackets to buy
        price = np.where(up, sig.p_up, sig.p_lo)               # price of the set we actually buy
        fee = FEE * price * (1 - price)
        for h in (15, 30):
            move = np.where(up, sig[f"fUp{h}"], sig[f"fLo{h}"])
            pnl = move - 2 * n * HALF_SPREAD - 2 * fee
            res.append(dict(kalshi_move=f">= {thr*100:.0f}c", exit=f"sell after {h} min", trades=len(sig),
                            avg_follow_move_c=100 * move.mean(), cost_c=100 * (2 * n * HALF_SPREAD + 2 * fee).mean(),
                            profit_c=100 * pnl.mean(), t=pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl)))))
        won = np.where(up, sig.won_up, sig.won_dn)
        pnl = won - price - n * HALF_SPREAD - fee
        res.append(dict(kalshi_move=f">= {thr*100:.0f}c", exit="hold to resolution", trades=len(sig),
                        avg_follow_move_c=np.nan, cost_c=100 * (n * HALF_SPREAD + fee).mean(),
                        profit_c=100 * pnl.mean(), t=pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl)))))
    print(pd.DataFrame(res).round(2).to_string(index=False))


def main():
    km, pm = load()
    s = features(series(km, pm))
    print(f"{len(s):,} usable minutes, {s.groupby(['city', 'date']).ngroups} city-days, "
          f"{s.groupby(['city', 'date', 'X']).ngroups} thresholds\n")
    regressions(s)
    trades(s)


if __name__ == "__main__":
    main()
