"""
backtest/kalshi_fills.py
------------------------
Step 1 of checking the Kalshi/Polymarket "safe" trade (backtest/kalshi_test.py):
could each leg really have been bought at the assumed price?

For every opportunity (first minute with edge > 1c per city/day/threshold), and every leg:
  Polymarket leg (buy YES of a bracket, assumed at mid + 1c): look for real taker
      purchases of that YES within +-5 minutes: taker BUY of the YES token at p,
      or taker SELL of the NO token at q (= YES bought at 1 - q).
  Kalshi leg (buy YES, assumed at the candle's ask): real trades where the taker
      bought YES at price p.
  A leg is "verified" if such a trade happened at or below the assumed price.
  "Shares seen" = shares traded at or below the assumed price in the window: a lower
  bound on what was available (more may have been resting).
Then the pair cost is recomputed with the closest-in-time real trade price per leg.
Example: assumed Polymarket 46-47°F at 0.107 (mid 0.097 + 1c); a taker bought it at
0.10 two minutes later for 150 shares -> verified, real price 0.10, 150 shares seen.

Run:  python3 -m backtest.kalshi_fills    (needs backtest_data/kalshi incl. pm_trades.parquet)
"""
import os

import numpy as np
import pandas as pd

from backtest.kalshi_test import D, K_FEE, PM_FEE, PM_HALF_SPREAD, ladders, load

WIN = 300
CHEAP = 0.03          # legs this cheap barely move the result; "lenient" check allows them unverified


SAFETY_S = 60        # use a report only from 60 s after the source received it


def load_floor():
    """Running daily high (°F, Polymarket rules) per city/local date, as known at each receipt time.
    Source: aviationweather.gov = Polymarket's resolution source (backtest/awc_obs.py)."""
    ob = pd.read_csv(os.path.join(D, "awc_obs.csv"))
    ob = ob[ob.routine & ob.temp_f.notna() & ob.receipt_ts.notna()].sort_values("receipt_ts")
    fl = {}
    for (city, date), g in ob.groupby(["city", "local_date"]):
        fl[(city, date)] = (g.receipt_ts.to_numpy() + SAFETY_S, np.maximum.accumulate(g.temp_f.to_numpy()))
    return ob, fl


def floor_at(fl, city, date, t):
    """Highest routine reading of that local day already public at times t (array); -999 if none yet."""
    if (city, date) not in fl:
        return np.full(len(t), -999)
    rt, cm = fl[(city, date)]
    i = np.searchsorted(rt, t, side="right") - 1
    return np.where(i >= 0, cm[np.clip(i, 0, None)], -999)


def validate(ob, pm):
    """Rebuilt daily high from the resolution source must fall in Polymarket's winning bracket."""
    mx = ob.groupby(["city", "local_date"]).temp_f.max().rename("max_f").reset_index()
    w = pm[pm.won == 1].merge(mx, left_on=["city", "date"], right_on=["city", "local_date"])
    w["ok"] = (w.max_f >= w.lo) & (w.max_f <= w.hi)
    print(f"check: rebuilt daily high inside Polymarket's winning bracket on {w.ok.sum()} of {len(w)} city-days "
          f"({w.ok.mean():.1%})")
    if (~w.ok).any():
        print(w.loc[~w.ok, ["city", "date", "label", "max_f"]].head(10).to_string(index=False))
    return set(zip(w.loc[w.ok, "city"], w.loc[w.ok, "date"]))


def signals(km, pm, thr=0.01, fl=None, valid_days=None):
    out = []
    for city, date, kb, pb, ka, pmid in ladders(km, pm):
        if valid_days is not None and (city, date) not in valid_days:
            continue
        A, P = ka.to_numpy(float), pmid.to_numpy(float)
        ok = ~np.isnan(A).any(1) & ~np.isnan(P).any(1) & (np.abs(P.sum(1) - 1) <= 0.1)
        if not ok.any():
            continue
        t_all = ka.index.to_numpy()
        kac = np.where(A >= 1, np.inf, A + K_FEE * A * (1 - A))
        pc = np.minimum(P + PM_HALF_SPREAD, 0.999); pmc = pc + PM_FEE * pc * (1 - pc)
        k_lo, p_lo, p_hi = kb.lo.to_numpy(), pb.lo.to_numpy(), pb.hi.to_numpy()
        k_hi = kb.hi.to_numpy()
        F = floor_at(fl, city, date, t_all) if fl is not None else np.full(len(t_all), -999)
        # dead legs cost nothing: a bracket whose top is below the floor can no longer win
        # (Polymarket: its own high >= floor; Kalshi: its high >= Polymarket's)
        kac = np.where(k_hi[None, :] < F[:, None], 0.0, kac)
        pmc = np.where(p_hi[None, :] < F[:, None], 0.0, pmc)
        for X in sorted(set(k_lo[k_lo > -500])):
            cand = p_lo[(p_lo >= X) & (p_lo > -500)]
            if len(cand) == 0:
                continue
            Y = cand.min()
            ks, ps = k_lo >= X, p_hi < Y
            edge = 1 - kac[:, ks].sum(1) - pmc[:, ps].sum(1)
            hit = np.where(ok & (edge > thr))[0]
            if not len(hit):
                continue
            i = hit[0]
            live_k = ks & (k_hi >= F[i]); live_p = ps & (p_hi >= F[i])
            legs = [("K", tk, A[i, j], None) for j, tk in enumerate(ka.columns) if live_k[j]] + \
                   [("P", tok, P[i, j] + PM_HALF_SPREAD, pb.loc[tok, "no_token"]) for j, tok in enumerate(pmid.columns) if live_p[j]]
            out.append(dict(city=city, date=date, X=X, Y=Y, t=int(t_all[i]), edge=edge[i], floor=int(F[i]),
                            dead_legs=int((ks & ~live_k).sum() + (ps & ~live_p).sum()),
                            payoff=int(kb.k_won.to_numpy()[live_k].sum() + pb.won.to_numpy()[live_p].sum()), legs=legs))
    return out


def main(use_floor=False):
    km, pm = load()
    fl = valid = None
    if use_floor:
        ob, fl = load_floor()
        valid = validate(ob, pm)
        km = km[pd.to_datetime(km.date) >= "2026-11-01"]          # resolution-source history starts 1 Nov
        pm = pm[pd.to_datetime(pm.date) >= "2026-11-01"]
    kt = pd.read_parquet(os.path.join(D, "k_trades.parquet"))
    kt = kt[kt.taker_side == "yes"]
    kt_by = {t: g for t, g in kt.groupby("ticker")}
    pt = pd.read_parquet(os.path.join(D, "pm_trades.parquet"))
    pt["token_id"] = pt.token_id.astype(str)
    yes_buys = pt[pt.side == "BUY"].assign(yes_px=lambda d: d.price)
    no_sells = pt[pt.side == "SELL"].assign(yes_px=lambda d: 1 - d.price)
    pt_yes = {t: g for t, g in yes_buys.groupby("token_id")}
    pt_no = {t: g for t, g in no_sells.groupby("token_id")}

    rows = []
    for s in signals(km, pm, fl=fl, valid_days=valid):
        legs_out = []
        for plat, key, assumed, no_tok in s["legs"]:
            if plat == "K":
                g = kt_by.get(key)
                tr = g[(g.t >= s["t"] - WIN) & (g.t <= s["t"] + WIN)].rename(columns={"yes_price": "px", "count": "sz"}) \
                    if g is not None else pd.DataFrame(columns=["t", "px", "sz"])
            else:
                parts = []
                for d, k in ((pt_yes, key), (pt_no, str(no_tok))):
                    g = d.get(k)
                    if g is not None:
                        parts.append(g[(g.timestamp >= s["t"] - WIN) & (g.timestamp <= s["t"] + WIN)]
                                     .rename(columns={"timestamp": "t", "yes_px": "px", "size": "sz"})[["t", "px", "sz"]])
                tr = pd.concat(parts) if parts else pd.DataFrame(columns=["t", "px", "sz"])
            ok_tr = tr[tr.px <= assumed + 1e-9]
            closest = tr.iloc[(tr.t - s["t"]).abs().argsort()[:1]] if len(tr) else None
            real_px = float(closest.px.iloc[0]) if closest is not None else np.nan
            fee = (K_FEE if plat == "K" else PM_FEE)
            legs_out.append(dict(plat=plat, assumed=assumed, verified=len(ok_tr) > 0, shares=float(ok_tr.sz.sum()),
                                 real_px=real_px, cheap=assumed < CHEAP,
                                 real_cost=(real_px if not np.isnan(real_px) else assumed) +
                                 fee * min(real_px if not np.isnan(real_px) else assumed, 1) *
                                 (1 - min(real_px if not np.isnan(real_px) else assumed, 1))))
        L = pd.DataFrame(legs_out)
        rows.append(dict(city=s["city"], date=s["date"], edge_assumed=s["edge"], payoff=s["payoff"], legs=len(L),
                         dead_legs=s["dead_legs"], floor=s["floor"],
                         all_verified=L.verified.all(), lenient=(L.verified | L.cheap).all(),
                         pm_verified=L[L.plat == "P"].verified.mean(), k_verified=L[L.plat == "K"].verified.mean(),
                         min_shares=L[~L.cheap].shares.min() if (~L.cheap).any() else np.nan,
                         edge_real=1 - L.real_cost.sum(), real_px_missing=L.real_px.isna().sum()))
    r = pd.DataFrame(rows)
    r["profit_real"] = r.payoff - (1 - r.edge_real)
    print(f"{len(r)} opportunities (edge > 1c at first sight), same-airport cities, "
          f"{'1-26 Nov, dead legs removed with the resolution source' if use_floor else 'last 30 days'}; "
          f"live legs per trade: median {r.legs.median():.0f}, dead legs removed: median {r.dead_legs.median():.0f}\n")
    print("Share of legs with a real purchase at or below the assumed price within +-5 min:")
    print(f"   Polymarket legs: {r.pm_verified.mean():.0%}   Kalshi legs: {r.k_verified.mean():.0%}")
    print(f"Opportunities where EVERY leg was verified: {r.all_verified.mean():.0%} ({r.all_verified.sum()}); "
          f"every leg except ones under {CHEAP*100:.0f}c: {r.lenient.mean():.0%} ({r.lenient.sum()})\n")
    for name, sub in (("all opportunities", r), ("strict: every leg verified", r[r.all_verified]),
                      ("lenient: every leg >= 3c verified", r[r.lenient])):
        if len(sub) == 0:
            print(f"{name}: none"); continue
        print(f"{name}: n={len(sub)}, assumed edge {sub.edge_assumed.mean()*100:.1f}c, edge at real trade prices "
              f"{sub.edge_real.mean()*100:.1f}c (positive in {np.mean(sub.edge_real > 0):.0%}), realised profit at real "
              f"prices {sub.profit_real.mean()*100:.1f}c/pair, total ${sub.profit_real.sum():.2f}; "
              f"smallest leg's shares seen: median {sub.min_shares.median():.0f}")
    print("\nby city (lenient):")
    print(r[r.lenient].groupby("city").agg(n=("edge_real", "size"), edge_real=("edge_real", "mean"),
                                           profit_real=("profit_real", "sum"), min_shares=("min_shares", "median")).round(3).to_string())
    r.to_csv(os.path.join(D, "fills_check_floor.csv" if use_floor else "fills_check.csv"), index=False)


if __name__ == "__main__":
    import sys
    main(use_floor=len(sys.argv) > 1 and sys.argv[1] == "floor")
