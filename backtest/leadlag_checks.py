"""Robustness checks for the single-bracket lead-lag bet (clustered t-stats, opposite direction, placebo, split periods).
Run from the repo root: python3 -m backtest.leadlag_checks"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os
from backtest.leadlag import series, features, HALF_SPREAD, FEE
from backtest.kalshi_test import load, D
km, pm = load(); s = features(series(km, pm))

def pick(sig, direction):
    """direction: 'with' = bracket on Kalshi's side; 'against' = bracket on the other side."""
    up = (sig.dK > 0).to_numpy()
    if direction == "against": up = ~up
    price = np.where(up, sig.nb_up_p, sig.nb_dn_p); won = np.where(up, sig.nb_up_won, sig.nb_dn_won)
    tok_side = np.where(up, "nb_up", "nb_dn")
    return price, won, tok_side

def dedupe(sig):
    sig = sig.sort_values("t"); keep, last = [], {}
    for r in sig.itertuples():
        k = (r.city, r.date, r.X)
        if k in last and r.t - last[k] < 1800: continue
        last[k] = r.t; keep.append(r.Index)
    return sig.loc[keep]

def stats(price, won, day):
    ok = ~np.isnan(price) & ~np.isnan(won) & (price > .02) & (price < .98)
    price, won, day = price[ok], won[ok], day[ok]
    pnl = won - price - HALF_SPREAD - FEE * price * (1 - price)
    d = pd.Series(pnl).groupby(day).sum()            # cluster: one number per city-day
    return dict(bets=ok.sum(), city_days=len(d), gross_c=100*(won-price).mean(), profit_c=100*pnl.mean(),
                t_city_day=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))), total=pnl.sum())

rows = []
for thr in (0.05, 0.10):
    sig = dedupe(s[(s.dK.abs() >= thr) & (s.dP.abs() < thr/2)])
    day = (sig.city + sig.date.astype(str)).to_numpy()
    for direction in ("with", "against"):
        p, w, _ = pick(sig, direction)
        rows.append(dict(test=f"Kalshi >= {thr*100:.0f}c, bracket {direction} Kalshi", **stats(p, w, day)))
    half1 = (sig.date < "2026-11-12").to_numpy()
    p, w, _ = pick(sig, "with")
    rows.append(dict(test=f"  same, 27 Oct-11 Nov", **stats(np.where(half1, p, np.nan), w, day)))
    rows.append(dict(test=f"  same, 12-26 Nov", **stats(np.where(~half1, p, np.nan), w, day)))
    rows.append(dict(test=f"  same, same-airport cities", **stats(np.where(sig.same_airport, p, np.nan), w, day)))
# placebo: random minutes, no Kalshi signal, both neighbour brackets
rng = np.random.default_rng(0)
base = dedupe(s.sample(40000, random_state=0))
day = (base.city + base.date.astype(str)).to_numpy()
for side in ("nb_up", "nb_dn"):
    rows.append(dict(test=f"placebo: random minutes, {side}", **stats(base[f"{side}_p"].to_numpy(), base[f"{side}_won"].to_numpy(), day)))
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))

