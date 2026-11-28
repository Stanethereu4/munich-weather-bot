"""
backtest/leadlag_real.py
------------------------
The single-bracket lead-lag bet with REAL Polymarket prices, same-airport cities only.

Signal (as backtest/leadlag.py): Kalshi's "at least X" moved >= 5c (or >= 10c) in 15 min while
Polymarket moved less than half of that -> buy YES of the Polymarket bracket just above X
(Kalshi up) or just below X (Kalshi down).

Entry: only if someone really BOUGHT that bracket's YES within 10 minutes after the signal
(taker BUY of the YES token, or taker SELL of the NO token at q = YES bought at 1 - q).
We pay the first such price + the 0.01*p*(1-p) fee. No real purchase -> no bet.
"Shares seen" = shares bought at or below that price in those 10 minutes (lower bound on size).

Exit, two rules:
  hold         keep to resolution: $1 if the bracket wins, $0 if not
  take-profit  sell early the first time someone really SOLD into a bid (taker SELL of YES,
               or taker BUY of NO at q = YES sold at 1 - q) at >= entry + 4c; otherwise hold.
Example: bought at 0.27; 2 hours later a seller hits a 0.32 bid -> sell there: +5c - fees.

Run:  python3 -m backtest.leadlag_real
"""
import os
import warnings

import numpy as np
import pandas as pd

from backtest.kalshi_test import D, load
from backtest.leadlag import features, series

warnings.filterwarnings("ignore")
FEE, ENTRY_WIN, TP = 0.01, 600, 0.04


def dedupe(sig):
    # reset the index first: the minute table is built by concatenating many small tables, so its
    # index labels repeat, and .loc[labels] would bring back rows that should have been dropped
    # (bug found 28 Nov 2026: it inflated the 30-day bet counts)
    sig = sig.reset_index(drop=True).sort_values("t"); keep, last = [], {}
    for r in sig.itertuples():
        k = (r.city, r.date, r.X)
        if k in last and r.t - last[k] < 1800:
            continue
        last[k] = r.t; keep.append(r.Index)
    return sig.loc[keep]


def main():
    km, pm = load()
    km = km[km.same_airport]
    s = features(series(km, pm))
    pt = pd.read_parquet(os.path.join(D, "pm_trades.parquet"))
    pt["token_id"] = pt.token_id.astype(str)
    buys = pd.concat([pt[pt.side == "BUY"].assign(px=lambda d: d.price, tok=lambda d: d.token_id),
                      pt[pt.side == "SELL"].assign(px=lambda d: 1 - d.price, tok=lambda d: d.token_id)])
    sells = pd.concat([pt[pt.side == "SELL"].assign(px=lambda d: d.price, tok=lambda d: d.token_id),
                       pt[pt.side == "BUY"].assign(px=lambda d: 1 - d.price, tok=lambda d: d.token_id)])
    # index fills by YES token: yes-side rows keep their token; no-side rows are mapped to the YES token
    no2yes = dict(zip(pm.no_token.astype(str), pm.yes_token.astype(str)))
    for df in (buys, sells):
        df["yes_tok"] = df.tok.map(lambda t: no2yes.get(t, t))
    b_ok = buys[((buys.side == "BUY") & ~buys.tok.isin(no2yes)) | ((buys.side == "SELL") & buys.tok.isin(no2yes))]
    s_ok = sells[((sells.side == "SELL") & ~sells.tok.isin(no2yes)) | ((sells.side == "BUY") & sells.tok.isin(no2yes))]
    B = {k: g.sort_values("timestamp") for k, g in b_ok.groupby("yes_tok")}
    S = {k: g.sort_values("timestamp") for k, g in s_ok.groupby("yes_tok")}

    out = []
    for thr in (0.05, 0.10):
        sig = dedupe(s[(s.dK.abs() >= thr) & (s.dP.abs() < thr / 2)])
        for r in sig.itertuples():
            side = "nb_up" if r.dK > 0 else "nb_dn"
            tok, won, mid = getattr(r, f"{side}_tok"), getattr(r, f"{side}_won"), getattr(r, f"{side}_p")
            if tok is None or pd.isna(won) or tok not in B:
                continue
            g = B[tok]
            w = g[(g.timestamp >= r.t) & (g.timestamp <= r.t + ENTRY_WIN)]
            if w.empty:
                out.append(dict(thr=thr, city=r.city, date=r.date, t=r.t, filled=False)); continue
            e = float(w.px.iloc[0])
            if not 0.02 < e < 0.98:
                continue
            shares = float(w[w.px <= e + 1e-9]["size"].sum())
            cost = e + FEE * e * (1 - e)
            hold = won - cost
            tp = hold
            gs = S.get(tok)
            if gs is not None:
                later = gs[(gs.timestamp > w.timestamp.iloc[0]) & (gs.px >= e + TP)]
                if len(later):
                    x = float(later.px.iloc[0])
                    tp = x - FEE * x * (1 - x) - cost
            out.append(dict(thr=thr, city=r.city, date=r.date, t=r.t, filled=True, mid=mid, entry=e,
                            gap_vs_mid=e - mid, shares=shares, hold=hold, tp=tp, won=won))
    o = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    for thr, g in o.groupby("thr"):
        f = g[g.filled]
        day = f.city + f.date.astype(str)
        t = lambda x: (lambda d: d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))(x.groupby(day.to_numpy()).sum())
        print(f"\nKalshi move >= {thr*100:.0f}c, same-airport cities: {len(g)} signals, "
              f"{f.shape[0]} with a real purchase within 10 min ({len(f)/len(g):.0%})")
        print(f"   real entry price vs 1-minute mid: {100*f.gap_vs_mid.mean():+.1f}c on average "
              f"(we used to assume +1.75c); shares bought at that price: median {f.shares.median():.0f}")
        print(f"   HOLD:        {100*f.hold.mean():+.2f}c per bet, t (by city-day) {t(f.hold):.2f}, "
              f"win rate {f.won.mean():.0%} at avg price {f.entry.mean():.2f}")
        print(f"   TAKE-PROFIT: {100*f.tp.mean():+.2f}c per bet, t {t(f.tp):.2f}, "
              f"sold early on {np.mean(f.tp != f.hold):.0%} of bets")
        half = f.date < "2026-11-12"
        print(f"   by period (hold): 27 Oct-11 Nov {100*f[half].hold.mean():+.2f}c ({half.sum()} bets), "
              f"12-26 Nov {100*f[~half].hold.mean():+.2f}c ({(~half).sum()} bets)")
        per_day = f.groupby(day.to_numpy()).size()
        print(f"   bets per city-day: median {per_day.median():.0f}; city-days with bets: {len(per_day)}; "
              f"bets per calendar day (all 7 cities): {len(f)/f.date.nunique():.1f}")
    o.to_csv(os.path.join(D, "leadlag_real.csv"), index=False)


if __name__ == "__main__":
    main()
