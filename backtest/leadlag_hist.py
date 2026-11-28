"""
backtest/leadlag_hist.py
------------------------
The lead-lag bet on the long history (May to November 2026, 7 same-airport cities), with the same
real-fill rule as backtest/leadlag_real.py:

  signal  Kalshi's "at least X" moved >= 10c (or >= 5c) in 15 min, Polymarket less than half of that
  entry   only if someone really BOUGHT the neighbouring Polymarket bracket within 10 minutes;
          we pay that first price + the 0.01*p*(1-p) fee. Otherwise no bet.
  exit    hold to resolution ($1 if the bracket wins)

The rule was chosen on 27 Oct - 26 Nov, so everything before 27 Oct is NEW data for it
(out of sample); the results are shown for both parts.
Units: 1 share per bet, results in cents. t-stats are clustered by city-day.

Run:  python3 -m backtest.leadlag_hist
"""
import glob
import os

os.environ.setdefault("KALSHI_DIR", os.path.join("backtest_data", "kalshi_hist"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest.kalshi_test import D, load  # noqa: E402
from backtest.leadlag import signals_chunked  # noqa: E402
from backtest.leadlag_real import dedupe  # noqa: E402

FEE, WIN = 0.01, 600
SPLIT = "2026-10-27"          # first day of the period the rule was chosen on


def buys_by_token(pm):
    """{YES token: DataFrame(t, px)} of real purchases of YES (BUY of YES, or SELL of NO at 1 - price)."""
    no2yes = dict(zip(pm.no_token.astype(str), pm.yes_token.astype(str)))
    parts = []
    for f in sorted(glob.glob(os.path.join(D, "pm_trades_*.parquet"))):
        df = pd.read_parquet(f, columns=["token_id", "side", "price", "size", "t"])
        tok = df.token_id.astype(str)
        is_no = tok.isin(no2yes)
        keep = (df.side == "BUY") != is_no
        parts.append(pd.DataFrame({"tok": tok[keep].map(lambda x: no2yes.get(x, x)),
                                   "t": df.t[keep], "px": np.where(is_no[keep], 1 - df.price[keep], df.price[keep])}))
    b = pd.concat(parts, ignore_index=True)
    return {k: g.sort_values("t") for k, g in b.groupby("tok")}


def tstat(x):
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def main():
    km, pm = load()
    km = km[km.same_airport]
    s = signals_chunked(km, pm)
    B = buys_by_token(pm)
    print(f"\nPolymarket purchases loaded for {len(B)} brackets\n")
    rows = []
    for thr in (0.10, 0.05):
        sig = dedupe(s[(s.dK.abs() >= thr) & (s.dP.abs() < thr / 2)])
        for r in sig.itertuples():
            side = "nb_up" if r.dK > 0 else "nb_dn"
            tok, won = getattr(r, f"{side}_tok"), getattr(r, f"{side}_won")
            if tok is None or pd.isna(won):
                continue
            g = B.get(str(tok))
            w = None if g is None else g[(g.t >= r.t) & (g.t <= r.t + WIN)]
            if w is None or w.empty:
                rows.append(dict(thr=thr, city=r.city, date=r.date, filled=False)); continue
            e = float(w.px.iloc[0])
            if not 0.02 < e < 0.98:
                continue
            rows.append(dict(thr=thr, city=r.city, date=r.date, filled=True, entry=e, won=won,
                             pnl=100 * (won - e - FEE * e * (1 - e))))
    o = pd.DataFrame(rows)
    o.to_csv(os.path.join(D, "leadlag_hist.csv"), index=False)

    for thr, g in o.groupby("thr", sort=False):
        f = g[g.filled]
        print(f"=== Kalshi move >= {thr*100:.0f}c: {len(g)} signals, {len(f)} with a real purchase within 10 min ===")
        for name, part in (("NEW days (1 May - 26 Oct, never used to pick the rule)", f[f.date < SPLIT]),
                           ("tuning period (27 Oct - 27 Nov)", f[f.date >= SPLIT]), ("all", f)):
            day = part.groupby(["city", "date"]).pnl.sum()
            print(f"  {name}: {len(part)} bets on {len(day)} city-days, {part.pnl.mean():+.2f}c per bet, "
                  f"{day.mean():+.1f}c per city-day, t {tstat(day):.2f}, win rate {part.won.mean():.0%} "
                  f"at avg price {part.entry.mean():.2f}")
        f = f.assign(month=f.date.str[:7])
        by = f.groupby("month").agg(bets=("pnl", "size"), per_bet=("pnl", "mean"), total=("pnl", "sum"))
        by["city_days"] = f.groupby("month").apply(lambda x: x[["city", "date"]].drop_duplicates().shape[0])
        by["t"] = f.groupby("month").apply(lambda x: tstat(x.groupby(["city", "date"]).pnl.sum()))
        print("  by month:\n" + by.round(2).to_string())
        byc = f.groupby("city").agg(bets=("pnl", "size"), per_bet=("pnl", "mean"), total=("pnl", "sum"))
        byc["t"] = f.groupby("city").apply(lambda x: tstat(x.groupby("date").pnl.sum()))
        print("  by city:\n" + byc.round(2).to_string())
        day = f.groupby(["city", "date"]).pnl.sum().sort_values()
        print(f"  without the 5 best city-days: {day.iloc[:-5].sum():+.0f}c (all: {day.sum():+.0f}c); "
              f"best day {day.index[-1]} {day.iloc[-1]:+.0f}c, worst {day.index[0]} {day.iloc[0]:+.0f}c")
        boot = [np.random.default_rng(i).choice(day.to_numpy(), len(day)).sum() for i in range(2000)]
        print(f"  bootstrap over city-days, total: 5% {np.percentile(boot, 5):+.0f}c, "
              f"95% {np.percentile(boot, 95):+.0f}c, share of draws > 0: {np.mean(np.array(boot) > 0):.0%}\n")


if __name__ == "__main__":
    main()
