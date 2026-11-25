"""
backtest/same_day.py
--------------------
Same-day tests on Munich HIGHEST-temperature markets: on the target day the
answer is revealed report by report (routine METAR at :20 and :50 UTC; the
official value is the max so far, rounded half up -- rebuilt from
metars.parquet it matches Polymarket's winner on all 356 markets).

A. Impossible brackets: once a report pushes the running max above a
   bracket, that bracket is certain to lose. How fast does its price reach ~0
   after the report, and what does it still trade at meanwhile?
B. Pricing of "it will still get warmer": at each local hour, the market's
   total price on all brackets above the current max vs how often the final
   max really ended above it.
C. Calibration through the day: at fixed local hours, do brackets priced
   p win with frequency p?

Prices are CLOB per-minute values (midpoint-like, not executable prices).
Uncertainties: day-level bootstrap (all brackets of a day move together).

Run:  python3 -m backtest.same_day
"""
import glob
import math
import os

import numpy as np
import pandas as pd

from backtest.event_study import load_prices
from munich_bot import parse_label_to_bounds

DATA = "backtest_data"
OUT = os.path.join(DATA, "analysis")
LAGS_MIN = [0, 2, 5, 10, 15, 30, 60]
HOURS = [9, 11, 13, 15, 17, 19, 21]
rng = np.random.default_rng(0)


def rh(x):
    return int(math.floor(x + 0.5))


def p_at(series, ts):
    t, p = series
    i = np.searchsorted(t, ts, side="right") - 1
    return p[i] if i >= 0 and ts - t[i] <= 600 else np.nan


def bootstrap_mean(values, groups, n=2000):
    """Mean and 95% CI, resampling whole days."""
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    if df.empty:
        return np.nan, np.nan, np.nan
    per = df.groupby("g")["v"].agg(["sum", "count"])
    idx = rng.integers(0, len(per), (n, len(per)))
    s, c = per["sum"].to_numpy()[idx].sum(1), per["count"].to_numpy()[idx].sum(1)
    est = per["sum"].sum() / per["count"].sum()
    return est, *np.percentile(s / c, [2.5, 97.5])


def main():
    mk = pd.read_csv(os.path.join(DATA, "markets.csv"), dtype={"yes_token": str})
    mk = mk[mk["kind"] == "high"]
    met = pd.read_parquet(os.path.join(DATA, "metars.parquet"))
    met = met[met["routine"]].dropna(subset=["temp_c"])
    met["date"] = pd.to_datetime(met["t"], unit="s", utc=True).dt.tz_convert("Europe/Berlin").dt.strftime("%Y-%m-%d")
    prices = load_prices(set(mk["yes_token"]))

    A, B, C = [], [], []
    for (date, slug), g in mk.groupby(["target_date", "event_slug"]):
        rep = met[met["date"] == date].sort_values("t")
        if len(rep) < 40:
            continue
        final = rh(rep["temp_c"].max())
        day_start = int(pd.Timestamp(date, tz="Europe/Berlin").timestamp())
        run_raw = rep["temp_c"].cummax().to_numpy()
        run_int = np.array([rh(x) for x in run_raw])
        rep_t = rep["t"].to_numpy()
        brk = [(r.bracket_label, parse_label_to_bounds(r.bracket_label), r.won, prices.get(r.yes_token))
               for r in g.itertuples()]

        def running_max_at(ts):
            k = np.searchsorted(rep_t, ts, side="right") - 1
            return run_int[k] if k >= 0 else None

        # ---- A: brackets made impossible by a report during the day ----
        for label, (lo, hi), won, s in brk:
            if s is None or hi is None:
                continue
            k = np.argmax(run_int > hi) if (run_int > hi).any() else None
            if k is None or k == 0 and False:
                continue
            t_k = rep_t[k]
            row = {"date": date, "label": label, "report_local": pd.Timestamp(t_k, unit="s", tz="UTC")
                   .tz_convert("Europe/Berlin").strftime("%H:%M")}
            for lag in LAGS_MIN:
                row[f"p+{lag}m"] = p_at(s, t_k + 60 * lag)
            row["p-10m"] = p_at(s, t_k - 600)
            A.append(row)

        # ---- B and C at fixed local hours ----
        for h in HOURS:
            ts = day_start + h * 3600
            R = running_max_at(ts)
            if R is None:
                continue
            above = []
            for label, (lo, hi), won, s in brk:
                if s is None:
                    continue
                p = p_at(s, ts)
                if np.isnan(p):
                    continue
                C.append({"date": date, "hour": h, "p": p, "won": won})
                if lo is not None and lo > R:          # bracket entirely above the current max
                    above.append(p)
            B.append({"date": date, "hour": h, "running_max": R, "final": final,
                      "mkt_p_higher": float(np.sum(above)) if above else 0.0,
                      "went_higher": int(final > R)})

    A, B, C = pd.DataFrame(A), pd.DataFrame(B), pd.DataFrame(C)
    os.makedirs(OUT, exist_ok=True)
    A.to_csv(os.path.join(OUT, "same_day_A.csv"), index=False)
    B.to_csv(os.path.join(OUT, "same_day_B.csv"), index=False)

    print(f"days: {B['date'].nunique()}")
    print("\nA. brackets ruled out by a report: median price after the report (cents)")
    cols = ["p-10m"] + [f"p+{l}m" for l in LAGS_MIN]
    print((A[cols].median() * 100).round(2).to_string())
    print("   mean price after the report (cents):")
    print((A[cols].mean() * 100).round(2).to_string())
    print(f"   share still >= 2c at +5 min: {(A['p+5m'] >= 0.02).mean():.1%} | at +15 min: "
          f"{(A['p+15m'] >= 0.02).mean():.1%} | n = {A['p+5m'].notna().sum()} ruled-out brackets "
          f"on {A['date'].nunique()} days")
    print("   price 10 min BEFORE the report that rules it out (median cents):", round(A["p-10m"].median() * 100, 2))

    print("\nB. 'it will still get warmer': market price vs reality, by local hour")
    rows = []
    for h, gb in B.groupby("hour"):
        mkt, mlo, mhi = bootstrap_mean(gb["mkt_p_higher"], gb["date"])
        act, alo, ahi = bootstrap_mean(gb["went_higher"], gb["date"])
        diff, dlo, dhi = bootstrap_mean(gb["mkt_p_higher"] - gb["went_higher"], gb["date"])
        rows.append({"hour": f"{h}:00", "days": len(gb), "market says": round(mkt, 3),
                     "actually happened": round(act, 3), "overpricing": round(diff, 3),
                     "95% CI": f"{dlo:+.3f} to {dhi:+.3f}"})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nC. calibration: brackets priced p, how often they won (by hour, priced 5-95c)")
    C = C[(C["p"] >= 0.05) & (C["p"] <= 0.95)]
    C["bucket"] = pd.cut(C["p"], [0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    for h in (11, 15, 19):
        gc = C[C["hour"] == h]
        t = gc.groupby("bucket", observed=True).agg(n=("won", "size"), avg_price=("p", "mean"), win_rate=("won", "mean"))
        print(f"  {h}:00"); print(t.round(3).to_string())


if __name__ == "__main__":
    main()
