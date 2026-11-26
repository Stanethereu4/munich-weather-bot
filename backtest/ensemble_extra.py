"""
backtest/ensemble_extra.py
--------------------------
Two follow-up questions on the ensemble super-model (daily HIGHS only; low
prices are incoherent):

1) "Is there a combination of model weights that beats the market?"
   Upper-bound test: choose weights, bias and spread->sigma rule WITH HINDSIGHT
   (fitted on the very days we score). This is deliberately optimistic: if even
   the best possible weights can't beat the market, no weights can. If they do,
   we re-check with honest leave-one-day-out fitting.

2) "Bet whenever model and market disagree, not at fixed times."
   Every 15 minutes while the market is open (until the target day starts),
   compare the model with the price; take the FIRST signal per bracket where
   expected profit > threshold after a crossing cost and the fee, hold to
   resolution. Model = the freshest ensemble forecast already published
   (issued 3 days earlier from D-2 09 UTC, 2 days earlier from D-1 09 UTC).

Run:  python3 -m backtest.ensemble_extra      (needs backtest_data/, no network)
"""
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from munich_bot import parse_label_to_bounds
from backtest.ensemble_test import (DATA, daily_super, calibrate, load_prices, price_at, decision_ts,
                                    FEE_RATE)

BERLIN = ZoneInfo("Europe/Berlin")


def per_model(ens, kind="high"):
    """Each model's daily max (of the hourly mean) and its spread at that hour."""
    e = ens.copy()
    e["date"] = pd.to_datetime(e["t"], unit="s", utc=True).dt.tz_convert(BERLIN).dt.strftime("%Y-%m-%d")
    e = e[e.groupby(["model", "lead_days", "date"])["mean"].transform("size") >= 20]
    idx = e.groupby(["model", "lead_days", "date"])["mean"].idxmax() if kind == "high" else \
        e.groupby(["model", "lead_days", "date"])["mean"].idxmin()
    d = e.loc[idx, ["model", "lead_days", "date", "mean", "spread"]]
    x = d.pivot_table(index=["lead_days", "date"], columns="model", values="mean").dropna()
    s = d.pivot_table(index=["lead_days", "date"], columns="model", values="spread").reindex(x.index)
    return x, s


def bracket_matrix(markets):
    """date -> (lower bounds, upper bounds, won array, tokens) for HIGH markets."""
    out = {}
    for d, g in markets[markets["kind"] == "high"].groupby("target_date"):
        b = [parse_label_to_bounds(l) for l in g["bracket_label"]]
        lo = np.array([-np.inf if a is None else a for a, _ in b])
        hi = np.array([np.inf if c is None else c for _, c in b])
        out[d] = (lo, hi, g["won"].to_numpy(int), g["yes_token"].to_numpy())
    return out


def probs(mu, sig, lo, hi):
    p = norm.cdf((hi - mu) / sig) - norm.cdf((lo - mu) / sig)
    return p / p.sum()


# ------------------------------------------------------------ 1) best weights --
def oracle(lead, x, s, br, prices):
    """Fit weights/bias/sigma rule on the scored days themselves (hindsight)."""
    days = []
    for d in x.loc[lead].index:
        if d not in br:
            continue
        lo, hi, won, toks = br[d]
        m = np.array([price_at(prices, t, decision_ts(d, lead)) for t in toks])
        if np.isnan(m).any() or not 0.9 <= m.sum() <= 1.1:
            continue
        days.append((d, x.loc[(lead, d)].to_numpy(), s.loc[(lead, d)].to_numpy(), lo, hi, won, m / m.sum()))
    X = np.array([r[1] for r in days]); S = np.array([r[2] for r in days])
    U = np.sqrt((S ** 2).mean(1) + X.var(1))

    def model_p(th, i):
        w, c, a, b = th[:4], th[4], th[5], th[6]
        mu = X[i] @ w + c
        sig = np.exp(a + b * np.log(U[i]))
        _, _, lo, hi, _, _ = days[i][1:]
        return probs(mu, max(sig, 0.3), lo, hi)

    def brier(th, idx):
        return np.mean([((model_p(th, i) - days[i][5]) ** 2).sum() for i in idx])

    th0 = np.array([.25, .25, .25, .25, 0, np.log(1.2), 0])
    allidx = range(len(days))
    fit = minimize(brier, th0, args=(allidx,), method="Powell", options={"maxiter": 20000, "xtol": 1e-4, "ftol": 1e-6})
    mkt = np.mean([((days[i][6] - days[i][5]) ** 2).sum() for i in allidx])
    # honest version: leave-one-day-out
    loo = []
    for i in allidx:
        others = [j for j in allidx if j != i]
        f = minimize(brier, fit.x, args=(others,), method="Powell", options={"maxiter": 3000, "xtol": 1e-3, "ftol": 1e-5})
        loo.append(((model_p(f.x, i) - days[i][5]) ** 2).sum())
    return len(days), fit.fun, mkt, np.mean(loo), fit.x


# --------------------------------------------------- 2) bet whenever it's off --
def continuous(cal, br, prices, markets, thresholds=(0.05, 0.10, 0.20), half_spreads=(0.01, 0.02)):
    cal = cal[cal["kind"] == "high"].set_index(["date", "lead"])
    starts = markets[markets["kind"] == "high"].groupby("target_date")["market_start"].first()
    res = []
    for hs in half_spreads:
        for th in thresholds:
            trades = []
            for d, (lo, hi, won, toks) in br.items():
                if d not in starts.index:
                    continue
                open_ts = int(pd.Timestamp(starts[d]).timestamp())
                day_start = int(datetime.fromisoformat(d).replace(tzinfo=BERLIN).timestamp())
                taken = set()
                for t in range(open_ts + 900, day_start, 900):
                    lead = 2 if t >= decision_ts(d, 2) else (3 if t >= decision_ts(d, 3) else None)
                    if lead is None or (d, lead) not in cal.index:
                        continue
                    r = cal.loc[(d, lead)]
                    pm = probs(r["mu"], r["sigma_spread"], lo, hi)
                    mk = np.array([price_at(prices, tok, t) for tok in toks])
                    if np.isnan(mk).any() or not 0.9 <= mk.sum() <= 1.1:
                        continue
                    for j in range(len(toks)):
                        yp, npx = mk[j] + hs, 1 - mk[j] + hs
                        if (j, "Y") not in taken and pm[j] - yp - FEE_RATE * yp * (1 - yp) > th and yp < 1:
                            taken.add((j, "Y")); trades.append((d, won[j] - yp - FEE_RATE * yp * (1 - yp)))
                        if (j, "N") not in taken and (1 - pm[j]) - npx - FEE_RATE * npx * (1 - npx) > th and npx < 1:
                            taken.add((j, "N")); trades.append((d, (1 - won[j]) - npx - FEE_RATE * npx * (1 - npx)))
            t = pd.DataFrame(trades, columns=["date", "pnl"])
            per_day = t.groupby("date")["pnl"].sum() if len(t) else pd.Series(dtype=float)
            tstat = per_day.mean() / (per_day.std(ddof=1) / np.sqrt(len(per_day))) if len(per_day) > 2 else 0
            res.append((hs, th, len(t), t["pnl"].sum() if len(t) else 0, t["pnl"].mean() if len(t) else 0,
                        (t["pnl"] > 0).mean() if len(t) else 0, len(per_day), tstat))
    return pd.DataFrame(res, columns=["half_spread", "threshold", "trades", "pnl", "per_trade", "win_rate", "days", "t"])


def main():
    ens = pd.read_parquet(os.path.join(DATA, "ensemble", "ens_means.parquet"))
    station = pd.read_csv(os.path.join(DATA, "station.csv"))
    markets = pd.read_csv(os.path.join(DATA, "markets.csv"), dtype={"yes_token": str})
    markets = markets[markets["won"].notna()]
    prices = load_prices(set(markets["yes_token"]))
    br = bracket_matrix(markets)

    print("1) BEST POSSIBLE WEIGHTS (fitted with hindsight on the scored days) -- highs, Brier per day")
    x, s = per_model(ens)
    for lead, when in ((1, "same morning 09 UTC"), (2, "day before 09 UTC")):
        n, b_or, b_mkt, b_loo, th = oracle(lead, x, s, br, prices)
        print(f"  N={lead} ({when}), {n} days: hindsight-best model {b_or:.3f} | honest leave-one-out {b_loo:.3f} "
              f"| market {b_mkt:.3f}")
        print(f"      weights {dict(zip([c.split('_')[0] + '_' + c.split('_')[1] for c in x.columns], np.round(th[:4], 2)))}"
              f", bias {th[4]:+.2f}, sigma = {np.exp(th[5]):.2f} * u^{th[6]:.2f}")

    print("\n2) BET WHENEVER MODEL AND MARKET DISAGREE (every 15 min until the day starts, first signal per bracket)")
    cal = calibrate(daily_super(ens), station)
    out = continuous(cal, br, prices, markets)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
