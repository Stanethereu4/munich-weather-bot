"""
backtest/ensemble_test.py
-------------------------
"Super-model" test: is an average of 4 ensemble means (ECMWF IFS, ICON-EPS,
GEFS, ECMWF AIFS), with an uncertainty taken from the ensembles' spread, more
accurate than the Polymarket price?

No look-ahead:
  * Forecasts: "previous_dayN" values (backtest/ensemble.py). For target day D,
    every hour of D comes from runs started >= N*24 h earlier. The last hour
    of the Munich day (22-23 UTC) therefore comes from a run started by
    D-N 23 UTC at the latest, published within ~9 h. So the forecast is
    safely known at   T = D-(N-1) 09:00 UTC  -> that is when we read the price.
      N=1 -> morning of the target day itself (market already sees the
             morning temperatures: unfair to the model for LOWS, since the
             minimum usually happens around dawn; mild for highs)
      N=2 -> the day before, N=3 -> two days before (clean)
  * Calibration (bias and sigma) uses only days whose station value was known
    at T: the 30 most recent days up to D-N.

Super-model for day D, lead N, kind (high/low):
  x      = average of the 4 models' daily max (or min) of the hourly mean
  u      = sqrt(mean spread^2 + variance between the 4 models)  (day's uncertainty)
  mu     = x + bias          bias  = mean(station - x) over the training days
  sigma  "flat":   RMSE of (station - x - bias) over the training days
         "spread": flat sigma * u / mean(u over training days), clipped to 0.5x..2x
  probabilities per bracket with the live bot's calculate_bracket_probability()

Outputs printed: Brier/log loss vs market, 50/50 mix test, paper-trading PnL,
and a detail table for the last 3 resolved days.

Run:  python3 -m backtest.ensemble_test        (needs backtest_data/, no network)
"""
import glob
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from munich_bot import parse_label_to_bounds, calculate_bracket_probability

DATA = "backtest_data"
BERLIN = ZoneInfo("Europe/Berlin")
TRAIN_DAYS, MIN_TRAIN = 30, 14
MAX_STALENESS_S = 2 * 3600
FEE_RATE = 0.01          # weather fee since 20 Nov (applied to all days: conservative)
HALF_SPREAD = 0.01       # assumed cost of crossing the spread (we only have mids)
EDGE = 0.05              # trade only if expected profit per share > 5 cents


# ---------------------------------------------------------------- forecasts --
def daily_super(ens):
    """ens: model, t, lead_days, mean, spread -> per (date, lead, kind): x, u."""
    e = ens.copy()
    e["date"] = pd.to_datetime(e["t"], unit="s", utc=True).dt.tz_convert(BERLIN).dt.strftime("%Y-%m-%d")
    rows = []
    for (m, n, d), g in e.groupby(["model", "lead_days", "date"]):
        if len(g) < 20:
            continue
        hi, lo = g.loc[g["mean"].idxmax()], g.loc[g["mean"].idxmin()]
        rows.append((m, n, d, "high", hi["mean"], hi["spread"]))
        rows.append((m, n, d, "low", lo["mean"], lo["spread"]))
    per = pd.DataFrame(rows, columns=["model", "lead", "date", "kind", "val", "spread"])
    nmod = per["model"].nunique()
    agg = per.groupby(["lead", "date", "kind"]).agg(
        x=("val", "mean"), between=("val", "var"), s2=("spread", lambda s: np.mean(s ** 2)), k=("val", "size"))
    agg = agg[agg["k"] == nmod].reset_index()
    agg["u"] = np.sqrt(agg["s2"] + agg["between"])
    return agg[["lead", "date", "kind", "x", "u"]]


def calibrate(sup, station):
    """Adds mu, sigma_flat, sigma_spread using only days <= D-N."""
    st = station.set_index("date")
    out = []
    for (n, kind), g in sup.groupby(["lead", "kind"]):
        g = g.sort_values("date").copy()
        g["truth"] = g["date"].map(st[kind])
        for r in g.itertuples(index=False):
            last_known = (datetime.fromisoformat(r.date) - timedelta(days=int(n))).strftime("%Y-%m-%d")
            tr = g[(g["date"] <= last_known) & g["truth"].notna()].tail(TRAIN_DAYS)
            if len(tr) < MIN_TRAIN:
                continue
            bias = float((tr["truth"] - tr["x"]).mean())
            rmse = float(np.sqrt(((tr["truth"] - tr["x"] - bias) ** 2).mean()))
            flat = max(rmse, 0.5)
            spread = float(np.clip(flat * r.u / tr["u"].mean(), 0.5 * flat, 2 * flat))
            out.append((n, r.date, kind, r.x + bias, flat, max(spread, 0.5), r.truth, len(tr)))
    return pd.DataFrame(out, columns=["lead", "date", "kind", "mu", "sigma_flat", "sigma_spread", "truth", "ntrain"])


# ------------------------------------------------------------------- prices --
def load_prices(tokens):
    prices = {}
    for path in glob.glob(os.path.join(DATA, "prices", "*.parquet")):
        df = pd.read_parquet(path)
        df = df[df["yes_token"].isin(tokens)]
        for tok, g in df.groupby("yes_token"):
            g = g.sort_values("t")
            prices[tok] = (g["t"].to_numpy(), g["p"].to_numpy(dtype=float))
    return prices


def price_at(prices, tok, ts):
    if tok not in prices:
        return np.nan
    t, p = prices[tok]
    i = np.searchsorted(t, ts, side="right") - 1
    if i < 0 or ts - t[i] > MAX_STALENESS_S:
        return np.nan
    return p[i]


def decision_ts(date, lead):
    d = datetime.fromisoformat(date).replace(tzinfo=timezone.utc) - timedelta(days=int(lead) - 1)
    return int(d.replace(hour=9).timestamp())


# --------------------------------------------------------------- evaluation --
def build_rows(cal, markets, prices):
    rows = []
    mk = markets.groupby(["kind", "target_date"])
    for r in cal.itertuples(index=False):
        if (r.kind, r.date) not in mk.groups:
            continue
        b = mk.get_group((r.kind, r.date))
        ts = decision_ts(r.date, r.lead)
        mkt = np.array([price_at(prices, tok, ts) for tok in b["yes_token"]])
        if np.isnan(mkt).any() or not (0.9 <= mkt.sum() <= 1.1):
            continue                              # market missing/not open/incoherent at T
        mkt = mkt / mkt.sum()                     # normalise to a probability set
        for var, sig in (("flat", r.sigma_flat), ("spread", r.sigma_spread)):
            p = np.array([calculate_bracket_probability(r.mu, sig, parse_label_to_bounds(l)) for l in b["bracket_label"]])
            p = p / p.sum()
            for lab, pm, pk, won in zip(b["bracket_label"], p, mkt, b["won"]):
                rows.append((r.kind, int(r.lead), var, r.date, lab, pm, pk, int(won), r.mu, sig, r.truth))
    return pd.DataFrame(rows, columns=["kind", "lead", "variant", "date", "label", "p_model", "p_mkt",
                                       "won", "mu", "sigma", "truth"])


def day_scores(d):
    """Per market-day: Brier (sum over brackets) and log loss of the winner."""
    def f(g):
        w = g["won"].to_numpy()
        pm, pk = g["p_model"].to_numpy(), g["p_mkt"].to_numpy()
        mix = (pm + pk) / 2
        ll = lambda p: -np.log(np.clip(p[w == 1], 1e-3, 1)).sum()
        return pd.Series({"brier_model": ((pm - w) ** 2).sum(), "brier_mkt": ((pk - w) ** 2).sum(),
                          "brier_mix": ((mix - w) ** 2).sum(), "ll_model": ll(pm), "ll_mkt": ll(pk)})
    return d.groupby("date").apply(f, include_groups=False)


def paper_trades(d):
    """1 share per signal. YES if model > ask+fee+EDGE, NO if (1-model) > (1-bid)+fee+EDGE."""
    yes_px = d["p_mkt"] + HALF_SPREAD
    no_px = (1 - d["p_mkt"]) + HALF_SPREAD
    fee_y, fee_n = FEE_RATE * yes_px * (1 - yes_px), FEE_RATE * no_px * (1 - no_px)
    buy_yes = (d["p_model"] - yes_px - fee_y) > EDGE
    buy_no = ((1 - d["p_model"]) - no_px - fee_n) > EDGE
    pnl = np.where(buy_yes, d["won"] - yes_px - fee_y, 0.0) + np.where(buy_no, (1 - d["won"]) - no_px - fee_n, 0.0)
    t = d.assign(pnl=pnl, n=buy_yes.astype(int) + buy_no.astype(int))
    per_day = t.groupby("date")[["pnl", "n"]].sum()
    return per_day


def summarize(rows):
    print("\nScores per market-day (lower = better). 'diff' = model - market; t = diff / its standard error.")
    print("'mix' = 50/50 average of model and market: if it beats the market, the model adds information.")
    hdr = f"{'kind':5} {'N':>2} {'sigma':7} {'days':>4} | {'Brier mod':>9} {'Brier mkt':>9} {'diff':>7} {'t':>5} | {'mix-mkt':>7} {'t':>5} | {'LL mod':>6} {'LL mkt':>6} | {'trades':>6} {'PnL':>6} {'/trade':>6} {'t':>5}"
    print(hdr); print("-" * len(hdr))
    for (kind, n, var), g in rows.groupby(["kind", "lead", "variant"]):
        s = day_scores(g)
        k = len(s)
        dif = s["brier_model"] - s["brier_mkt"]
        mix = s["brier_mix"] - s["brier_mkt"]
        tt = lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if x.std() > 0 else 0
        pt = paper_trades(g)
        ntr, pnl = int(pt["n"].sum()), pt["pnl"].sum()
        note = "  (unfair: dawn minimum already seen)" if (kind == "low" and n == 1) else ""
        print(f"{kind:5} {n:>2} {var:7} {k:>4} | {s['brier_model'].mean():9.3f} {s['brier_mkt'].mean():9.3f} "
              f"{dif.mean():+7.3f} {tt(dif):5.1f} | {mix.mean():+7.3f} {tt(mix):5.1f} | "
              f"{s['ll_model'].mean():6.2f} {s['ll_mkt'].mean():6.2f} | {ntr:>6} {pnl:+6.2f} "
              f"{(pnl / ntr if ntr else 0):+6.3f} {tt(pt['pnl']):5.1f}{note}")


def last_days(rows, n_days=3):
    days = sorted(rows["date"].unique())[-n_days:]
    print(f"\nLast {n_days} resolved days (sigma = spread variant): favourite bracket and probability given to the winner")
    d = rows[(rows["variant"] == "spread") & rows["date"].isin(days)]
    for (date, kind, n), g in d.groupby(["date", "kind", "lead"]):
        win = g[g["won"] == 1].iloc[0]
        fm, fk = g.loc[g["p_model"].idxmax()], g.loc[g["p_mkt"].idxmax()]
        print(f"  {date} {kind:4} N={n}: actual {win['truth']:.0f}°C | model mu {win['mu']:5.1f} sigma {win['sigma']:.2f}, "
              f"fav {fm['label']:>12} ({fm['p_model']:.2f}), P(winner) {win['p_model']:.2f} | "
              f"market fav {fk['label']:>12} ({fk['p_mkt']:.2f}), P(winner) {win['p_mkt']:.2f}")


def main():
    ens = pd.read_parquet(os.path.join(DATA, "ensemble", "ens_means.parquet"))
    station = pd.read_csv(os.path.join(DATA, "station.csv"))
    markets = pd.read_csv(os.path.join(DATA, "markets.csv"), dtype={"yes_token": str})
    markets = markets[markets["won"].notna()]
    sup = daily_super(ens)
    cal = calibrate(sup, station)
    print(f"super-model days: {cal.groupby(['kind', 'lead']).size().to_dict()}")
    print("forecast error of mu (station - mu), all days: "
          + ", ".join(f"{k} N={n}: bias {(g.truth - g.mu).mean():+.2f} rmse {np.sqrt(((g.truth - g.mu) ** 2).mean()):.2f}"
                      for (k, n), g in cal.dropna(subset=["truth"]).groupby(["kind", "lead"])))
    prices = load_prices(set(markets["yes_token"]))
    rows = build_rows(cal, markets, prices)
    os.makedirs(os.path.join(DATA, "analysis"), exist_ok=True)
    rows.to_parquet(os.path.join(DATA, "analysis", "ensemble_rows.parquet"), index=False)
    print(f"period: {rows['date'].min()} .. {rows['date'].max()}")
    summarize(rows)
    last_days(rows)


if __name__ == "__main__":
    main()
