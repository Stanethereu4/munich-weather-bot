"""
backtest/event_study.py  (Part D1, D2, D3, + D5 time split)
-----------------------------------------------------------
Works on the historical backtest files (and later on live data in the same
format).

D1  For each model release (event) and each open bracket:
        dp_model     = p_after - p_before        (our model's change at t0)
        dp_mkt(h)    = price(t0 + h) - price(t0)  for h = 15m, 30m, 1h, 2h, 4h, 8h
        pre          = price(t0) - price(t0 - 1h) (did the market move BEFORE t0?)
    and fit  dp_mkt(h) = alpha + beta_h * dp_model  for each h.
    beta_h ~ 1: the market fully follows our model's change by h;
    beta_h ~ 0: it doesn't follow (at least not because of this release).
D2  Standard errors clustered by market-day (all brackets and releases of one
    market are not independent). Counts of events and market-days reported.
D3  Level test on an hourly grid: Brier score and log loss of the model vs the
    market price against the actual winner, by hours before the target day.
D5  Same beta table on the first 60% and the last 40% of the period.

Only moves that end before the target day starts are used (after midnight,
prices react to live temperature reports, which isn't what we test here).

Run:  python3 -m backtest.event_study [high|low]     (default high)
Outputs in backtest_data/analysis/: beta_<kind>.csv, level_<kind>.csv, beta_<kind>.png
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = "backtest_data"
OUT = os.path.join(DATA, "analysis")
HORIZONS_MIN = [15, 30, 60, 120, 240, 480]
MAX_STALENESS_S = 600       # a minute price older than 10 min counts as missing


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


def clustered_ols(y, x, groups):
    X = sm.add_constant(np.asarray(x, dtype=float))
    res = sm.OLS(np.asarray(y, dtype=float), X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return res.params[1], res.bse[1], res.params[0]


def beta_table(ev, label):
    rows = []
    for h in ["pre_1h"] + [f"{m}m" for m in HORIZONS_MIN]:
        col = "pre" if h == "pre_1h" else f"d_{h}"
        d = ev[[col, "dp_model", "event_slug"]].dropna()
        if len(d) < 30 or d["event_slug"].nunique() < 10:
            continue
        b, se, a = clustered_ols(d[col], d["dp_model"], pd.factorize(d["event_slug"])[0])
        rows.append({"sample": label, "horizon": h, "beta": b, "se": se, "ci_low": b - 1.96 * se,
                     "ci_high": b + 1.96 * se, "alpha": a, "n_rows": len(d),
                     "n_market_days": d["event_slug"].nunique()})
    return pd.DataFrame(rows)


def level_table(lg):
    eps = 0.01
    lg = lg.dropna(subset=["p_mkt"]).copy()
    lg["bucket"] = pd.cut(lg["horizon_h"], [0, 12, 24, 36, 48, 72, 999],
                          labels=["0-12h", "12-24h", "24-36h", "36-48h", "48-72h", "72h+"], right=False)
    for src in ("model", "mkt"):
        p = lg[f"p_{src}"].clip(eps, 1 - eps)
        lg[f"brier_{src}"] = (lg[f"p_{src}"] - lg["won"]) ** 2
        lg[f"logloss_{src}"] = -(lg["won"] * np.log(p) + (1 - lg["won"]) * np.log(1 - p))
    rows = []
    for bucket, g in lg.groupby("bucket", observed=True):
        # per market-day sums -> one number per cluster, then compare across clusters
        per = g.groupby("event_slug")[["brier_model", "brier_mkt", "logloss_model", "logloss_mkt"]].mean()
        diff = per["brier_model"] - per["brier_mkt"]
        rows.append({"bucket": bucket, "n_rows": len(g), "n_market_days": len(per),
                     "brier_model": per["brier_model"].mean(), "brier_mkt": per["brier_mkt"].mean(),
                     "brier_diff": diff.mean(), "brier_diff_se": diff.std(ddof=1) / np.sqrt(len(per)),
                     "logloss_model": per["logloss_model"].mean(), "logloss_mkt": per["logloss_mkt"].mean()})
    return pd.DataFrame(rows)


def main(kind="high"):
    me = pd.read_parquet(os.path.join(OUT, "model_events.parquet"))
    me = me[me["kind"] == kind].copy()
    lg = pd.read_parquet(os.path.join(OUT, "level_grid.parquet"))
    lg = lg[lg["kind"] == kind].copy()
    prices = load_prices(set(me["yes_token"]) | set(lg["yes_token"]))
    day_start = {s: t for s, t in pd.read_csv(os.path.join(DATA, "markets.csv"))
                 .groupby("event_slug")["target_date"].first().items()}

    # ---- D1 ----
    me["dp_model"] = me["p_after"] - me["p_before"]
    me["p0"] = [price_at(prices, tok, t) for tok, t in zip(me["yes_token"], me["t0"])]
    me["pre"] = me["p0"] - [price_at(prices, tok, t - 3600) for tok, t in zip(me["yes_token"], me["t0"])]
    for mins in HORIZONS_MIN:
        later = np.array([price_at(prices, tok, t + 60 * mins) for tok, t in zip(me["yes_token"], me["t0"])])
        d = later - me["p0"].to_numpy()
        d[me["horizon_h"].to_numpy() * 60 < mins] = np.nan     # would cross into the target day
        me[f"d_{mins}m"] = d
    # brackets that can move: not pinned at ~0 or ~1 at t0
    ev = me[(me["p0"] > 0.02) & (me["p0"] < 0.98)].copy()

    order = np.sort(ev["t0"].unique())
    cut = order[int(len(order) * 0.6)] if len(order) else 0
    bt = pd.concat([beta_table(ev, "all"),
                    beta_table(ev[ev["t0"] < cut], "first 60%"),
                    beta_table(ev[ev["t0"] >= cut], "last 40%")]
                   + [beta_table(ev[ev["model"] == m], f"model={m}") for m in sorted(ev["model"].unique())])
    bt.to_csv(os.path.join(OUT, f"beta_{kind}.csv"), index=False)

    # ---- D3 ----
    lg["p_mkt"] = [price_at(prices, tok, t) for tok, t in zip(lg["yes_token"], lg["t"])]
    lt = level_table(lg)
    lt.to_csv(os.path.join(OUT, f"level_{kind}.csv"), index=False)

    # ---- summary ----
    n_rel = me.groupby(["model", "init"]).ngroups
    print(f"\n=== {kind.upper()} markets ===")
    print(f"releases: {n_rel} | event rows used (bracket 2-98% at t0): {len(ev)} | "
          f"market-days: {ev['event_slug'].nunique()} | split at {pd.to_datetime(cut, unit='s'):%Y-%m-%d}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\nD1/D2 beta by horizon (clustered by market-day):")
        print(bt.round(3).to_string(index=False))
        print("\nD3 level test (lower is better):")
        print(lt.round(4).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if bt.empty:
            raise ValueError("no beta estimates (too few market-days)")
        a = bt[(bt["sample"] == "all") & (bt["horizon"] != "pre_1h")]
        x = [int(h[:-1]) / 60 for h in a["horizon"]]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(x, a["beta"], yerr=1.96 * a["se"], fmt="o-", capsize=4)
        ax.axhline(1, ls="--", lw=1, color="grey"); ax.axhline(0, lw=1, color="grey")
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(a["horizon"])
        ax.set_xlabel("time after the model release"); ax.set_ylabel("beta (share of model change followed)")
        ax.set_title(f"Munich {kind}: market response to model releases (95% CI)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"beta_{kind}.png"), dpi=130)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "high")
