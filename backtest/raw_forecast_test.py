"""
backtest/raw_forecast_test.py
-----------------------------
Lag test WITHOUT our blend model: does the market re-rate toward what a new
raw model run says, gradually over hours?

For each release of a model run (t0 = init + delay, config.py) and each open
HIGH market (before the target day starts):
  news      = new run's forecast of the day's max  minus  that model's previous
              run's forecast of the same day (°C)
  M(t)      = the market's own expected max temperature at time t, read from
              the bracket prices: sum(price_i * temperature_i) / sum(price_i)
              (only when bracket prices add up to 0.9-1.1, i.e. real
              probabilities)
  dM(h)     = M(t0 + h) - M(t0)                     -> market reaction
  surprise  = actual station max - M(t0)            -> what was still to learn

Two regressions, standard errors clustered by market-day:
  dM(h)    = a + beta_h * news       how many °C the market moves per °C of news
  surprise = a + gamma  * news       how many °C of the news were really right
                                     (and not yet in the price at t0)
If the market lags: beta_h starts small and climbs toward gamma over hours.
If it is efficient: beta_h is ~flat and gamma ~0 (nothing left to learn).
share_h = beta_h / (beta_h(at 0) + gamma) is the share of the useful news
priced in by h.

Run:  python3 -m backtest.raw_forecast_test
"""

import glob
import os
import re

import numpy as np
import pandas as pd

from backtest.config import MODELS
from backtest.event_study import load_prices, clustered_ols
from backtest.model_probs import daily_extremes, Latest, local_midnight_ts
from backtest.timeconv import to_unix_seconds

DATA = "backtest_data"
OUT = os.path.join(DATA, "analysis")
HORIZONS_MIN = [15, 30, 60, 120, 240, 480]
MAX_STALENESS_S = 600


def bracket_value(label):
    """Temperature a bracket stands for: '3°C' -> 3; tails one degree beyond
    their edge ('-2°C or below' -> -3, '8°C or higher' -> 9)."""
    v = float(re.findall(r"[-+]?\d+(?:\.\d+)?", label.replace("−", "-"))[0])
    low = label.lower()
    if any(w in low for w in ("below", "lower", "less", "under")):
        return v - 1
    if any(w in low for w in ("higher", "above", "more", "greater")):
        return v + 1
    return v


def market_mean(series, ts):
    """series: list of (values array, t array, p array). Market expected temp at ts,
    or NaN if a price is stale/missing or the prices aren't coherent."""
    tot = num = 0.0
    for val, t, p in series:
        i = np.searchsorted(t, ts, side="right") - 1
        if i < 0 or ts - t[i] > MAX_STALENESS_S:
            return np.nan
        tot += p[i]
        num += p[i] * val
    if not 0.9 <= tot <= 1.1:
        return np.nan
    return num / tot


def main():
    markets = pd.read_csv(os.path.join(DATA, "markets.csv"), dtype={"yes_token": str})
    markets = markets[markets["kind"] == "high"]
    station = pd.read_csv(os.path.join(DATA, "station.csv")).set_index("date")["high"]
    fc = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(DATA, "forecasts", "runs_*.parquet")))])
    ext = daily_extremes(fc[fc["model"].isin(MODELS)])
    latest = Latest(ext)
    prices = load_prices(set(markets["yes_token"]))

    ev = markets.groupby(["target_date", "event_slug"]).agg(start=("market_start", "first")).reset_index()
    ev["open_ts"] = to_unix_seconds(ev["start"])
    ev["day_start"] = ev["target_date"].map(local_midnight_ts)
    series = {}
    for slug, g in markets.groupby("event_slug"):
        s = [(bracket_value(r.bracket_label),) + prices[r.yes_token] for r in g.itertuples() if r.yes_token in prices]
        if len(s) == len(g):
            series[slug] = s

    rows = []
    for m in MODELS:
        av, inits = latest.avail[m]
        for t0, init in zip(av, inits):
            t0, init = int(t0), int(init)
            for e in ev[(ev["open_ts"] <= t0) & (ev["day_start"] > t0)].itertuples(index=False):
                if e.event_slug not in series or (m, init, e.target_date) not in latest.val:
                    continue
                prev = latest.runs_before(m, t0, e.target_date, strict=True)
                if not prev or pd.isna(station.get(e.target_date)):
                    continue
                news = (latest.value(m, init, e.target_date, "high")
                        - latest.value(m, prev[0], e.target_date, "high"))
                s = series[e.event_slug]
                m0 = market_mean(s, t0)
                if np.isnan(m0):
                    continue
                row = {"model": m, "t0": t0, "event_slug": e.event_slug, "news": news,
                       "M0": m0, "pre": m0 - market_mean(s, t0 - 3600),
                       "surprise": station[e.target_date] - m0,
                       "horizon_h": (e.day_start - t0) / 3600}
                for mins in HORIZONS_MIN:
                    row[f"d_{mins}m"] = (market_mean(s, t0 + 60 * mins) - m0
                                         if mins <= row["horizon_h"] * 60 else np.nan)
                rows.append(row)
    d = pd.DataFrame(rows)
    d.to_parquet(os.path.join(OUT, "raw_forecast_events.parquet"), index=False)

    def fit(sub, ycol):
        x = sub[[ycol, "news", "event_slug"]].dropna()
        if x["event_slug"].nunique() < 10:
            return np.nan, np.nan, len(x)
        b, se, _ = clustered_ols(x[ycol], x["news"], pd.factorize(x["event_slug"])[0])
        return b, se, len(x)

    out = []
    for label, sub in [("all models", d)] + [(m, d[d["model"] == m]) for m in MODELS]:
        g, gse, n = fit(sub, "surprise")
        row = {"sample": label, "n_events": n, "market_days": sub["event_slug"].nunique(),
               "gamma (still right at t0)": g, "gamma_se": gse}
        pre, pse, _ = fit(sub, "pre")
        row["beta pre_1h"] = pre
        for mins in HORIZONS_MIN:
            b, se, _ = fit(sub, f"d_{mins}m")
            row[f"beta {mins}m"], row[f"se {mins}m"] = b, se
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(OUT, "raw_forecast_test.csv"), index=False)

    print(f"events: {len(d)} (release x open market), market-days: {d['event_slug'].nunique()}, "
          f"median |news| {d['news'].abs().median():.2f} °C, share |news| >= 0.5 °C: {(d['news'].abs() >= 0.5).mean():.0%}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        cols = ["sample", "n_events", "gamma (still right at t0)", "gamma_se", "beta pre_1h"] + \
               [f"beta {m}m" for m in HORIZONS_MIN]
        print(res[cols].round(3).to_string(index=False))
        print("\nstandard errors of beta:")
        print(res[["sample"] + [f"se {m}m" for m in HORIZONS_MIN]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
