"""
backtest/model_probs.py  (Part C4 + C5)
---------------------------------------
Rebuilds, for any past moment, what our model would have said -- using only
information available at that moment (no look-ahead):

  * a model run counts as available at  t0 = init + AVAILABILITY_DELAY_HOURS[model]
    (config.py; historical release times can't be observed)
  * blend weights for Munich day F are fitted on the 30 days ending 2 days
    before F, like the live bot (blendfit.fit_blend, same method). Training
    pairs: each model's 00 UTC run of day d (its forecast of day d) vs the
    station value of day d -- the same short-lead data the live bot's
    calibration uses.
  * daily max/min over the full local day (daymax.py), sigma per horizon and
    kind from config.SIGMA, bracket probabilities with the live bot's
    calculate_bracket_probability().

Only times BEFORE the target day starts (Munich midnight) are used here, so
no same-day observation floor/ceiling is needed.

Outputs (backtest_data/analysis/):
  model_events.parquet -- one row per (model release, open market, bracket):
      model, init, t0, kind, target_date, event_slug, yes_token, bracket_label,
      p_before, p_after, horizon_h, won
  level_grid.parquet   -- hourly grid of model probability per bracket while
      the market is open (for the Brier / log-loss level test, D3):
      t, kind, target_date, event_slug, yes_token, bracket_label, p_model, horizon_h, won
"""

import glob
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtest.config import MODELS, AVAILABILITY_DELAY_HOURS, SIGMA, BLEND_WINDOW_DAYS
from blendfit import fit_blend
from munich_bot import parse_label_to_bounds, calculate_bracket_probability

DATA = "backtest_data"
OUT = os.path.join(DATA, "analysis")
BERLIN = ZoneInfo("Europe/Berlin")
MIN_VALID_HOURS = 20


def local_midnight_ts(day_str):
    d = datetime.fromisoformat(day_str).replace(tzinfo=BERLIN)
    return int(d.timestamp())


def daily_extremes(fc):
    """fc: model, init, t, temp -> DataFrame model, init, date, high, low (full local day,
    only days with >= MIN_VALID_HOURS hourly values)."""
    fc = fc.copy()
    fc["date"] = pd.to_datetime(fc["t"], unit="s", utc=True).dt.tz_convert(BERLIN).dt.strftime("%Y-%m-%d")
    g = fc.groupby(["model", "init", "date"])["temp"]
    out = g.agg(high="max", low="min", n="count").reset_index()
    return out[out["n"] >= MIN_VALID_HOURS].drop(columns="n")


def sigma_at(horizon_h, kind):
    pts = SIGMA[kind]
    days = max(0.0, horizon_h / 24.0)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return float(np.interp(days, xs, ys))


class Blend:
    """Frozen-per-day blend weights fitted without look-ahead."""

    def __init__(self, ext, station):
        self.station = station.set_index("date")
        # training forecasts: the 00 UTC run of day d, forecasting day d
        e = ext.copy()
        e["init_date"] = pd.to_datetime(e["init"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
        e["init_hour"] = pd.to_datetime(e["init"], unit="s", utc=True).dt.hour
        train = e[(e["init_hour"] == 0) & (e["init_date"] == e["date"])]
        self.train = {k: train.pivot_table(index="date", columns="model", values=k) for k in ("high", "low")}
        self.cache = {}

    def weights(self, fit_day, kind):
        key = (fit_day, kind)
        if key not in self.cache:
            end = datetime.fromisoformat(fit_day) - timedelta(days=2)
            start = end - timedelta(days=BLEND_WINDOW_DAYS)
            X = self.train[kind]
            X = X[(X.index >= start.strftime("%Y-%m-%d")) & (X.index <= end.strftime("%Y-%m-%d"))]
            X = X.reindex(columns=MODELS).dropna()
            y = self.station[kind].reindex(X.index)
            ok = y.notna()
            X, y = X[ok], y[ok]
            if len(y) < 15:
                self.cache[key] = None            # not enough history yet
            else:
                w, b = fit_blend(X.values, y.values)
                self.cache[key] = (np.round(w, 6), round(b, 6), len(y))
        return self.cache[key]


class Latest:
    """For each model: which run is the latest AVAILABLE one at time t, and its
    forecast of (date, kind)."""

    def __init__(self, ext):
        self.val = {}
        for r in ext.itertuples(index=False):
            self.val[(r.model, r.init, r.date)] = (r.high, r.low)
        self.avail = {}
        for m in MODELS:
            inits = sorted(ext.loc[ext["model"] == m, "init"].unique())
            self.avail[m] = (np.array([i + AVAILABILITY_DELAY_HOURS[m] * 3600 for i in inits]), np.array(inits))

    def runs_before(self, m, t, date, n=1, strict=False):
        """Inits of the last n runs of model m available at time t (avail <= t, or < t
        if strict) that have a full-day value for `date`, newest first."""
        av, inits = self.avail[m]
        k = np.searchsorted(av, t, side="left" if strict else "right")
        out = []
        for i in inits[:k][::-1]:
            if (m, int(i), date) in self.val:
                out.append(int(i))
                if len(out) == n:
                    break
        return out

    def value(self, m, init, date, kind):
        hi, lo = self.val[(m, init, date)]
        return hi if kind == "high" else lo


def bracket_probs(mean, sigma, labels):
    return {lab: float(calculate_bracket_probability(mean, sigma, parse_label_to_bounds(lab))) for lab in labels}


def main():
    os.makedirs(OUT, exist_ok=True)
    markets = pd.read_csv(os.path.join(DATA, "markets.csv"), dtype={"yes_token": str})
    station = pd.read_csv(os.path.join(DATA, "station.csv"))
    fc = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(DATA, "forecasts", "runs_*.parquet")))])
    fc = fc[fc["model"].isin(MODELS)]
    ext = daily_extremes(fc)
    blend, latest = Blend(ext, station), Latest(ext)
    print(f"forecast runs: {ext.groupby('model')['init'].nunique().to_dict()}")

    ev = (markets.groupby(["kind", "target_date", "event_slug"])
          .agg(start=("market_start", "first")).reset_index())
    ev["open_ts"] = pd.to_datetime(ev["start"], utc=True).astype("int64") // 10**9
    ev["day_start"] = ev["target_date"].map(local_midnight_ts)
    brackets = markets.groupby("event_slug")[["bracket_label", "yes_token", "won"]].apply(
        lambda d: d.to_dict("records")).to_dict()

    def blend_mean(t, kind, date, override=None):
        """Model mean at time t from the latest available run of each model.
        override=(model, init) forces one model's run (used for 'after')."""
        fit_day = datetime.fromtimestamp(t, BERLIN).strftime("%Y-%m-%d")
        wb = blend.weights(fit_day, kind)
        if wb is None:
            return None
        w, b, _ = wb
        xs = []
        for m in MODELS:
            if override and override[0] == m:
                init = override[1]
            else:
                r = latest.runs_before(m, t, date)
                if not r:
                    return None
                init = r[0]
            xs.append(latest.value(m, init, date, kind))
        return float(np.dot(w, xs) + b)

    # ---- model events (D1) -------------------------------------------------
    rows = []
    for m in MODELS:
        av, inits = latest.avail[m]
        for t0, init in zip(av, inits):
            t0, init = int(t0), int(init)
            live = ev[(ev["open_ts"] <= t0) & (ev["day_start"] > t0)]
            for e in live.itertuples(index=False):
                if (m, init, e.target_date) not in latest.val:
                    continue
                prev = latest.runs_before(m, t0, e.target_date, strict=True)
                if not prev:
                    continue
                before = blend_mean(t0, e.kind, e.target_date, override=(m, prev[0]))
                after = blend_mean(t0, e.kind, e.target_date, override=(m, init))
                if before is None or after is None:
                    continue
                horizon = (e.day_start - t0) / 3600
                sig = sigma_at(horizon, e.kind)
                bk = brackets[e.event_slug]
                pb = bracket_probs(before, sig, [b["bracket_label"] for b in bk])
                pa = bracket_probs(after, sig, [b["bracket_label"] for b in bk])
                for b in bk:
                    rows.append((m, init, t0, e.kind, e.target_date, e.event_slug, b["yes_token"],
                                 b["bracket_label"], pb[b["bracket_label"]], pa[b["bracket_label"]],
                                 horizon, int(b["won"]), before, after))
    me = pd.DataFrame(rows, columns=["model", "init", "t0", "kind", "target_date", "event_slug", "yes_token",
                                     "bracket_label", "p_before", "p_after", "horizon_h", "won",
                                     "mean_before", "mean_after"])
    me.to_parquet(os.path.join(OUT, "model_events.parquet"), index=False)
    print(f"model_events: {len(me)} rows, {me.groupby(['model', 'init']).ngroups} releases, "
          f"{me['event_slug'].nunique()} markets")

    # ---- hourly level grid (D3) --------------------------------------------
    rows = []
    for e in ev.itertuples(index=False):
        bk = brackets[e.event_slug]
        t = e.open_ts - e.open_ts % 3600 + 3600
        while t < e.day_start:
            mean = blend_mean(t, e.kind, e.target_date)
            if mean is not None:
                horizon = (e.day_start - t) / 3600
                probs = bracket_probs(mean, sigma_at(horizon, e.kind), [b["bracket_label"] for b in bk])
                for b in bk:
                    rows.append((t, e.kind, e.target_date, e.event_slug, b["yes_token"], b["bracket_label"],
                                 probs[b["bracket_label"]], horizon, int(b["won"])))
            t += 3600
    lg = pd.DataFrame(rows, columns=["t", "kind", "target_date", "event_slug", "yes_token", "bracket_label",
                                     "p_model", "horizon_h", "won"])
    lg.to_parquet(os.path.join(OUT, "level_grid.parquet"), index=False)
    print(f"level_grid: {len(lg)} rows over {lg['event_slug'].nunique()} markets")


if __name__ == "__main__":
    main()
