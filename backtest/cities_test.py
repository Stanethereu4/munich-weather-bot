"""
backtest/cities_test.py
-----------------------
Same ensemble super-model test as backtest/ensemble_test.py, for 10 cities
(Munich = control), highs and lows. Data from backtest/cities.py.

Differences from the Munich-only test:
  * Truth = Polymarket's winning bracket (no station download): its middle,
    or for "X or below / X or higher", half a bracket beyond X. Error <= 0.5°C
    for 1°C brackets, <= ~0.6°C for 2°F brackets. Used only to fit bias/sigma.
  * Local time per city. Forecast issued N days earlier is safely published by
    10:00 local on day D-(N-1) (same rule as Munich's 09:00 UTC), so prices are read then.
      N=2 -> 10:00 local the day before, N=3 -> 10:00 local two days before.
    (N=1 is skipped: on the day itself the market already sees observations.)
  * °F brackets ("44-45°F") are converted to °C bounds.

Run:  python3 -m backtest.cities_test      (needs backtest_data/cities, no network)
"""
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest.cities import CITIES, OUT

TRAIN_DAYS, MIN_TRAIN = 30, 14
MAX_STALENESS_S = 2 * 3600
FEE_RATE, HALF_SPREAD, EDGE = 0.01, 0.01, 0.05
LEADS = (2, 3)


def bounds_c(label):
    """'44-45°F' -> (6.39, 7.5) °C ; '3°C or below' -> (-inf, 3.5) ; '-2°C' -> (-2.5, -1.5)."""
    f = "°F" in label
    clean = label.replace("°C", "").replace("°F", "").replace("°", "").replace("−", "-").strip()
    m = re.match(r"^(-?\d+(?:\.\d+)?)(?:\s*-\s*(-?\d+(?:\.\d+)?))?", clean)
    a = float(m.group(1)); b = float(m.group(2)) if m.group(2) else a
    low = label.lower()
    if any(w in low for w in ("below", "lower", "less", "under")):
        lo, hi = -np.inf, b + 0.5
    elif any(w in low for w in ("higher", "above", "more", "greater")):
        lo, hi = a - 0.5, np.inf
    else:
        lo, hi = a - 0.5, b + 0.5
    conv = (lambda x: (x - 32) * 5 / 9) if f else (lambda x: x)
    return conv(lo), conv(hi)


def truth_from_winner(lo, hi, width):
    if np.isfinite(lo) and np.isfinite(hi):
        return (lo + hi) / 2
    return hi - width / 2 if np.isfinite(hi) else lo + width / 2


def daily_super(ens, tz):
    e = ens.copy()
    e["date"] = pd.to_datetime(e["t"], unit="s", utc=True).dt.tz_convert(tz).dt.strftime("%Y-%m-%d")
    e = e[e.groupby(["model", "lead_days", "date"])["mean"].transform("size") >= 20]
    rows = []
    for kind, fn in (("high", "idxmax"), ("low", "idxmin")):
        idx = getattr(e.groupby(["model", "lead_days", "date"])["mean"], fn)()
        d = e.loc[idx, ["model", "lead_days", "date", "mean", "spread"]].assign(kind=kind)
        rows.append(d)
    per = pd.concat(rows)
    n = per["model"].nunique()
    agg = per.groupby(["lead_days", "date", "kind"]).agg(
        x=("mean", "mean"), between=("mean", "var"), s2=("spread", lambda s: np.mean(s ** 2)), k=("mean", "size"))
    agg = agg[agg["k"] == n].reset_index().rename(columns={"lead_days": "lead"})
    agg["u"] = np.sqrt(agg["s2"] + agg["between"])
    return agg


def run_city(city, mk, prices):
    icao, tzname, _, _ = CITIES[city]
    tz = ZoneInfo(tzname)
    ens = pd.read_parquet(os.path.join(OUT, f"ens_{city}.parquet"))
    sup = daily_super(ens, tz)
    out = []
    for kind, g in mk.groupby("kind"):
        days = {}
        for d, b in g.groupby("target_date"):
            bd = [bounds_c(l) for l in b["bracket_label"]]
            lo = np.array([x[0] for x in bd]); hi = np.array([x[1] for x in bd])
            fin = np.isfinite(lo) & np.isfinite(hi)
            width = float(np.median((hi - lo)[fin])) if fin.any() else 1.0
            w = int(np.argmax(b["won"].to_numpy()))
            days[d] = (lo, hi, b["won"].to_numpy(int), b["yes_token"].to_numpy(), truth_from_winner(lo[w], hi[w], width))
        truth = pd.Series({d: v[4] for d, v in days.items()})
        for n in LEADS:
            s = sup[(sup["kind"] == kind) & (sup["lead"] == n)].set_index("date").sort_index()
            s["truth"] = truth.reindex(s.index)
            for d in s.index:
                if d not in days:
                    continue
                last = (datetime.fromisoformat(d) - timedelta(days=n)).strftime("%Y-%m-%d")
                tr = s[(s.index <= last) & s["truth"].notna()].tail(TRAIN_DAYS)
                if len(tr) < MIN_TRAIN:
                    continue
                bias = float((tr["truth"] - tr["x"]).mean())
                rmse = max(float(np.sqrt(((tr["truth"] - tr["x"] - bias) ** 2).mean())), 0.5)
                sig = float(np.clip(rmse * s.at[d, "u"] / tr["u"].mean(), 0.5 * rmse, 2 * rmse))
                mu = s.at[d, "x"] + bias
                lo, hi, won, toks, _ = days[d]
                ts = int((datetime.fromisoformat(d) - timedelta(days=n - 1)).replace(hour=10, tzinfo=tz).timestamp())
                mkt = np.array([price_at(prices, t, ts) for t in toks])
                if np.isnan(mkt).any() or not 0.9 <= mkt.sum() <= 1.1:
                    continue
                pm = norm.cdf((hi - mu) / sig) - norm.cdf((lo - mu) / sig)
                pm, mkt = pm / pm.sum(), mkt / mkt.sum()
                out.append((city, kind, n, d, pm, mkt, won))
    return out


def price_at(prices, tok, ts):
    if tok not in prices:
        return np.nan
    t, p = prices[tok]
    i = np.searchsorted(t, ts, side="right") - 1
    if i < 0 or ts - t[i] > MAX_STALENESS_S:
        return np.nan
    return p[i]


def score(rows):
    bm = np.array([((pm - w) ** 2).sum() for _, _, _, _, pm, mk, w in rows])
    bk = np.array([((mk - w) ** 2).sum() for _, _, _, _, pm, mk, w in rows])
    bx = np.array([(((pm + mk) / 2 - w) ** 2).sum() for _, _, _, _, pm, mk, w in rows])
    pnl, ntr = [], 0
    for _, _, _, _, pm, mk, w in rows:
        day = 0.0
        for p, m, won in zip(pm, mk, w):
            y, no = m + HALF_SPREAD, 1 - m + HALF_SPREAD
            if p - y - FEE_RATE * y * (1 - y) > EDGE:
                day += won - y - FEE_RATE * y * (1 - y); ntr += 1
            if (1 - p) - no - FEE_RATE * no * (1 - no) > EDGE:
                day += (1 - won) - no - FEE_RATE * no * (1 - no); ntr += 1
        pnl.append(day)
    pnl = np.array(pnl)
    t = lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0
    return dict(days=len(rows), brier_model=bm.mean(), brier_mkt=bk.mean(), diff=(bm - bk).mean(), t_diff=t(bm - bk),
                mix_minus_mkt=(bx - bk).mean(), t_mix=t(bx - bk), trades=ntr, pnl=pnl.sum(),
                per_trade=pnl.sum() / ntr if ntr else 0.0, t_pnl=t(pnl))


def main():
    mk = pd.read_csv(os.path.join(OUT, "markets.csv"), dtype={"yes_token": str})
    vol = mk.groupby(["city", "kind"]).apply(lambda g: g["volume"].sum() / g["target_date"].nunique(), include_groups=False)
    allrows = []
    for city, g in mk.groupby("city"):
        p = pd.read_parquet(os.path.join(OUT, "prices", f"{city}.parquet"))
        prices = {tok: (x["t"].to_numpy(), x["p"].to_numpy(float)) for tok, x in p.sort_values("t").groupby("yes_token")}
        allrows += run_city(city, g, prices)
    res = []
    for (city, kind, n), grp in pd.DataFrame(allrows, columns=["city", "kind", "lead", "date", "pm", "mk", "won"]).groupby(
            ["city", "kind", "lead"]):
        rows = list(grp.itertuples(index=False, name=None))
        res.append(dict(city=city, kind=kind, lead=n, vol_day=vol[(city, kind)], **score(rows)))
    r = pd.DataFrame(res).sort_values(["kind", "lead", "vol_day"])
    pd.set_option("display.width", 250)
    print("Brier per market-day (lower = better). diff = model - market (negative = model better).")
    print("mix = 50/50 model+market vs market alone (negative = model adds information). PnL: 1 share per signal.\n")
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    r.to_csv(os.path.join(OUT, "results.csv"), index=False)


if __name__ == "__main__":
    main()
