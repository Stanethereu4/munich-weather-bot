"""
backtest/pmdata_targets.py
--------------------------
Which pmdata.dev order-book files would the backtest need? (No download is made here.)

Runs the lead-lag signal on the long history (backtest_data/kalshi_hist, May 2026 on) and lists,
for every signal, the Polymarket bracket we would buy and the UTC day of the signal.
pmdata.dev sells one file per bracket per UTC day, so each distinct (bracket, UTC day) = 1 download.

Output: backtest_data/kalshi_hist/pmdata_targets.csv, plus counts per month, to decide how far
600 free downloads go (free plan: only data older than about 3 weeks).

Run:  python3 -m backtest.pmdata_targets
"""
import os

os.environ.setdefault("KALSHI_DIR", os.path.join("backtest_data", "kalshi_hist"))

import pandas as pd  # noqa: E402

from backtest.kalshi_test import D, load  # noqa: E402
from backtest.leadlag import features, series  # noqa: E402
from backtest.leadlag_real import dedupe  # noqa: E402

FREE_DOWNLOADS = 600
FREE_LAG_DAYS = 21


def main():
    km, pm = load()
    km = km[km.same_airport]                       # the 7 cities with the same airport on both platforms
    s = features(series(km, pm))
    slug = dict(zip(pm.yes_token.astype(str), pm.market_slug))
    cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=FREE_LAG_DAYS)).strftime("%Y-%m-%d")
    print(f"city-days with both platforms: {s.groupby(['city', 'date']).ngroups}, "
          f"from {s.date.min()} to {s.date.max()}; free-plan cutoff (UTC day) {cutoff}\n")

    rows = []
    for thr in (0.10, 0.05):
        sig = dedupe(s[(s.dK.abs() >= thr) & (s.dP.abs() < thr / 2)])
        for r in sig.itertuples():
            side = "nb_up" if r.dK > 0 else "nb_dn"
            tok, won = getattr(r, f"{side}_tok"), getattr(r, f"{side}_won")
            if tok is None or pd.isna(won):
                continue
            utc_day = pd.Timestamp(r.t, unit="s").strftime("%Y-%m-%d")
            rows.append(dict(thr=thr, city=r.city, date=r.date, t=r.t, utc_day=utc_day, token=str(tok),
                             market_slug=slug.get(str(tok))))
    o = pd.DataFrame(rows)
    o.to_csv(os.path.join(D, "pmdata_targets.csv"), index=False)

    for thr, g in o.groupby("thr", sort=False):
        files = g.drop_duplicates(["market_slug", "utc_day"]).sort_values("utc_day")
        files = files.assign(month=files.utc_day.str[:7], free=files.utc_day <= cutoff)
        days = g.assign(month=g.date.str[:7]).drop_duplicates(["city", "date"]).groupby("month").size()
        per = files.groupby("month").agg(files=("market_slug", "size"))
        per["city_days"] = days
        per["cumulative_files"] = per.files.cumsum()
        print(f"Kalshi move >= {thr*100:.0f}c: {len(g)} signals, {len(files)} files (bracket x UTC day), "
              f"{g[['city','date']].drop_duplicates().shape[0]} city-days; missing slug: {files.market_slug.isna().sum()}")
        print(per.to_string())
        free = files[files.free]
        covered = free.head(FREE_DOWNLOADS - 1)       # 1 download already used by the probe
        print(f"   files old enough for the free plan: {len(free)}; 599 left would cover up to UTC day "
              f"{covered.utc_day.max() if len(covered) else '-'}\n")


if __name__ == "__main__":
    main()
