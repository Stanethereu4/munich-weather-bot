"""
backtest/ensemble.py
--------------------
Downloads the ensemble MEAN and SPREAD temperature forecasts for EDDM, as
they were issued 1, 2 and 3 days earlier (no look-ahead).

Source (checked with the ens-prev probe on 25 Nov 2026):
    GET https://ensemble-api.open-meteo.com/v1/ensemble
        ?latitude&longitude&models=<mean model>&start_date&end_date&timezone=UTC
        &hourly=temperature_2m_previous_day1,temperature_2m_spread_previous_day1,...
    -> {"hourly": {"time": ["2026-08-24T00:00", ...],
                   "temperature_2m_previous_day1": [..], "temperature_2m_spread_previous_day1": [..], ...}}
    History starts 24 Aug 2026 for these four models (complete, no gaps).

"previous_dayN" at hour h = the value from the run initialised about N*24 h
before h. So it was knowable N days ahead (minus the model's publication delay,
handled in ensemble_test.py).

Output: backtest_data/ensemble/ens_means.parquet
    model, t (unix s UTC), lead_days (1..3), mean, spread

Run:  python3 -m backtest.ensemble
"""
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from backtest.config import LAT, LON
from backtest.timeconv import to_unix_seconds

URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
MODELS = ["ecmwf_ifs025_ensemble_mean", "dwd_icon_eps_ensemble_mean_seamless",
          "ncep_gefs_ensemble_mean_seamless", "ecmwf_aifs025_ensemble_mean"]
FIRST_DAY = "2026-08-24"
LEADS = (1, 2, 3)
OUT = os.path.join("backtest_data", "ensemble")


def fetch(model, start, end, tries=5):
    hourly = ",".join(f"temperature_2m_previous_day{n},temperature_2m_spread_previous_day{n}" for n in LEADS)
    params = {"latitude": LAT, "longitude": LON, "models": model, "hourly": hourly,
              "start_date": start, "end_date": end, "timezone": "UTC"}
    for k in range(tries):
        try:
            r = requests.get(URL, params=params, timeout=(10, 120))
            r.raise_for_status()
            return r.json()["hourly"]
        except requests.RequestException as e:
            print(f"  retry {k + 1} {model}: {type(e).__name__}")
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"could not download {model}")


def main():
    os.makedirs(OUT, exist_ok=True)
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parts = []
    for m in MODELS:
        h = fetch(m, FIRST_DAY, end)
        t = to_unix_seconds(pd.Series(h["time"])).to_numpy()
        for n in LEADS:
            parts.append(pd.DataFrame({"model": m, "t": t, "lead_days": n,
                                       "mean": h[f"temperature_2m_previous_day{n}"],
                                       "spread": h[f"temperature_2m_spread_previous_day{n}"]}))
        print(f"{m}: {len(h['time'])} hours")
    df = pd.concat(parts).dropna(subset=["mean"])
    df.to_parquet(os.path.join(OUT, "ens_means.parquet"), index=False)
    print(f"saved {len(df)} rows")


if __name__ == "__main__":
    main()
