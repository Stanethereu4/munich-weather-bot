"""
backtest/forecasts.py  (Part C3)
--------------------------------
Downloads every past run of every model in the historical blend from
Open-Meteo's Single Runs API:

    GET https://single-runs-api.open-meteo.com/v1/forecast
        ?latitude&longitude&run=YYYY-MM-DDTHH:MM&models=a,b,c,d
        &hourly=temperature_2m&timezone=UTC
    -> {"hourly": {"time": [...], "temperature_<model>": [...]}} (several models)
       or "temperature_2m" when one model is requested. Shape checked live
       25 Nov 2026; 168 hours per run, models that don't reach that far
       return nulls.

`run` is the model's INITIALISATION time, not when it became available
(see config.AVAILABILITY_DELAY_HOURS for that).

Output: backtest_data/forecasts/runs_<YYYY-MM>.parquet, one row per
(model, init_time, valid hour):  model, init (unix s UTC), t (unix s UTC), temp
Each month file is saved with whatever was downloaded; a re-run fetches only
the runs still missing (runs the API reports as not archived are listed in
not_archived.csv and skipped).
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from backtest.config import MODELS, RUN_HOURS_UTC, FIRST_RUN, LAT, LON

URL = "https://single-runs-api.open-meteo.com/v1/forecast"
HEADERS = {"User-Agent": "munich-weather-bot backtest (research)"}
DATA = "backtest_data"
OUT = os.path.join(DATA, "forecasts")
MAX_MINUTES = 270


def _get(params, tries=6):
    """GET with retries. Returns (json, None), (None, reason) for an unarchived
    run (HTTP 400), or (None, 'FAILED: ...') after repeated network errors."""
    err = "unknown"
    for k in range(tries):
        try:
            r = requests.get(URL, params=params, headers=HEADERS, timeout=(10, 25))
            if r.status_code == 429:
                time.sleep(30); continue
            if r.status_code == 400:
                return None, r.json().get("reason", "400")
            r.raise_for_status()
            return r.json(), None
        except (requests.RequestException, ValueError) as e:
            err = type(e).__name__
            time.sleep(2 * (k + 1))
    return None, f"FAILED: {err}"


def _frame(h, model, col, init):
    t = pd.to_datetime(h["time"], utc=True).astype("int64") // 10**9
    df = pd.DataFrame({"t": t, "temp": h[col]})
    df["model"], df["init"] = model, int(init.timestamp())
    return df.dropna(subset=["temp"])[["model", "init", "t", "temp"]]


def fetch_run(init):
    """All blend models for one init time -> (DataFrame(model, init, t, temp), note).
    DataFrame is None only if a network failure persisted (retry later)."""
    base = {"latitude": LAT, "longitude": LON, "run": init.strftime("%Y-%m-%dT%H:%M"),
            "hourly": "temperature_2m", "timezone": "UTC"}
    j, err = _get({**base, "models": ",".join(MODELS)})
    if j is not None:
        h = j["hourly"]
        cols = {m: f"temperature_2m_{m}" for m in MODELS}
        if not any(c in h for c in cols.values()):
            raise RuntimeError(f"unexpected multi-model response keys: {list(h.keys())}")
        parts = [_frame(h, m, c, init) for m, c in cols.items() if c in h]
        return pd.concat(parts), None
    if err and err.startswith("FAILED"):
        return None, err
    # One model lacks this run, so the combined call was refused: ask one by one.
    parts, notes = [], []
    for m in MODELS:
        j, e = _get({**base, "models": m})
        if j is not None:
            parts.append(_frame(j["hourly"], m, "temperature_2m", init))
        elif e and e.startswith("FAILED"):
            return None, e
        else:
            notes.append(f"{m}: {e}")
    out = pd.concat(parts) if parts else pd.DataFrame(columns=["model", "init", "t", "temp"])
    return out, "; ".join(notes) or None


def all_inits(last_target):
    start = datetime.fromisoformat(FIRST_RUN).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(last_target).replace(tzinfo=timezone.utc) + timedelta(days=1)
    d, out = start, []
    while d < end:
        if d.hour in RUN_HOURS_UTC:
            out.append(d)
        d += timedelta(hours=6) if RUN_HOURS_UTC == [0, 6, 12, 18] else timedelta(hours=1)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    markets = pd.read_csv(os.path.join(DATA, "markets.csv"))
    inits = all_inits(markets["target_date"].max())
    by_month = {}
    for i in inits:
        by_month.setdefault(i.strftime("%Y-%m"), []).append(i)

    # Runs the API said are not archived (HTTP 400): remembered so they are
    # not asked again on every re-run.
    na_path = os.path.join(OUT, "not_archived.csv")
    not_archived = set(pd.read_csv(na_path)["init"]) if os.path.exists(na_path) else set()

    t_start = time.time()
    for month, month_inits in sorted(by_month.items()):
        path = os.path.join(OUT, f"runs_{month}.parquet")
        have = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(
            columns=["model", "init", "t", "temp"])
        # a run counts as done when all blend models are in, or it was marked not archived
        per_init = have.groupby("init")["model"].nunique() if len(have) else pd.Series(dtype=int)
        todo = [i for i in month_inits
                if per_init.get(int(i.timestamp()), 0) < len(MODELS) and int(i.timestamp()) not in not_archived]
        if not todo:
            continue
        if (time.time() - t_start) / 60 > MAX_MINUTES:
            print("time budget reached -- re-run to continue"); break

        parts, failed = [have], []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fetch_run, i): i for i in todo}
            for f in as_completed(futures):
                df, note = f.result()
                init = futures[f]
                if df is None:                       # network kept failing: retry next time
                    failed.append(init)
                    continue
                if not df.empty:
                    parts.append(df)
                if note and df["model"].nunique() < len(MODELS):
                    not_archived.add(int(init.timestamp()))   # some model simply lacks this run

        out = (pd.concat(parts).astype({"init": "int64", "t": "int64", "temp": "float32"})
               .drop_duplicates(["model", "init", "t"]))
        out.to_parquet(path, compression="zstd", index=False)     # save whatever we have
        pd.DataFrame({"init": sorted(not_archived)}).to_csv(na_path, index=False)
        cover = out.groupby("model")["init"].nunique().to_dict()
        print(f"{month}: {len(month_inits)} runs | fetched now {len(todo) - len(failed)}, "
              f"still failing {len(failed)} (re-run to retry) | runs per model {cover} "
              f"({(time.time() - t_start) / 60:.0f} min)")


if __name__ == "__main__":
    main()
