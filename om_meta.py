"""
om_meta.py
----------
Which model RUN is behind each forecast?

Open-Meteo publishes a small status file per model:
    https://api.open-meteo.com/data/<model>/static/meta.json
with (checked live 25 Nov 2026, times in unix seconds UTC):
    last_run_initialisation_time  -- the run's start ("init") time, e.g. 06:00 UTC
    last_run_availability_time    -- when that run became available on Open-Meteo
    update_interval_seconds       -- how often the model runs (3 h, 6 h, 12 h)

A run is NOT available at its init time: e.g. on 25 Nov the 06:00 UTC ICON-D2
run appeared at 10:22 UTC; ECMWF IFS 0.25 00:00 UTC appeared at 08:04 UTC.

Our blend uses "seamless" models that stitch several sub-models together
(ICON over Munich = ICON-D2 first, then ICON-EU, then global ICON), so each
blend model maps to the status files of all its components.
"""

from datetime import datetime, timezone

import requests

META_URL = "https://api.open-meteo.com/data/{name}/static/meta.json"

# Blend model -> status-file names of its components (names checked live;
# note GEM global is 'cmc_gem_gdps_15km': the older 'cmc_gem_gdps' file
# stopped updating in May 2026).
MODEL_COMPONENTS = {
    "ecmwf_ifs025": ["ecmwf_ifs025"],
    "icon_seamless": ["dwd_icon_d2", "dwd_icon_eu", "dwd_icon"],
    "gfs_seamless": ["ncep_gfs013", "ncep_gfs025"],
    "gem_global": ["cmc_gem_gdps_15km"],
    "meteofrance_seamless": ["meteofrance_arome_france_hd", "meteofrance_arpege_europe",
                             "meteofrance_arpege_world025"],
}


def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat() if isinstance(ts, (int, float)) else None


def fetch_component_status(name):
    """{'init': ISO UTC, 'available': ISO UTC} for one component, or None."""
    try:
        r = requests.get(META_URL.format(name=name), timeout=15)
        r.raise_for_status()
        j = r.json()
        return {"init": _iso(j.get("last_run_initialisation_time")),
                "available": _iso(j.get("last_run_availability_time"))}
    except Exception as e:
        print(f"[om-meta] {name}: {e}")
        return None


def current_runs(model_keys):
    """
    For each blend model: {'components': {name: {'init', 'available'}},
                           'init_time': latest init among its components}
    Read just before a forecast fetch, so it describes the runs the forecast
    was built from (a run landing in the seconds between the two calls is
    possible but rare).
    """
    cache = {}
    out = {}
    for model in model_keys:
        comps = {}
        for name in MODEL_COMPONENTS.get(model, []):
            if name not in cache:
                cache[name] = fetch_component_status(name)
            if cache[name]:
                comps[name] = cache[name]
        inits = [c["init"] for c in comps.values() if c.get("init")]
        out[model] = {"components": comps, "init_time": max(inits) if inits else None}
    return out
