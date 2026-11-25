"""
backtest/config.py -- editable settings for the historical lag backtest.

AVAILABILITY_DELAY_HOURS (C4): historical "release" times (t0) can't be
observed, so t0 = run init time + a typical delay per model. Starting values
are the delays measured live on 25 Nov 2026 from Open-Meteo's status files
(init -> available on Open-Meteo). For seamless blends the SLOWEST component
used over Munich is taken, since the full series exists only once it's in:
  ECMWF IFS 0.25     00z run -> available 08:04   => 8.1 h
  ICON seamless      D2 1h22, EU 2h53, global 3h33 => 3.6 h
  GFS seamless       gfs013 5h17, gfs025 5h46     => 5.8 h
  Meteo-France seam. AROME-HD 5h02, ARPEGE-EU 4h21, ARPEGE world 5h10 => 5.2 h
One sample each: refine with the live om-meta logging before trusting
sub-hour timing. Only horizons >= 1 h are interpreted in the backtest.
"""

# Models in the historical blend. GEM is in the live blend but the Single
# Runs API returns no GEM data for 2026, so the backtest uses these four.
MODELS = ["ecmwf_ifs025", "icon_seamless", "gfs_seamless", "meteofrance_seamless"]

AVAILABILITY_DELAY_HOURS = {
    "ecmwf_ifs025": 8.1,
    "icon_seamless": 3.6,
    "gfs_seamless": 5.8,
    "meteofrance_seamless": 5.2,
}

RUN_HOURS_UTC = [0, 6, 12, 18]      # runs used for every model
FIRST_RUN = "2026-04-02T00:00"      # Single Runs archive start for non-ECMWF models

BLEND_WINDOW_DAYS = 30              # same as live
# Sigma per horizon (days) and kind: station-based values measured 25 Nov
SIGMA = {
    "high": [(0, 0.58), (1, 1.15), (2, 1.35), (3, 1.50)],
    "low":  [(0, 0.96), (1, 1.53), (2, 1.88), (3, 2.06)],
}

LAT, LON = 48.3537, 11.786          # EDDM (same as the live bot)
