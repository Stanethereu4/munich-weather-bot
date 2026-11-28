"""
livepaper/config.py
-------------------
Settings for the live PAPER trader (no real orders, ever).

Strategy (pre-registered on 27 Nov 2026, before any live data; see project doc
results/kalshi-leadlag-2026-11-27.md):
  When Kalshi's "high is at least X" (sum of its brackets from X up, mid of bid/ask)
  has moved by >= 10c in the last 15 minutes and Polymarket's matching number
  (average of its thresholds X-1 and X+1) has moved by less than half of that,
  buy YES of the Polymarket bracket just above X (Kalshi up) or just below X
  (Kalshi down), and hold it to resolution.  This is the MAIN rule; the other
  variants run alongside it on the same data and are judged on new days only.
"""

# Polymarket city slug -> Kalshi series, ICAO station (same airport on both platforms), time zone
CITIES = {
    "austin":        ("KXHIGHAUS",  "KAUS", "America/Chicago"),
    "denver":        ("KXHIGHDEN",  "KDEN", "America/Denver"),
    "los-angeles":   ("KXHIGHLAX",  "KLAX", "America/Los_Angeles"),
    "miami":         ("KXHIGHMIA",  "KMIA", "America/New_York"),
    "houston":       ("KXHIGHTHOU", "KHOU", "America/Chicago"),
    "atlanta":       ("KXHIGHTATL", "KATL", "America/New_York"),
    "san-francisco": ("KXHIGHTSFO", "KSFO", "America/Los_Angeles"),
}

LOOKBACK_S = 15 * 60        # "moved in the last 15 minutes"
DEDUPE_S = 30 * 60          # at most one bet per threshold and variant every 30 minutes
UNDECIDED = (0.05, 0.95)    # only thresholds still in play on both platforms
COHERENT = 0.10             # bracket mids must add up to 1 +- 0.10 on both platforms
PM_FEE_RATE = 0.01          # Polymarket taker fee per share = rate * p * (1 - p)
DEFAULT_BUDGET = 25.0       # default paper bet: as much as the best ask holds, up to $25
MAKER_WAIT_S = 60 * 60      # patient variant: how long a resting bid may wait for a fill
LADDER_LEVELS = 30          # how many ask levels are stored per bet (for other bet sizes later)

# name -> rule. thr = minimum Kalshi move; extra keys switch on the refinements.
VARIANTS = {
    "main":       dict(thr=0.10, label="Main rule: Kalshi >= 10c, hold (pre-registered)"),
    "move5":      dict(thr=0.05, label="Kalshi >= 5c"),
    "move15":     dict(thr=0.15, label="Kalshi >= 15c"),
    "confirmed":  dict(thr=0.10, min_kalshi_trades=3, label="Kalshi >= 10c with >= 3 Kalshi trades in the move"),
    "first":      dict(thr=0.10, first_per_city_day=True, label="Kalshi >= 10c, first signal per city-day only"),
    "band":       dict(thr=0.10, price_band=(0.10, 0.50), skip_dead=True,
                       label="Kalshi >= 10c, bracket 10-50c and not ruled out by the readings"),
    "patient":    dict(thr=0.10, maker=True, label="Kalshi >= 10c, resting bid instead of paying the ask"),
    # added 28 Nov 2026: a signal the other way SELLS the open bets of that city-day at the best bid
    # instead of buying the opposite bracket (backtest: backtest/leadlag_net.py)
    "net":        dict(thr=0.10, net=True, label="Kalshi >= 10c, an opposite signal sells instead of buying the other side"),
}
