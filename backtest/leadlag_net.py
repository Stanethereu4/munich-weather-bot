"""
backtest/leadlag_net.py
-----------------------
Same signals and real fills as backtest/leadlag_real.py (Kalshi >= 10c, same-airport cities),
but with three ways of handling a signal that points the OTHER way on a city-day where we
already hold a position:

  hold   every signal is its own bet, held to resolution (the pre-registered main rule)
  net    an opposite signal SELLS the open positions at a real bid instead of buying the other side
  flip   an opposite signal sells the open positions AND buys the other side

Example (LA, 28 Nov): 07:38 Kalshi up -> buy 76-77F at 0.19. 09:32 Kalshi down.
  hold: also buy 72-73F at 0.20  -> two brackets, at most one can win
  net:  sell the 76-77F at the bid (say 0.14) -> lose 5c + fees, no second bet
  flip: sell the 76-77F and buy 72-73F

Selling uses a real trade: the first time someone sold YES into a bid (or bought NO) within
10 minutes after the opposite signal. If nobody did, we can't sell and keep holding.
Units: 1 share per bet; results in cents.

Run:  python3 -m backtest.leadlag_net
"""
import os
import warnings

import numpy as np
import pandas as pd

from backtest.kalshi_test import D, load
from backtest.leadlag import features, series
from backtest.leadlag_real import dedupe

warnings.filterwarnings("ignore")
FEE, WIN, THR = 0.01, 600, 0.10


def fee(p):
    return FEE * p * (1 - p)


def fills(pm):
    """Real taker buys and sells of each YES token (NO-side trades turned into YES at 1 - price)."""
    pt = pd.read_parquet(os.path.join(D, "pm_trades.parquet"))
    pt["token_id"] = pt.token_id.astype(str)
    no2yes = dict(zip(pm.no_token.astype(str), pm.yes_token.astype(str)))
    is_no = pt.token_id.isin(no2yes)
    yes_tok = pt.token_id.map(lambda t: no2yes.get(t, t))
    px = np.where(is_no, 1 - pt.price, pt.price)
    buy = (pt.side == "BUY") != is_no          # buying YES = BUY of YES or SELL of NO
    df = pd.DataFrame({"tok": yes_tok, "t": pt.timestamp, "px": px, "buy": buy})
    B = {k: g.sort_values("t") for k, g in df[df.buy].groupby("tok")}
    S = {k: g.sort_values("t") for k, g in df[~df.buy].groupby("tok")}
    return B, S


def first_fill(book, tok, t):
    """Price of the first real trade on this token within 10 minutes after t, or None."""
    g = book.get(tok)
    if g is None:
        return None
    w = g[(g.t >= t) & (g.t <= t + WIN)]
    return None if w.empty else float(w.px.iloc[0])


def run(sig, B, S, mode):
    """Walk the signals of each city-day in time order. Returns profit (cents) per city-day."""
    out = {}
    for (city, date), g in sig.sort_values("t").groupby(["city", "date"]):
        pos, pnl, n_bets, n_closed = [], 0.0, 0, 0       # pos: open [direction, token, cost, won]
        for r in g.itertuples():
            d = 1 if r.dK > 0 else -1
            opp = [p for p in pos if p[0] == -d]
            if opp and mode in ("net", "flip"):
                for p in opp:
                    x = first_fill(S, p[1], r.t)
                    if x is not None:                    # someone really sold: we sell there too
                        pnl += x - fee(x) - p[2]
                        pos.remove(p); n_closed += 1
                if mode == "net":
                    continue
            side = "nb_up" if d > 0 else "nb_dn"
            tok, won = getattr(r, f"{side}_tok"), getattr(r, f"{side}_won")
            if tok is None or pd.isna(won):
                continue
            e = first_fill(B, tok, r.t)
            if e is None or not 0.02 < e < 0.98:
                continue
            pos.append([d, tok, e + fee(e), won]); n_bets += 1
        pnl += sum(p[3] - p[2] for p in pos)            # whatever is still open is held to the end
        if n_bets:
            out[(city, date)] = (100 * pnl, n_bets, n_closed)
    return out


def main():
    km, pm = load()
    km = km[km.same_airport]
    s = features(series(km, pm))
    B, S = fills(pm)
    sig = dedupe(s[(s.dK.abs() >= THR) & (s.dP.abs() < THR / 2)])

    # how often does a city-day get signals in both directions?
    dirs = sig.assign(d=np.sign(sig.dK)).groupby(["city", "date"]).d.nunique()
    print(f"Kalshi >= 10c, same-airport cities: {len(sig)} signals on {len(dirs)} city-days; "
          f"{(dirs == 2).sum()} city-days have signals in BOTH directions")

    res = {}
    for mode in ("hold", "net", "flip"):
        r = run(sig, B, S, mode)
        res[mode] = pd.Series({k: v[0] for k, v in r.items()})
        bets = sum(v[1] for v in r.values()); closed = sum(v[2] for v in r.values())
        x = res[mode]
        t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        print(f"\n{mode.upper():5} {bets} bets on {len(x)} city-days, {closed} positions sold early")
        print(f"      total {x.sum():+.0f}c per share-bet, {x.sum()/bets:+.2f}c per bet, "
              f"t (by city-day) {t:.2f}")

    # only the city-days where the rules differ
    both = dirs[dirs == 2].index
    print("\nOnly the city-days with signals both ways (where the rules actually differ):")
    for mode in ("hold", "net", "flip"):
        x = res[mode].reindex(both).dropna()
        print(f"   {mode:5} {x.sum():+.0f}c over {len(x)} city-days")
    d = (res["net"] - res["hold"]).reindex(both).dropna()
    print(f"   net minus hold, per city-day: mean {d.mean():+.1f}c, better on {(d > 0).sum()}, "
          f"worse on {(d < 0).sum()}, same on {(d == 0).sum()}")


if __name__ == "__main__":
    main()
