"""
backtest/traders.py
-------------------
Who makes money in the Munich markets, and how? Uses backtest/trades.py output.

Profit of one fill, held to resolution (payout = 1 if that token won, else 0):
    BUY  size s at price p:  s * (payout - p)
    SELL size s at price p:  s * (p - payout)
    Example: buy 100 "4°C" YES at 0.30, 4°C wins -> 100 * (1 - 0.30) = +70
             sell 100 at 0.30, it loses        -> 100 * (0.30 - 0)  = +30
Selling tokens you minted (1$ -> 1 YES + 1 NO) is covered by the same formula.
Taker fee from 20 Nov 2026: size * 0.01 * p * (1 - p). Maker liquidity rewards
(paid off-chain) are NOT included, so makers' true profit is a bit higher.

Run:  python3 -m backtest.traders      (needs backtest_data/, no network)
"""
import glob
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DATA = "backtest_data"
BERLIN = ZoneInfo("Europe/Berlin")
FEE_START = int(datetime(2026, 11, 20, tzinfo=ZoneInfo("UTC")).timestamp())


def load():
    tr = pd.concat([pd.read_parquet(p).assign(event_slug=os.path.basename(p)[:-8])
                    for p in glob.glob(os.path.join(DATA, "trades", "*.parquet"))])
    mk = pd.read_csv(os.path.join(DATA, "markets.csv"), dtype={"yes_token": str, "no_token": str})
    mk = mk[mk["won"].notna()]
    payout = {**dict(zip(mk["yes_token"], mk["won"].astype(int))), **dict(zip(mk["no_token"], 1 - mk["won"].astype(int)))}
    day = mk.drop_duplicates("event_slug").set_index("event_slug")[["kind", "target_date"]]
    tr["payout"] = tr["token_id"].map(payout)
    tr = tr[tr["payout"].notna()].join(day, on="event_slug")
    sgn = np.where(tr["side"] == "BUY", 1, -1)
    tr["pnl"] = sgn * tr["size"] * (tr["payout"] - tr["price"])
    fee = np.where((tr["role"] == "taker") & (tr["timestamp"] >= FEE_START),
                   tr["size"] * 0.01 * tr["price"] * (1 - tr["price"]), 0.0)
    tr["pnl"] -= fee
    tr["usd"] = tr["size"] * tr["price"]
    start = tr["target_date"].map(lambda d: int(datetime.fromisoformat(d).replace(tzinfo=BERLIN).timestamp()))
    tr["when"] = np.where(tr["timestamp"] >= start, "on the day", "before the day")
    return tr


def main():
    tr = load()
    print(f"{len(tr):,} fill rows, {tr['event_slug'].nunique()} markets "
          f"({tr['target_date'].min()} .. {tr['target_date'].max()}), {tr['wallet'].nunique():,} wallets, "
          f"volume ${tr.loc[tr.role == 'taker', 'usd'].sum():,.0f} (taker side)")
    print(f"sanity: sum of all profits = {tr['pnl'].sum():+,.0f} (should be about minus the fees)")

    print("\nPROFIT BY ROLE AND TIMING (all wallets together)")
    t = tr.groupby(["role", "when"]).agg(volume=("usd", "sum"), pnl=("pnl", "sum"))
    t["pnl per $100 traded"] = 100 * t["pnl"] / t["volume"]
    print(t.round(1).to_string())

    w = tr.groupby("wallet").agg(name=("name", "first"), pnl=("pnl", "sum"), volume=("usd", "sum"),
                                 fills=("pnl", "size"), markets=("event_slug", "nunique"))
    w["maker share"] = tr[tr.role == "maker"].groupby("wallet")["usd"].sum().reindex(w.index).fillna(0) / w["volume"]
    w["on-day share"] = tr[tr.when == "on the day"].groupby("wallet")["usd"].sum().reindex(w.index).fillna(0) / w["volume"]
    w["pnl maker"] = tr[tr.role == "maker"].groupby("wallet")["pnl"].sum().reindex(w.index).fillna(0)
    w["pnl taker"] = w["pnl"] - w["pnl maker"]
    active = w[w["markets"] >= 5]
    print(f"\nWALLETS: {len(w):,} in total, {len(active):,} active in >= 5 markets. "
          f"Profitable: {(w.pnl > 0).mean():.0%} of all, {(active.pnl > 0).mean():.0%} of active.")
    pos = w[w.pnl > 0].pnl.sort_values(ascending=False)
    print(f"Top 10 winners take {pos.head(10).sum() / pos.sum():.0%} of all profits "
          f"(${pos.head(10).sum():,.0f} of ${pos.sum():,.0f}); total losses ${w[w.pnl < 0].pnl.sum():,.0f}")

    cols = ["name", "pnl", "volume", "markets", "maker share", "on-day share", "pnl maker", "pnl taker"]
    fmt = lambda df: df[cols].assign(wallet=lambda d: d.index.str[:8]).set_index("wallet").round(
        {"pnl": 0, "volume": 0, "maker share": 2, "on-day share": 2, "pnl maker": 0, "pnl taker": 0}).to_string()
    print("\nTOP 15 WALLETS BY PROFIT"); print(fmt(w.sort_values("pnl", ascending=False).head(15)))
    print("\nBOTTOM 10 WALLETS"); print(fmt(w.sort_values("pnl").head(10)))

    print("\nPROFIT BY WALLET TYPE (active wallets)")
    a = active.copy()
    a["type"] = np.select([a["maker share"] >= 0.7, a["maker share"] <= 0.3], ["mostly maker", "mostly taker"], "mixed")
    g = a.groupby("type").agg(wallets=("pnl", "size"), profitable=("pnl", lambda x: (x > 0).mean()),
                              total_pnl=("pnl", "sum"), volume=("volume", "sum"))
    g["pnl per $100"] = 100 * g["total_pnl"] / g["volume"]
    print(g.round(2).to_string())


if __name__ == "__main__":
    main()
