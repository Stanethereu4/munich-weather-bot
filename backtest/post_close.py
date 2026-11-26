"""
backtest/post_close.py
----------------------
After the local day ends, the day's high is already fixed by the station
reports, but Polymarket settles hours later. Is the winning bracket still
cheap enough in that gap to buy it almost risk-free?

For each city/day (daily HIGHS): price of the winning bracket (10-min mids)
at local midnight + 10 min, +30 min, +1 h, +2 h, +4 h, and before settlement
(closed_time). Profit if bought there = 1 - (price + 1c spread) - fee.
Example: winner trading at 0.96 -> cost 0.97 + 0.0003 fee -> +2.97c in a few hours.

Run:  python3 -m backtest.post_close     (needs backtest_data/cities)
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from backtest.cities import CITIES

OFFSETS_MIN = [10, 30, 60, 120, 240]


def main():
    mk = pd.read_csv("backtest_data/cities/markets.csv", dtype={"yes_token": str})
    mk = mk[mk.kind == "high"]
    rows = []
    for city, g in mk.groupby("city"):
        tz = ZoneInfo(CITIES[city][1])
        p = pd.read_parquet(f"backtest_data/cities/prices/{city}.parquet")
        pr = {t: (x.t.to_numpy(), x.p.to_numpy(float)) for t, x in p.sort_values("t").groupby("yes_token")}
        for d, b in g.groupby("target_date"):
            end = int((datetime.fromisoformat(d) + timedelta(days=1)).replace(tzinfo=tz).timestamp())
            closed = int(pd.Timestamp(b.closed_time.iloc[0]).timestamp()) if pd.notna(b.closed_time.iloc[0]) else None
            w = b[b.won == 1].yes_token.iloc[0]
            if w not in pr:
                continue
            t, px = pr[w]
            for off in OFFSETS_MIN + ["pre-close"]:
                ts = end + off * 60 if off != "pre-close" else (closed - 600 if closed else None)
                if ts is None or (closed and ts >= closed):
                    continue
                i = np.searchsorted(t, ts, side="right") - 1
                if i < 0 or ts - t[i] > 3600:
                    continue
                # is the winner also the top-priced bracket at that moment?
                others = [pr[x][1][max(np.searchsorted(pr[x][0], ts, side="right") - 1, 0)]
                          for x in b.yes_token if x != w and x in pr]
                rows.append((city, d, str(off), px[i], (closed - end) / 3600 if closed else np.nan,
                             px[i] >= max(others, default=0)))
    df = pd.DataFrame(rows, columns=["city", "date", "after_midnight", "winner_price", "hours_to_close", "winner_is_top"])
    cost = np.minimum(df.winner_price + 0.01, 1.0)
    df["profit"] = 1 - cost - 0.01 * cost * (1 - cost)
    print(f"hours from local midnight to settlement: median {df.drop_duplicates(['city','date']).hours_to_close.median():.1f}, "
          f"10-90%: {df.drop_duplicates(['city','date']).hours_to_close.quantile(.1):.1f}-"
          f"{df.drop_duplicates(['city','date']).hours_to_close.quantile(.9):.1f}\n")
    order = [str(o) for o in OFFSETS_MIN] + ["pre-close"]
    s = df.groupby("after_midnight").agg(days=("profit", "size"), median_price=("winner_price", "median"),
                                         share_below_97c=("winner_price", lambda x: (x < 0.97).mean()),
                                         winner_is_top=("winner_is_top", "mean"), avg_profit=("profit", "mean"))
    print("Winning bracket's price after local midnight (minutes), all cities:")
    print(s.reindex(order).round(3).to_string())
    print("\nper city at +10 min / +60 min (median winner price, avg profit per share):")
    c = df[df.after_midnight.isin(["10", "60"])].pivot_table(index="city", columns="after_midnight",
                                                            values=["winner_price", "profit"], aggfunc=["median", "mean"])
    print(c[[("median", "winner_price", "10"), ("median", "winner_price", "60"),
             ("mean", "profit", "10"), ("mean", "profit", "60")]].round(3).to_string())
    print("\ncheapest winners at +10 min:")
    print(df[df.after_midnight == "10"].nsmallest(8, "winner_price")[["city", "date", "winner_price", "winner_is_top"]].to_string(index=False))


if __name__ == "__main__":
    main()
