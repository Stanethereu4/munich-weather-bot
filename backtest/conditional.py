"""
backtest/conditional.py
-----------------------
#3  Is the market over-confident when forecasts disagree?
    For each city/day (daily highs), u = ensemble spread + disagreement between
    the 4 ensemble means for the forecast issued 2 days earlier (known by 10:00
    local the day before). Days split into thirds by u / that city's median u.
    At 10:00 the day before: does the favourite win as often as its price says,
    in each third? Strategies: buy the favourite, or sell it (buy NO).
#4  Do price jumps overshoot?
    A jump = a bracket's price moves >= 15c within 60 minutes, before the
    target day starts (so not caused by that day's weather reports).
    After the jump: does the price keep going or come back? Strategies:
    "fade" (bet against the move) vs "follow" (bet with it), entered 10 min
    after the jump, held to resolution, 1c spread + fee.

Run:  python3 -m backtest.conditional       (needs backtest_data/, no network)
"""
import glob, os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from backtest.cities import CITIES, OUT
from backtest.cities_test import daily_super

HS, FEE = 0.01, 0.01


def cost(p):
    c = np.minimum(p + HS, 0.999)
    return c + FEE * c * (1 - c)


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def price_index(path_list):
    p = pd.concat([pd.read_parquet(x) for x in path_list])
    return {tok: (g.t.to_numpy(), g.p.to_numpy(float)) for tok, g in p.sort_values("t").groupby("yes_token")}


def at(pr, tok, ts, stale=7200):
    t, p = pr[tok]
    i = np.searchsorted(t, ts, side="right") - 1
    return p[i] if i >= 0 and ts - t[i] <= stale else np.nan


def test3():
    mk = pd.read_csv(os.path.join(OUT, "markets.csv"), dtype={"yes_token": str})
    rows = []
    for city, g in mk[mk.kind == "high"].groupby("city"):
        tz = ZoneInfo(CITIES[city][1])
        sup = daily_super(pd.read_parquet(os.path.join(OUT, f"ens_{city}.parquet")), tz)
        u = sup[(sup.kind == "high") & (sup.lead == 2)].set_index("date")["u"]
        u = u / u.median()
        pr = price_index([os.path.join(OUT, "prices", f"{city}.parquet")])
        for d, b in g.groupby("target_date"):
            if d not in u.index:
                continue
            ts = int((datetime.fromisoformat(d) - timedelta(days=1)).replace(hour=10, tzinfo=tz).timestamp())
            px = np.array([at(pr, t, ts) if t in pr else np.nan for t in b.yes_token])
            if np.isnan(px).any() or not 0.9 <= px.sum() <= 1.1:
                continue
            k = int(px.argmax())
            rows.append((city, d, u[d], px[k], int(b.won.iloc[k]), -(np.sort(px)[::-1] * 0).sum()))
    df = pd.DataFrame(rows, columns=["city", "date", "u_rel", "fav_price", "fav_won", "_"])
    df["third"] = pd.qcut(df.u_rel, 3, labels=["models agree", "middle", "models disagree"])
    df["buy_fav"] = df.fav_won - cost(df.fav_price)
    df["sell_fav"] = (1 - df.fav_won) - cost(1 - df.fav_price)
    s = df.groupby("third", observed=True).agg(days=("fav_won", "size"), fav_price=("fav_price", "mean"),
                                                fav_win_rate=("fav_won", "mean"), buy_fav=("buy_fav", "mean"),
                                                t_buy=("buy_fav", tstat), sell_fav=("sell_fav", "mean"), t_sell=("sell_fav", tstat))
    s["win - price"] = s.fav_win_rate - s.fav_price
    print("#3  FAVOURITE AT 10:00 THE DAY BEFORE, split by how much the forecast models disagree (10 cities, highs)")
    print(s.round(3).to_string())


def jumps(pr, mk, tzs, step_s):
    out = []
    for (city, d), b in mk.groupby(["city", "target_date"]):
        tz = tzs[city]
        start = int(datetime.fromisoformat(d).replace(tzinfo=tz).timestamp())
        for tok, won in zip(b.yes_token, b.won):
            if tok not in pr:
                continue
            t, p = pr[tok]
            # regular grid at the data's resolution, only before the day starts
            grid = np.arange(t[0], min(start, t[-1]), step_s)
            if len(grid) < 10:
                continue
            idx = np.searchsorted(t, grid, side="right") - 1
            g = p[idx]
            lag = int(3600 // step_s)
            skip_until = -1
            for i in range(lag, len(g) - 1):
                if grid[i] <= skip_until:
                    continue
                move = g[i] - g[i - lag]
                if abs(move) < 0.15:
                    continue
                j = min(i + max(1, int(600 // step_s)), len(g) - 1)      # enter 10 min later
                pe = g[j]
                h1 = g[min(j + lag, len(g) - 1)]
                h4 = g[min(j + 4 * lag, len(g) - 1)]
                up = move > 0
                follow = (won - cost(pe)) if up else ((1 - won) - cost(1 - pe))
                fade = ((1 - won) - cost(1 - pe)) if up else (won - cost(pe))
                ssum = sum(at(pr, x, int(grid[j])) if x in pr else np.nan for x in b.yes_token)
                out.append((city, d, tok, move, g[i - lag], pe, h1 - pe, h4 - pe, won, follow, fade,
                            (start - grid[i]) / 3600, int(grid[j]), ssum))
                skip_until = grid[i] + 2 * 3600
    return out


def test4():
    rows = []
    mk = pd.read_csv(os.path.join(OUT, "markets.csv"), dtype={"yes_token": str})
    mk = mk[mk.kind == "high"]
    tzs = {c: ZoneInfo(v[1]) for c, v in CITIES.items()}
    for city, g in mk.groupby("city"):
        pr = price_index([os.path.join(OUT, "prices", f"{city}.parquet")])
        rows += [r + ("10 cities Aug-Nov",) for r in jumps(pr, g, tzs, 600)]
    mm = pd.read_csv("backtest_data/markets.csv", dtype={"yes_token": str})
    mm = mm[(mm.kind == "high") & mm.won.notna() & (mm.target_date < "2026-08-24")].assign(city="munich")
    pr = price_index(glob.glob("backtest_data/prices/highest-*.parquet"))
    rows += [r + ("Munich Mar-Aug",) for r in jumps(pr, mm, tzs, 60)]
    df = pd.DataFrame(rows, columns=["city", "date", "tok", "move", "before", "entry", "next_1h", "next_4h", "won",
                                     "follow", "fade", "hours_before_day", "entry_ts", "sum_at_entry", "sample"])
    df.to_parquet("backtest_data/analysis/jumps.parquet", index=False)
    print(f"jumps where all bracket prices add up to 0.9-1.1 at entry: {((df.sum_at_entry >= .9) & (df.sum_at_entry <= 1.1)).mean():.0%}")
    coh = df[(df.sum_at_entry >= .9) & (df.sum_at_entry <= 1.1)]
    print("COHERENT BOOKS ONLY:")
    print(coh.assign(dir=np.where(coh.move > 0, "jump up", "jump down")).groupby(["sample", "dir"]).agg(
        jumps=("move", "size"), entry=("entry", "mean"), follow=("follow", "mean"), t_follow=("follow", tstat),
        fade=("fade", "mean"), t_fade=("fade", tstat)).round(3).to_string())
    df["dir"] = np.where(df.move > 0, "jump up", "jump down")
    signed = lambda x: x  # placeholder
    df["cont_1h"] = np.sign(df.move) * df.next_1h
    df["cont_4h"] = np.sign(df.move) * df.next_4h
    print("\n#4  PRICE JUMPS of >= 15c within 1 hour, before the target day (highs)")
    print("cont = further move in the SAME direction after entry (negative = it comes back)")
    s = df.groupby(["sample", "dir"]).agg(jumps=("move", "size"), avg_move=("move", "mean"), entry=("entry", "mean"),
                                          cont_1h=("cont_1h", "mean"), cont_4h=("cont_4h", "mean"),
                                          follow=("follow", "mean"), t_follow=("follow", tstat),
                                          fade=("fade", "mean"), t_fade=("fade", tstat))
    print(s.round(3).to_string())
    a = df.groupby("sample").agg(jumps=("move", "size"), follow=("follow", "mean"), t_follow=("follow", tstat),
                                 fade=("fade", "mean"), t_fade=("fade", tstat))
    print("\nall jumps together:"); print(a.round(3).to_string())


if __name__ == "__main__":
    test3()
    test4()
