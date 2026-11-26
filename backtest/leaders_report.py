"""
backtest/leaders_report.py
--------------------------
Profiles the top weather traders from backtest/leaders.py output (last 30 days).

Profit of a fill held to resolution (as in backtest/traders.py):
    BUY s at p: s * (payout - p)      SELL s at p: s * (p - payout)
    minus the taker fee (0.01 * p * (1 - p) per share) on taker fills from 20 Nov.
Only resolved markets count. Maker rebates are not included.

Timing is measured in the city's local time against the target day:
    "2+ days before", "day before", "on the day", "after the day" (between local
    midnight and settlement).

Run:  python3 -m backtest.leaders_report
"""
import os
import re
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# "python3 -m backtest.leaders_report alltime" -> the all-time top 30, full history
DATA = os.path.join("backtest_data", "leaders_alltime" if len(sys.argv) > 1 and sys.argv[1] == "alltime" else "leaders")
FEE_START = int(datetime(2026, 11, 20, tzinfo=timezone.utc).timestamp())
TZ = {
    "seoul": "Asia/Seoul", "busan": "Asia/Seoul", "hong-kong": "Asia/Hong_Kong", "tokyo": "Asia/Tokyo",
    "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai", "shenzhen": "Asia/Shanghai", "guangzhou": "Asia/Shanghai",
    "chengdu": "Asia/Shanghai", "chongqing": "Asia/Shanghai", "wuhan": "Asia/Shanghai", "qingdao": "Asia/Shanghai",
    "jinan": "Asia/Shanghai", "zhengzhou": "Asia/Shanghai", "taipei": "Asia/Taipei", "lucknow": "Asia/Kolkata",
    "karachi": "Asia/Karachi", "manila": "Asia/Manila", "singapore": "Asia/Singapore", "kuala-lumpur": "Asia/Kuala_Lumpur",
    "jeddah": "Asia/Riyadh", "tel-aviv": "Asia/Jerusalem", "ankara": "Europe/Istanbul", "istanbul": "Europe/Istanbul",
    "moscow": "Europe/Moscow", "helsinki": "Europe/Helsinki", "warsaw": "Europe/Warsaw", "munich": "Europe/Berlin",
    "paris": "Europe/Paris", "london": "Europe/London", "madrid": "Europe/Madrid", "amsterdam": "Europe/Amsterdam",
    "milan": "Europe/Rome", "cape-town": "Africa/Johannesburg", "wellington": "Pacific/Auckland",
    "buenos-aires": "America/Argentina/Buenos_Aires", "sao-paulo": "America/Sao_Paulo", "nyc": "America/New_York",
    "miami": "America/New_York", "atlanta": "America/New_York", "toronto": "America/Toronto",
    "chicago": "America/Chicago", "houston": "America/Chicago", "austin": "America/Chicago", "dallas": "America/Chicago",
    "denver": "America/Denver", "mexico-city": "America/Mexico_City", "panama-city": "America/Panama",
    "seattle": "America/Los_Angeles", "los-angeles": "America/Los_Angeles", "san-francisco": "America/Los_Angeles",
}
SLUG = re.compile(r"^(highest|lowest)-temperature-in-(.+)-on-([a-z]+)-(\d+)-(\d{4})$")


def parse(slug):
    m = SLUG.match(slug or "")
    if not m:
        return None, None, None
    d = datetime.strptime(f"{m.group(3)} {m.group(4)} {m.group(5)}", "%B %d %Y").date()
    return ("high" if m.group(1) == "highest" else "low"), m.group(2), d


def load():
    fdir = os.path.join(DATA, "fills")
    f = pd.concat([pd.read_parquet(os.path.join(fdir, x)) for x in sorted(os.listdir(fdir))]) if os.path.isdir(fdir) \
        else pd.read_parquet(os.path.join(DATA, "fills.parquet"))
    for c in ("name", "role", "side", "event_slug", "outcome"):
        f[c] = f[c].astype(str)
    pay = pd.read_csv(os.path.join(DATA, "payout.csv"), dtype={"token_id": str}).drop_duplicates("token_id")
    f = f.merge(pay, on="token_id", how="left")
    meta = {s: parse(s) for s in f.event_slug.dropna().unique()}
    f["kind"] = f.event_slug.map(lambda s: meta.get(s, (None,) * 3)[0])
    f["city"] = f.event_slug.map(lambda s: meta.get(s, (None,) * 3)[1])
    f["date"] = f.event_slug.map(lambda s: meta.get(s, (None,) * 3)[2])
    f = f[f.city.notna()]
    start = {(c, d): int(datetime(d.year, d.month, d.day, tzinfo=ZoneInfo(TZ.get(c, "UTC"))).timestamp())
             for c, d in set(zip(f.city, f.date))}
    f["h_from_start"] = [(t - start[(c, d)]) / 3600 for t, c, d in zip(f.timestamp, f.city, f.date)]
    f["when"] = pd.cut(f.h_from_start, [-1e9, -24, 0, 24, 1e9],
                       labels=["2+ days before", "day before", "on the day", "after the day"])
    f["usd"] = f["size"] * f["price"]
    sgn = np.where(f.side == "BUY", 1, -1)
    fee = np.where((f.role == "taker") & (f.timestamp >= FEE_START), f["size"] * 0.01 * f.price * (1 - f.price), 0)
    f["pnl"] = sgn * f["size"] * (f.payout - f.price) - fee
    f["no_side"] = f.outcome.str.lower().eq("no")
    return f


def profile(f):
    r = f[f.payout.notna()]
    rows = []
    for (w, name), g in r.groupby(["wallet", "name"]):
        buys = g[g.side == "BUY"]
        day = g.groupby("date").pnl.sum()
        t = day.mean() / (day.std(ddof=1) / np.sqrt(len(day))) if len(day) > 2 and day.std() > 0 else np.nan
        bw = lambda lo, hi: buys.loc[(buys.price >= lo) & (buys.price < hi), "usd"].sum() / max(buys.usd.sum(), 1e-9)
        when = g.groupby("when", observed=False).usd.sum() / g.usd.sum()
        pnl_when = g.groupby("when", observed=False).pnl.sum()
        rows.append(dict(
            name=str(name)[:16], pnl=g.pnl.sum(), volume=g.usd.sum(), ret_per_100=100 * g.pnl.sum() / g.usd.sum(),
            days=len(day), win_days=(day > 0).mean(), t_daily=t, cities=g.city.nunique(),
            top_cities=", ".join(g.groupby("city").usd.sum().nlargest(3).index),
            lows=g.loc[g.kind == "low", "usd"].sum() / g.usd.sum(),
            maker=g.loc[g.role == "maker", "usd"].sum() / g.usd.sum(),
            buy_no=buys.loc[buys.no_side, "usd"].sum() / max(buys.usd.sum(), 1e-9),
            buy_lt10c=bw(0, .10), buy_10_50=bw(.10, .50), buy_50_90=bw(.50, .90), buy_gt90c=bw(.90, 1.01),
            sells=g.loc[g.side == "SELL", "usd"].sum() / g.usd.sum(),
            v_2d=when.get("2+ days before", 0), v_dbefore=when.get("day before", 0), v_onday=when.get("on the day", 0),
            v_after=when.get("after the day", 0),
            pnl_2d=pnl_when.get("2+ days before", 0), pnl_dbefore=pnl_when.get("day before", 0),
            pnl_onday=pnl_when.get("on the day", 0), pnl_after=pnl_when.get("after the day", 0),
            pnl_maker=g.loc[g.role == "maker", "pnl"].sum(), pnl_taker=g.loc[g.role == "taker", "pnl"].sum()))
    return pd.DataFrame(rows).sort_values("pnl", ascending=False)


def main():
    f = load()
    board = pd.read_csv(os.path.join(DATA, "board.csv"))
    print(f"{len(f):,} weather fills from {f.wallet.nunique()} wallets, {f.event_slug.nunique()} events, "
          f"{f.payout.notna().mean():.0%} of fills in resolved markets")
    p = profile(f)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    pct = lambda c: (100 * p[c]).round(0)
    a = p[["name", "pnl", "volume", "ret_per_100", "days", "win_days", "t_daily", "cities", "top_cities"]].copy()
    a[["pnl", "volume"]] = a[["pnl", "volume"]].round(0); a["ret_per_100"] = a.ret_per_100.round(1)
    a["win_days"] = pct("win_days"); a["t_daily"] = a.t_daily.round(1)
    print("\n1) RESULTS, last 30 days, resolved markets (our calculation, held to resolution, no rebates)")
    print(a.to_string(index=False))
    b = pd.DataFrame({"name": p.name})
    for c in ["maker", "lows", "buy_no", "buy_lt10c", "buy_10_50", "buy_50_90", "buy_gt90c", "sells",
              "v_2d", "v_dbefore", "v_onday", "v_after"]:
        b[c] = pct(c).astype(int)
    print("\n2) STYLE (% of their $ volume): maker share, lows share, buys of NO, price of what they buy, "
          "share sold, and WHEN they trade")
    print(b.to_string(index=False))
    c = p[["name", "pnl_2d", "pnl_dbefore", "pnl_onday", "pnl_after", "pnl_maker", "pnl_taker"]].round(0)
    print("\n3) WHERE THE PROFIT COMES FROM ($)")
    print(c.to_string(index=False))
    r = f[f.payout.notna()].copy()
    r["month"] = pd.to_datetime(r.timestamp, unit="s").dt.strftime("%Y-%m")
    m = r.pivot_table(index="name", columns="month", values="pnl", aggfunc="sum").fillna(0)
    m = m.loc[m.sum(axis=1).sort_values(ascending=False).index]
    m.index = [str(i)[:16] for i in m.index]
    print("\n4) PROFIT BY MONTH ($ thousands, resolved fills, by trade date)")
    print((m / 1000).map(lambda x: "." if abs(x) < 0.05 else f"{x:.1f}").to_string())   # in $ thousands
    v = r.pivot_table(index="name", columns="month", values="usd", aggfunc="sum").fillna(0)
    ret = (100 * r.groupby(["month"]).pnl.sum() / r.groupby("month").usd.sum()).round(1)
    print("\nall these wallets together, return per $100 traded by month:")
    print(ret.to_string())
    print("\nleaderboard figures for reference (their own PnL definition):")
    print(board[["user_name", "period", "pnl", "volume"]].round(0).to_string(index=False))


if __name__ == "__main__":
    main()
