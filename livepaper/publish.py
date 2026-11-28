"""
livepaper/publish.py
--------------------
Builds the dashboard (livepaper/dashboard/index.html + data.json) from the paper bets and, in
GitHub Actions, pushes it to the `dashboard` branch (served by GitHub Pages). Locally it only
writes ./dashboard_out/ for a preview.

data.json keeps, per bet, the ask ladder only as deep as a $500 bet would need, so the page can
recompute any bet size from $1 to $500 itself.
"""
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

from livepaper import store
from livepaper.config import DEFAULT_BUDGET, VARIANTS

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = "/tmp/dashboard_site" if os.environ.get("GITHUB_ACTIONS") else os.path.join(os.getcwd(), "dashboard_out")
MAX_BUDGET = 500.0


def trim(ladder):
    out, spent = [], 0.0
    for p, s in ladder:
        out.append([round(p, 4), round(s, 2)])
        spent += p * s
        if spent >= MAX_BUDGET:
            break
    return out


def build():
    bets = store.query("SELECT * FROM lp_bets ORDER BY ts")
    hb = store.query("SELECT * FROM lp_heartbeat ORDER BY ts DESC LIMIT 1")
    first = store.query("SELECT MIN(ts) AS t FROM lp_heartbeat")
    data = dict(
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        started_ts=(first[0]["t"] if first and first[0]["t"] else None),
        heartbeat=hb[0] if hb else None,
        default_budget=DEFAULT_BUDGET,
        variants={k: v["label"] for k, v in VARIANTS.items()},
        bets=[dict(id=b["id"], ts=b["ts"], v=b["variant"], city=b["city"], date=b["target_date"], x=b["x"],
                   dir=b["direction"], dk=round(b["k_move"], 4), dp=round(b["p_move"], 4), label=b["bracket_label"],
                   bid=b["best_bid"], ask=b["best_ask"], lad=trim(json.loads(b["ladder"] or "[]")),
                   maker=b["maker"], mp=b["maker_price"], ms=b["maker_status"], msh=b["maker_traded_shares"],
                   floor=b["floor_f"], kt=b["kalshi_trades"], st=b["status"], won=b["won"],
                   xt=b.get("exit_ts"), xp=b.get("exit_price"),
                   xb=[[round(p, 4), round(q, 2)] for p, q in json.loads(b.get("exit_bids") or "[]")]) for b in bets])
    os.makedirs(SITE, exist_ok=True)
    with open(os.path.join(SITE, "data.json"), "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    shutil.copy(os.path.join(HERE, "dashboard", "index.html"), os.path.join(SITE, "index.html"))
    open(os.path.join(SITE, ".nojekyll"), "w").close()
    return len(bets)


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def publish():
    if os.environ.get("GITHUB_ACTIONS") and not os.path.isdir(os.path.join(SITE, ".git")) \
            and not os.path.exists(os.path.join(SITE, ".git")):
        shutil.rmtree(SITE, ignore_errors=True)
        if git("fetch", "-q", "origin", "dashboard").returncode == 0:
            git("worktree", "add", "-B", "dashboard", SITE, "origin/dashboard")
        else:
            git("worktree", "add", "--detach", SITE)
            git("checkout", "--orphan", "dashboard", cwd=SITE)
            git("rm", "-rqf", ".", cwd=SITE)
    n = build()
    if not os.environ.get("GITHUB_ACTIONS"):
        print(f"[dashboard] {n} bets written to {SITE}")
        return
    git("add", "-A", cwd=SITE)
    if git("diff", "--cached", "--quiet", cwd=SITE).returncode == 0:
        return
    git("-c", "user.name=paper bot", "-c", "user.email=actions@users.noreply.github.com",
        "commit", "-qm", f"dashboard update ({n} bets)", cwd=SITE)
    r = git("push", "-q", "origin", "HEAD:refs/heads/dashboard", cwd=SITE)
    print(f"[dashboard] published {n} bets" + ("" if r.returncode == 0 else f" -- push failed: {r.stderr[:200]}"))


if __name__ == "__main__":
    publish()
