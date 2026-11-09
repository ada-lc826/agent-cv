"""Best-so-far curves of finished runs as one small JSON for the paper figures (2026-11-03).

For every (task, mode, variant) group: per round r = 1..max_rounds, the mean and sample sd over seeds of
the best observed score up to round r minus the baseline score (what budget_curve.py reports at a few
budgets, here at every round). tools/plot_figures.py draws the figures from this file anywhere.

    python tools/export_curves.py --runs runs_main runs_nb201 --out runs_nb201/analysis/curves.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def curve(run_dir: str):
    try:
        s = json.load(open(os.path.join(run_dir, "search_summary.json"), encoding="utf-8"))
        ev = json.load(open(os.path.join(run_dir, "evolution.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if s.get("stop_reason") == "running" or not s.get("baseline"):
        return None
    base = cur = s["baseline"]["score"]
    ys = []
    for r in range(1, s["max_rounds"] + 1):
        cur = max([cur] + [p["score"] for p in ev[1:] if p.get("round") == r and p.get("score") is not None])
        ys.append(cur - base)
    return s["task"], s["mode"], s.get("variant", ""), ys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    groups = defaultdict(list)
    for root in a.runs:
        for d in sorted(glob.glob(os.path.join(root, "*_s[0-9]*"))):
            c = curve(d)
            if c:
                groups[(c[0], c[1], c[2])].append(c[3])
    out = defaultdict(dict)
    for (task, mode, variant), yss in sorted(groups.items()):
        n = min(len(ys) for ys in yss)
        cols = [[ys[k] for ys in yss] for k in range(n)]
        out[task][f"{mode}:{variant}"] = {
            "seeds": len(yss),
            "mean": [round(statistics.mean(c), 3) for c in cols],
            "sd": [round(statistics.stdev(c), 3) if len(c) > 1 else 0.0 for c in cols]}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{a.out}: " + ", ".join(f"{t} {len(g)} groups" for t, g in out.items()))


if __name__ == "__main__":
    main()
