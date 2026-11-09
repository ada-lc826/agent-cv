"""Anytime view of finished runs: what each method has found after a short budget of B rounds.

    python tools/budget_curve.py --runs runs_main --score_table runs/score_table/score_table.json \\
        --budgets 5 10 15 20 30 --out runs_main/analysis/budget

On a small search space most methods reach the table optimum within 30 rounds, so the final score
cannot separate them; the score after B rounds can. Per group (mode, variant) and budget B, mean ± std
over seeds of:
  best_B     best improvement over the baseline among the trials of rounds 1..B (observed score)
  regret_B   table optimum of the run's seed minus the best observed score up to round B
  hit_B      share of runs whose trials up to round B include the seed's optimum architecture
and the area under the best-so-far curve (mean best improvement over rounds 1..max_rounds, "auc").
Scores come from the table, so they are exactly what the search saw (no noise).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional


def _ms(xs: List[Optional[float]], digits: int = 2) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.{digits}f}"
    return f"{statistics.mean(xs):.{digits}f} ± {statistics.stdev(xs):.{digits}f}"


def run_stats(run_dir: str, optimum: Dict[str, float], budgets: List[int]) -> Optional[Dict]:
    try:
        s = json.load(open(os.path.join(run_dir, "search_summary.json"), encoding="utf-8"))
        curve = json.load(open(os.path.join(run_dir, "evolution.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if s.get("stop_reason") == "running" or not s.get("baseline"):
        return None
    base = s["baseline"]["score"]
    seed = str(s["seed"])  # search seeds beyond the table's read table seed seed % n (run_matrix --table_seeds)
    opt = optimum[seed] if seed in optimum else optimum[str(s["seed"] % len(optimum))]
    trials = [p for p in curve[1:] if p.get("score") is not None]
    out = {"mode": s["mode"], "variant": s.get("variant", ""), "seed": s["seed"]}
    for b in budgets:
        seen = [p["score"] for p in trials if p["round"] <= b]
        top = max(seen + [base])
        out[f"best_{b}"] = top - base
        out[f"regret_{b}"] = opt - top
        out[f"hit_{b}"] = float(top >= opt - 1e-9)
    running, auc = base, []
    for r in range(1, s["max_rounds"] + 1):
        running = max([running] + [p["score"] for p in trials if p["round"] == r])
        auc.append(running - base)
    out["auc"] = statistics.mean(auc)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs_main")
    p.add_argument("--task", default="classification", help="classification or nb201_<dataset> (2026-10-30)")
    p.add_argument("--score_table", required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 15, 20, 30])
    p.add_argument("--out", default="")
    a = p.parse_args()

    table = json.load(open(a.score_table, encoding="utf-8"))
    optimum = {s: max(r["score"] for r in rows.values() if r.get("status") == "ok")
               for s, rows in table["seeds"].items()}
    runs = [x for d in sorted(glob.glob(os.path.join(a.runs, f"{a.task}_*")))
            if os.path.isdir(d) and (x := run_stats(d, optimum, a.budgets))]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["mode"], r["variant"])].append(r)

    cols = [f"best_{b}" for b in a.budgets] + [f"regret_{b}" for b in a.budgets] + \
           [f"hit_{b}" for b in a.budgets] + ["auc"]
    rows = [{"mode": m, "variant": v or "-", "seeds": len(rs),
             **{c: _ms([r[c] for r in rs]) for c in cols}}
            for (m, v), rs in sorted(groups.items(), key=lambda kv: -statistics.mean(r["auc"] for r in kv[1]))]
    head = ["mode", "variant", "seeds"] + cols
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(r[h]) for h in head) + " |" for r in rows]
    print(f"table optimum per seed: {optimum}")
    print("\n".join(lines))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out + ".csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=head)
            w.writeheader()
            w.writerows(rows)
        with open(a.out + ".md", "w", encoding="utf-8") as f:
            f.write(f"table optimum per seed: {optimum}\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
