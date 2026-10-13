"""Aggregate finished runs into the method-comparison table (mean ± std over seeds).

    python tools/compare_runs.py --runs runs --task classification --out runs/comparison

Groups runs by (mode, variant) and reports, per group:
  best improvement over the baseline, Δparams of the best architecture, rounds to best,
  invalid-trial rate, fallback proposals, GPU hours, LLM calls and tokens, and the sample-efficiency
  metric of Li & Talwalkar (UAI 2019): how many random-search trials are needed to reach the score
  the method reached ("random trials to match"), divided by the rounds the method needed.
  Random runs that never reach the score are counted as max_rounds + 1 and the value is marked ">".
Dry-run outputs (directory name ending in _dry) are skipped unless --include_dry is given.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _load(run_dir: str) -> Optional[Dict[str, Any]]:
    try:
        with open(os.path.join(run_dir, "search_summary.json"), encoding="utf-8") as f:
            s = json.load(f)
        with open(os.path.join(run_dir, "evolution.json"), encoding="utf-8") as f:
            s["_curve"] = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    s["_dir"] = run_dir
    return s


def _ms(xs: List[float], digits: int = 2) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.{digits}f}"
    return f"{statistics.mean(xs):.{digits}f} ± {statistics.stdev(xs):.{digits}f}"


def random_trials_to_reach(random_runs: List[Dict[str, Any]], target: float) -> Tuple[Optional[float], bool]:
    """Mean first round at which random search's best-so-far reaches `target` (censored -> max_rounds+1)."""
    if not random_runs:
        return None, False
    vals, censored = [], False
    for r in random_runs:
        hit = next((p["round"] for p in r["_curve"] if p["best_score"] is not None and p["best_score"] >= target), None)
        if hit is None:
            hit, censored = r["max_rounds"] + 1, True
        vals.append(hit)
    return statistics.mean(vals), censored


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--task", default="classification")
    p.add_argument("--out", default="")
    p.add_argument("--include_dry", action="store_true")
    a = p.parse_args()

    runs = [s for d in sorted(glob.glob(os.path.join(a.runs, "*")))
            if os.path.isdir(d) and (a.include_dry or not d.endswith("_dry")) and (s := _load(d))]
    runs = [r for r in runs if r.get("task") == a.task and r.get("stop_reason") != "running"]
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        groups[(r["mode"], r.get("variant", ""))].append(r)

    rows = []
    for (mode, variant), rs in sorted(groups.items()):
        randoms = [x for x in groups.get(("random", variant), []) if x["max_rounds"] == rs[0]["max_rounds"]]
        mult, ratios, any_cens = [], [], False
        for r in rs:
            best = r["best"]["score"] if r.get("best") else None
            if best is None:
                continue
            need, cens = random_trials_to_reach(randoms, best)
            if need is None:
                continue
            any_cens |= cens
            mult.append(need)
            if r.get("rounds_to_best"):
                ratios.append(need / r["rounds_to_best"])
        prefix = ">" if any_cens else ""
        rows.append({
            "mode": mode, "variant": variant or "-", "seeds": len(rs),
            "best_improvement": _ms([r.get("best_improvement") for r in rs]),
            "best_delta_params_M": _ms([r.get("best_delta_params") for r in rs], 3),
            "rounds_to_best": _ms([r.get("rounds_to_best") for r in rs], 1),
            "random_trials_to_match": (prefix + _ms(mult, 1)) if mult else "-",
            "multiple_of_random": (prefix + _ms(ratios, 2)) if ratios else "-",
            "invalid_trial_rate": _ms([r.get("invalid_trial_rate") for r in rs], 3),
            "fallback_proposals": _ms([r.get("n_fallback_proposals", 0) for r in rs], 1),
            "gpu_hours": _ms([r.get("gpu_hours") for r in rs], 2),
            "llm_calls": _ms([r["llm"]["calls"] for r in rs], 1),
            "llm_tokens_k": _ms([(r["llm"]["input_tokens"] + r["llm"]["output_tokens"]) / 1000 for r in rs], 1),
        })

    if not rows:
        print("no finished runs found")
        return
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    table = "\n".join(lines)
    print(table)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "comparison.md"), "w", encoding="utf-8") as f:
            f.write(table + "\n")
        with open(os.path.join(a.out, "comparison.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
