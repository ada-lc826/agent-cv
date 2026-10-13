"""Which two LLM comparison methods go on to segmentation? (rule fixed in advance, 2026-10-13)

Rule, agreed before any segmentation run and not to be changed afterwards:
  among the five LLM comparison methods (GENIUS, EvoPrompting, LLMatic, LAPT, NADER), take the two with
  the highest final best score in the formal classification comparison (30 trials, main setting: no
  ablation flags, thinking off), averaged over seeds 0, 1 and 2. Every one of the five must have all
  three seeds finished, otherwise no selection is made. Ties are broken by the lower standard deviation,
  then alphabetically.
Random, Reg-Evo and our method run on segmentation regardless.

    python tools/select_seg_methods.py --runs runs --out runs/seg_method_selection.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

LLM_BASELINES = ("genius", "evoprompting", "llmatic", "lapt", "nader")
SEEDS = (0, 1, 2)
VARIANT = "t30"           # main setting at the formal budget: only --max_rounds 30, nothing else
TASK = "classification"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="")
    a = p.parse_args()

    table, missing = [], []
    for m in LLM_BASELINES:
        scores = []
        for s in SEEDS:
            path = os.path.join(a.runs, f"{TASK}_{m}_{VARIANT}_s{s}", "search_summary.json")
            if not os.path.exists(path):
                missing.append(path)
                continue
            summary = json.load(open(path, encoding="utf-8"))
            if summary.get("stop_reason") == "running" or summary.get("rounds_done", 0) < 30:
                missing.append(f"{path} (unfinished)")
                continue
            best = summary.get("best") or {}
            scores.append(best.get("score"))
        if len(scores) == len(SEEDS) and None not in scores:
            table.append({"method": m, "mean": round(statistics.mean(scores), 3),
                          "std": round(statistics.stdev(scores), 3), "scores": scores})
    if missing:
        print("not all runs are finished; no selection made:")
        for x in missing:
            print("  -", x)
        sys.exit(1)

    table.sort(key=lambda r: (-r["mean"], r["std"], r["method"]))
    chosen = [r["method"] for r in table[:2]]
    print("| method | mean best score | std | seeds 0/1/2 |\n|---|---|---|---|")
    for r in table:
        print(f"| {r['method']}{' *' if r['method'] in chosen else ''} | {r['mean']} | {r['std']} | "
              f"{', '.join(str(x) for x in r['scores'])} |")
    print("selected for segmentation:", ", ".join(chosen))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"rule": __doc__.split("Rule, ")[1].split("\n\n")[0], "table": table, "selected": chosen},
                      f, indent=2)


if __name__ == "__main__":
    main()
