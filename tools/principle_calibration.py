"""Are higher-confidence principles better predictors? (after HypoExplore, Koo et al. 2026, Fig. 6 left)

Every time the Design Agent cites a principle, the trial records that principle's confidence at
proposal time. A citation is a prediction that the proposal improves on the best-so-far; it counts as
correct if the trial beats the previous best. Citations are binned by confidence and the hit rate per
bin is reported. A well-calibrated memory shows hit rates rising with confidence.

    python tools/principle_calibration.py --runs "runs/classification_agent_reflect*_s*" --out runs/calibration

Two criteria are reported: "improved" (delta_vs_previous > 0, HypoExplore's criterion) and
"improved_beyond_noise" (delta_vs_previous > the run's noise_level). Failed trials count as misses.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List

BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001)]


def _label(lo: float, hi: float) -> str:
    return f"[{lo:.2f}, {min(hi, 1.0):.2f}{']' if hi > 1 else ')'}"


def collect(pattern: str) -> List[Dict[str, Any]]:
    rows = []
    for d in sorted(glob.glob(pattern)):
        tm, rc = os.path.join(d, "trial_memory.json"), os.path.join(d, "run_config.json")
        if not os.path.exists(tm):
            continue
        noise = 0.0
        if os.path.exists(rc):
            with open(rc, encoding="utf-8") as f:
                noise = json.load(f).get("noise_level", 0.0)
        with open(tm, encoding="utf-8") as f:
            trials = json.load(f)
        for t in trials:
            delta = t.get("delta_vs_previous") if t.get("status") == "ok" else None
            for c in t.get("cited_principles", []):
                rows.append({"run": os.path.basename(d), "round": t["round"], "principle": c["id"],
                             "confidence": c["confidence"], "status": c.get("status", ""),
                             "improved": delta is not None and delta > 0,
                             "improved_beyond_noise": delta is not None and delta > noise})
    return rows


def table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for lo, hi in BINS:
        b = [r for r in rows if lo <= r["confidence"] < hi]
        n = len(b)
        out.append({"confidence": _label(lo, hi), "n": n,
                    "hit_rate": round(sum(r["improved"] for r in b) / n, 3) if n else None,
                    "hit_rate_beyond_noise": round(sum(r["improved_beyond_noise"] for r in b) / n, 3) if n else None})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="glob of run directories")
    p.add_argument("--out", default="")
    a = p.parse_args()
    rows = collect(a.runs)
    if not rows:
        print("no cited principles found (needs agent_reflect runs with the full reflection style)")
        return
    t = table(rows)
    print(f"{len(rows)} citations from {len({r['run'] for r in rows})} runs")
    print("| confidence | n | hit rate | hit rate beyond noise |\n|---|---|---|---|")
    for r in t:
        print(f"| {r['confidence']} | {r['n']} | {r['hit_rate']} | {r['hit_rate_beyond_noise']} |")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        for name, data in (("citations.csv", rows), ("calibration.csv", t)):
            with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)


if __name__ == "__main__":
    main()
