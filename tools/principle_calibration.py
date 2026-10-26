"""Are higher-confidence principles better predictors? (after HypoExplore, Koo et al. 2026, Fig. 6 left)

Every time the Design Agent cites a principle, the trial records that principle's confidence at
proposal time. A citation is a prediction that the proposal improves on the best-so-far; it counts as
correct if the trial beats the previous best. Citations are binned by confidence and the hit rate per
bin is reported. A well-calibrated memory shows hit rates rising with confidence.

    python tools/principle_calibration.py --runs "runs/classification_agent_reflect*_s*" --out runs/calibration

Two criteria are reported: "improved" (delta_vs_previous > 0) and "improved_beyond_noise"
(delta_vs_previous > the run's noise_level). Failed trials count as misses.

"Beat the best" saturates once the search reaches the top of the space (B run, 2026-10-21: nothing
beat the round-6 best, so every later citation was a miss). Hence a third criterion closer to
HypoExplore's (does the principle predict the direction of the parent -> child change?), for principles
with a machine-checkable claim (reflection v3): the citation is *applicable* when the child differs from
its parent (the best before the round) only in the claim's dimensions and one of them satisfies the
claim's condition while the other satisfies its reference; the claim then predicts the sign of
child - parent, and "direction_hit" says whether the measured delta_vs_previous has that sign
("direction_hit_beyond_noise": and |delta| > noise).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionevoagent.evidence import predicted_sign  # noqa: E402
from visionevoagent.search_space import from_signature  # noqa: E402

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
        claims, parent = {}, None
        pm, bl = os.path.join(d, "principle_memory.json"), os.path.join(d, "baseline.json")
        if os.path.exists(pm):
            with open(pm, encoding="utf-8") as f:
                claims = {p["id"]: p.get("claim") or {} for p in json.load(f)}
        if os.path.exists(bl):
            with open(bl, encoding="utf-8") as f:
                parent = (json.load(f) or {}).get("architecture")
        best = parent  # implicit parent of a trial: the best before its round
        for t in sorted(trials, key=lambda t: t["round"]):
            ok = t.get("status") == "ok"
            delta = t.get("delta_vs_previous") if ok else None
            if t.get("parent"):  # explicit parent (hypothesis-guided framework)
                parent, delta_dir = from_signature(t["parent"]), (t.get("delta_vs_parent") if ok else None)
            else:
                parent, delta_dir = best, delta
            for c in t.get("cited_principles", []):
                sign = predicted_sign(claims.get(c["id"], {}), parent, t["architecture"]) if parent else None
                hit = None if sign is None or delta_dir is None else (delta_dir * sign > 0)
                rows.append({"run": os.path.basename(d), "round": t["round"], "principle": c["id"],
                             "confidence": c["confidence"], "status": c.get("status", ""),
                             "improved": delta is not None and delta > 0,
                             "improved_beyond_noise": delta is not None and delta > noise,
                             "direction_applicable": sign is not None,
                             "direction_hit": hit,
                             "direction_hit_beyond_noise": None if hit is None else (hit and abs(delta_dir) > noise)})
            if t.get("is_best"):
                best = t["architecture"]
    return rows


def table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for lo, hi in BINS:
        b = [r for r in rows if lo <= r["confidence"] < hi]
        n = len(b)
        dirs = [r for r in b if r.get("direction_hit") is not None]
        nd = len(dirs)
        out.append({"confidence": _label(lo, hi), "n": n,
                    "hit_rate": round(sum(r["improved"] for r in b) / n, 3) if n else None,
                    "hit_rate_beyond_noise": round(sum(r["improved_beyond_noise"] for r in b) / n, 3) if n else None,
                    "n_direction": nd,
                    "direction_hit_rate": round(sum(r["direction_hit"] for r in dirs) / nd, 3) if nd else None,
                    "direction_hit_rate_beyond_noise":
                        round(sum(r["direction_hit_beyond_noise"] for r in dirs) / nd, 3) if nd else None})
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
    print("| confidence | n | beat best | beat best by > noise | n direction | direction right "
          "| direction right, > noise |\n|---|---|---|---|---|---|---|")
    for r in t:
        print(f"| {r['confidence']} | {r['n']} | {r['hit_rate']} | {r['hit_rate_beyond_noise']} | "
              f"{r['n_direction']} | {r['direction_hit_rate']} | {r['direction_hit_rate_beyond_noise']} |")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        for name, data in (("citations.csv", rows), ("calibration.csv", t)):
            with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)


if __name__ == "__main__":
    main()
