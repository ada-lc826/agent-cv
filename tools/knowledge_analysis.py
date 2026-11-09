"""Knowledge-quality analyses for innovation 2 (plan A, 2026-10-30), all from finished runs and the score
table, no training:

  1. calibration per group: the direction criterion of principle_calibration.py, split by (mode, variant),
     so v4 can be compared with D-8 fixed / D-9 random / D-10 no-update / D-11 / D-12 v3 / thinking.
  2. evidence model vs plain regression: for every run, fit the search's Bayesian linear model on the run's
     own trials, once with the interaction terms of its principles and once with main effects only, and
     score its ranking of (a) the proxy-table architectures the run never trained and (b) the fully
     trained architectures. Runs without principles (agent, random, ...) give the "just a regression on
     what was observed" reference.
  3. selection counterfactual: every candidate of a hypothesis-guided round is in the table, so the score
     of the chosen candidate can be compared with the Design Agent's first candidate and with the
     candidate a gain-only rule would have picked.
  4. curves.csv (mean best-so-far improvement per round) and calibration.csv for figures.

    python tools/knowledge_analysis.py --runs runs_main --score_table runs/score_table/score_table.json \\
        --full configs/full_results_cls.json --out runs_main/analysis/knowledge
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_principles_full import spearman  # noqa: E402
from principle_calibration import BINS, _label, collect  # noqa: E402
from visionevoagent.evidence import fit  # noqa: E402
from visionevoagent.search_space import activate, architecture_signature, from_signature  # noqa: E402

TASK = "classification"  # set from --task in main()


def _ms(xs: List[Optional[float]], digits: int = 2) -> str:
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.{digits}f}"
    return f"{statistics.mean(xs):.{digits}f} ± {statistics.stdev(xs):.{digits}f}"


def _json(path: str, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def norm(sig: str) -> str:
    return architecture_signature(from_signature(sig))


def load_runs(root: str, task: str = "classification") -> List[Dict[str, Any]]:
    runs = []
    for d in sorted(glob.glob(os.path.join(root, f"{task}_*"))):
        s = _json(os.path.join(d, "search_summary.json"))
        if not s or s.get("stop_reason") == "running":
            continue
        runs.append({"dir": d, "group": (s["mode"], s.get("variant", "")), "seed": str(s["seed"]),
                     "noise": (_json(os.path.join(d, "run_config.json"), {}) or {}).get("noise_level", 0.25),
                     "trials": _json(os.path.join(d, "trial_memory.json"), []) or [],
                     "principles": _json(os.path.join(d, "principle_memory.json"), []) or [],
                     "baseline": _json(os.path.join(d, "baseline.json")),
                     "curve": _json(os.path.join(d, "evolution.json"), []) or []})
    return runs


def calibration(runs) -> List[Dict[str, Any]]:
    by_group = defaultdict(list)
    for r in runs:
        if r["principles"]:
            by_group[r["group"]] += [c for c in collect(r["dir"]) if c["direction_hit"] is not None]
    rows = []
    for g, cs in sorted(by_group.items()):
        row = {"mode": g[0], "variant": g[1] or "-", "n_direction": len(cs)}
        ece = 0.0
        for lo, hi in BINS:
            b = [c for c in cs if lo <= c["confidence"] < hi]
            row[_label(lo, hi)] = f"{sum(c['direction_hit'] for c in b) / len(b):.2f} ({len(b)})" if b else "-"
            if b:  # expected calibration error: |accuracy - mean confidence| per bin, weighted by bin size
                ece += len(b) / len(cs) * abs(sum(c["direction_hit"] for c in b) / len(b)
                                               - sum(c["confidence"] for c in b) / len(b))
        row["ece"] = round(ece, 3) if cs else None
        rows.append(row)
    return rows


def tseed(table, seed: str) -> str:
    """Table seed read by a search seed (run_matrix --table_seeds: seed % number of table seeds)."""
    return seed if seed in table["seeds"] else str(int(seed) % len(table["seeds"]))


def model_quality(runs, table, full) -> List[Dict[str, Any]]:
    fsig = norm(full["baseline_signature"]) if full else None
    fscores = {norm(s): v for s, v in full["scores"].items()} if full else {}
    per_run = []
    for r in runs:
        if not r["baseline"] or not r["trials"]:
            continue
        rows = [{"architecture": t["architecture"], "status": t.get("status"), "score": t.get("score")}
                for t in r["trials"]]
        seen = {norm(t["signature"]) for t in r["trials"]} | {norm(r["baseline"]["signature"])}
        truth = {s: v["score"] for s, v in table["seeds"][tseed(table, r["seed"])].items() if v.get("status") == "ok"}
        unseen = [s for s in truth if s not in seen]
        if len(unseen) > 3000:  # NAS-Bench-201: a fixed random sample of the untrained cells keeps this fast
            unseen = random.Random(0).sample(sorted(unseen), 3000)
        claims = [p["claim"] for p in r["principles"] if p.get("claim")]
        out = {"group": r["group"], "n_unseen": len(unseen)}
        for tag, cl in (("main", []), ("claims", claims)):
            if tag == "claims" and not claims:
                continue
            m = fit(rows, r["baseline"], TASK, r["noise"], claims=cl)
            pred = lambda s: m.predict(from_signature(s))[0]  # noqa: E731
            out[f"table_unseen_{tag}"] = spearman([pred(s) for s in unseen], [truth[s] for s in unseen]) \
                if len(unseen) >= 4 else None
            fs = [s for s in fscores if s != fsig]
            out[f"full_{tag}"] = spearman([pred(s) for s in fs], [fscores[s] for s in fs]) if fs else None
            # true rank (1 = best) of the untrained architecture the model rates highest
            if unseen:
                top = max(unseen, key=pred)
                out[f"top_unseen_rank_{tag}"] = 1 + sum(v > truth[top] for v in truth.values())
        per_run.append(out)
    groups = defaultdict(list)
    for o in per_run:
        groups[o["group"]].append(o)
    res = []
    for g, os_ in sorted(groups.items()):
        res.append({"mode": g[0], "variant": g[1] or "-", "runs": len(os_),
                    "untrained": _ms([o["n_unseen"] for o in os_], 0),
                    "rho_untrained_main": _ms([o.get("table_unseen_main") for o in os_]),
                    "rho_untrained_claims": _ms([o.get("table_unseen_claims") for o in os_]),
                    "rho_full_main": _ms([o.get("full_main") for o in os_]),
                    "rho_full_claims": _ms([o.get("full_claims") for o in os_]),
                    "top_untrained_rank_main": _ms([o.get("top_unseen_rank_main") for o in os_], 1),
                    "top_untrained_rank_claims": _ms([o.get("top_unseen_rank_claims") for o in os_], 1)})
    return res


def selection_counterfactual(runs, table) -> List[Dict[str, Any]]:
    groups = defaultdict(lambda: defaultdict(list))
    for r in runs:
        truth = {s: v["score"] for s, v in table["seeds"][tseed(table, r["seed"])].items() if v.get("status") == "ok"}
        g = groups[r["group"]]
        for t in r["trials"]:
            sel = t.get("selection") or {}
            cands = sel.get("candidates") or []
            if len(cands) < 2:
                continue
            sc = [truth.get(norm(c["signature"])) for c in cands]
            if any(v is None for v in sc):
                continue
            chosen = sel["chosen"]
            by_score = max(range(len(cands)), key=lambda i: (cands[i]["score"], -i))
            by_gain = max(range(len(cands)), key=lambda i: (cands[i]["gain"], -i))
            g["rounds"].append(1)
            g["override"].append(float(by_score != 0))
            g["chosen_minus_first"].append(sc[chosen] - sc[0])
            g["framework_minus_first"].append(sc[by_score] - sc[0])
            g["gain_only_minus_first"].append(sc[by_gain] - sc[0])
            g["framework_minus_gain_only"].append(sc[by_score] - sc[by_gain])
            g["best_candidate_minus_first"].append(max(sc) - sc[0])
            if by_score != 0:
                d = sc[by_score] - sc[0]
                g["override_better"].append(float(d > r["noise"]))
                g["override_worse"].append(float(d < -r["noise"]))
    res = []
    for (mode, variant), g in sorted(groups.items()):
        if not g["rounds"]:
            continue
        mean = lambda k: round(statistics.mean(g[k]), 3) if g[k] else None  # noqa: E731
        res.append({"mode": mode, "variant": variant or "-", "rounds": len(g["rounds"]),
                    "framework_overrides_first": mean("override"),
                    "chosen-first": mean("chosen_minus_first"),
                    "framework_pick-first": mean("framework_minus_first"),
                    "gain_only_pick-first": mean("gain_only_minus_first"),
                    "framework_pick-gain_only": mean("framework_minus_gain_only"),
                    "best_candidate-first": mean("best_candidate_minus_first"),
                    "override_better_by_noise": mean("override_better"),
                    "override_worse_by_noise": mean("override_worse")})
    return res


def curves(runs, max_rounds: int = 30) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for r in runs:
        if not r["baseline"]:
            continue
        base, cur, ys = r["baseline"]["score"], r["baseline"]["score"], []
        for k in range(1, max_rounds + 1):
            cur = max([cur] + [p["score"] for p in r["curve"][1:] if p.get("round") == k and p.get("score") is not None])
            ys.append(cur - base)
        groups[r["group"]].append(ys)
    out = []
    for (mode, variant), yss in sorted(groups.items()):
        for k in range(max_rounds):
            v = [ys[k] for ys in yss]
            out.append({"mode": mode, "variant": variant or "-", "round": k + 1, "mean": round(statistics.mean(v), 4),
                        "min": round(min(v), 4), "max": round(max(v), 4), "seeds": len(v)})
    return out


def md(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(none)"
    h = list(rows[0])
    return "\n".join(["| " + " | ".join(h) + " |", "|" + "---|" * len(h)] +
                     ["| " + " | ".join(str(r[k]) for k in h) + " |" for r in rows])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs_main")
    p.add_argument("--score_table", required=True)
    p.add_argument("--full", default="", help="full-training results (classification only)")
    p.add_argument("--task", default="classification", help="classification or nb201_<dataset> (2026-10-30)")

    p.add_argument("--out", default="")
    a = p.parse_args()
    global TASK
    TASK = a.task
    activate(a.task)
    table = _json(a.score_table)
    table = {"seeds": {s: {norm(k): v for k, v in rows.items()} for s, rows in table["seeds"].items()}}
    full = _json(a.full) if a.full else None
    runs = load_runs(a.runs, a.task)

    parts = {"calibration": calibration(runs), "model_quality": model_quality(runs, table, full),
             "selection": selection_counterfactual(runs, table)}
    text = "\n\n".join(f"## {k}\n\n{md(v)}" for k, v in parts.items())
    print(text)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "knowledge.md"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        for name, data in list(parts.items()) + [("curves", curves(runs))]:
            if data:
                with open(os.path.join(a.out, name + ".csv"), "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(data[0]))
                    w.writeheader()
                    w.writerows(data)


if __name__ == "__main__":
    main()
