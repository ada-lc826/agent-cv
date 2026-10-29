"""Ground-truth audit of the principle library (plan A, measure 1; after POPPER's error control).

The proxy score table holds every architecture of the space, so each claim can be decided with complete
data and the search's verdicts (confirmed > 0.75 / refuted < 0.25, from a few trials) can be audited:

  truth (model)  the search's own Bayesian linear model (main effects + the claim's interaction), fitted
                 on all architectures with the 3-seed mean score (noise 0.25 / sqrt of the seed count); the claim holds
                 if its confidence is > 0.75, fails if < 0.25, else "unclear". Same semantics as the search.
  truth (pairs)  model-free: mean score difference over every matched pair (condition vs reference,
                 all other dimensions equal) in the table; holds / fails when the mean clears / misses the
                 claim's threshold by more than 1.64 standard errors.

Per group (mode, variant), over the distinct claims of each run:
  false confirmation rate  confirmed by the search, but the truth says the claim fails
  false refutation rate    refuted by the search, but the truth says the claim holds
  Brier                    (final confidence - truth)^2 over all claims the truth decides; a constant 0.5
                           scores 0.25. This also covers variants whose confidences never move (D-9, D-10).

    python tools/audit_principles.py --runs "runs_main/classification_agent_reflect*" \\
        --score_table runs/score_table/score_table.json --out runs_main/analysis/audit
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionevoagent.evidence import _terms, claim_key, claim_text, fit  # noqa: E402
from visionevoagent.search_space import (BASELINE, activate, architecture_signature, canonicalize,  # noqa: E402
                                         from_signature)

TASK = "classification"  # set from --task in main()


def mean_table(table: Dict[str, Any]) -> Dict[str, float]:
    by_sig = defaultdict(list)
    for rows in table["seeds"].values():
        for s, r in rows.items():
            if r.get("status") == "ok":
                by_sig[architecture_signature(from_signature(s))].append(r["score"])
    return {s: statistics.mean(v) for s, v in by_sig.items() if len(v) == len(table["seeds"])}


def pair_truth(claim: Dict[str, Any], scores: Dict[str, float], noise: float) -> Dict[str, Any]:
    cond = _terms(claim["condition"])
    given = dict(_terms(claim.get("reference")))
    ref = {d: given.get(d, BASELINE[d]) for d, _ in cond}
    diffs = []
    for s, v in scores.items():
        a = from_signature(s)
        if not all(a[d] == val for d, val in cond):
            continue
        b = architecture_signature(canonicalize({**a, **ref}))
        if b != s and b in scores:
            diffs.append(v - scores[b])
    if not diffs:
        return {"pairs": 0, "pair_effect": None, "pair_truth": "unclear"}
    mu = statistics.mean(diffs)
    se = (statistics.stdev(diffs) / math.sqrt(len(diffs))) if len(diffs) > 1 else noise
    se = max(se, noise / math.sqrt(len(diffs)))
    m = float(claim.get("min_effect", 0.0))
    margin = (mu - m) if claim["direction"] == "increase" else (-m - mu)
    verdict = "holds" if margin > 1.64 * se else "fails" if margin < -1.64 * se else "unclear"
    return {"pairs": len(diffs), "pair_effect": round(mu, 3), "pair_truth": verdict}


def audit_run(run_dir: str, scores: Dict[str, float], noise: float, n_seeds: int) -> List[Dict[str, Any]]:
    try:
        principles = json.load(open(os.path.join(run_dir, "principle_memory.json"), encoding="utf-8"))
        summary = json.load(open(os.path.join(run_dir, "search_summary.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return []
    claims = [p for p in principles if p.get("claim")]
    if not claims:
        return []
    base_sig = architecture_signature(BASELINE)
    rows = [{"architecture": from_signature(s), "status": "ok", "score": v} for s, v in scores.items() if s != base_sig]
    sd = noise / math.sqrt(n_seeds)
    model = fit(rows, {"architecture": dict(BASELINE), "status": "ok", "score": scores[base_sig]}, TASK, sd,
                claims=[p["claim"] for p in claims])
    out, seen = [], set()
    for p in sorted(claims, key=lambda p: -p.get("confidence", 0.5)):
        try:
            key = claim_key(p["claim"])
        except (KeyError, TypeError):
            continue
        if key in seen:
            continue
        seen.add(key)
        c = model.confidence(p["claim"])["confidence"]
        truth = "holds" if c > 0.75 else "fails" if c < 0.25 else "unclear"
        out.append({"run": os.path.basename(run_dir), "mode": summary["mode"], "variant": summary.get("variant", ""),
                    "claim": claim_text(p["claim"]), "status": p.get("status", ""),
                    "search_conf": round(p.get("confidence", 0.5), 3), "truth_conf": round(c, 3), "truth": truth,
                    **pair_truth(p["claim"], scores, sd)})
    return out


def summarize(rows: List[Dict[str, Any]], truth_key: str) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["mode"], r["variant"])].append(r)
    res = []
    for (mode, variant), rs in sorted(groups.items()):
        conf = [r for r in rs if r["status"] == "confirmed"]
        refu = [r for r in rs if r["status"] == "refuted"]
        dec = [r for r in rs if r[truth_key] != "unclear"]
        rate = lambda xs, bad: (f"{sum(r[truth_key] == bad for r in xs)}/"  # noqa: E731
                                f"{sum(r[truth_key] != 'unclear' for r in xs)}")
        brier = statistics.mean((r["search_conf"] - (r[truth_key] == "holds")) ** 2 for r in dec) if dec else None
        res.append({"mode": mode, "variant": variant or "-", "claims": len(rs),
                    "truth_holds/fails/unclear": f"{sum(r[truth_key] == 'holds' for r in rs)}/"
                                                 f"{sum(r[truth_key] == 'fails' for r in rs)}/"
                                                 f"{sum(r[truth_key] == 'unclear' for r in rs)}",
                    "confirmed": len(conf), "false_confirm": rate(conf, "fails"),
                    "refuted": len(refu), "false_refute": rate(refu, "holds"),
                    "brier": round(brier, 3) if brier is not None else None})
    return res


def md(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(none)"
    h = list(rows[0])
    return "\n".join(["| " + " | ".join(h) + " |", "|" + "---|" * len(h)] +
                     ["| " + " | ".join(str(r[k]) for k in h) + " |" for r in rows])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="glob of run directories")
    p.add_argument("--score_table", required=True)
    p.add_argument("--noise", type=float, default=0.25, help="seed noise of one training (nb201: the config's noise_level)")
    p.add_argument("--task", default="classification", help="classification or nb201_<dataset> (2026-10-30)")
    p.add_argument("--out", default="")
    a = p.parse_args()
    global TASK
    TASK = a.task
    activate(a.task)
    table = json.load(open(a.score_table, encoding="utf-8"))
    n_seeds = len(table["seeds"])
    scores = mean_table(table)
    rows = [r for d in sorted(glob.glob(a.runs)) if os.path.isdir(d) for r in audit_run(d, scores, a.noise, n_seeds)]
    text = (f"{len(rows)} distinct claims from {len({r['run'] for r in rows})} runs; truth from {len(scores)} "
            f"architectures ({n_seeds}-seed mean)\n\n## truth = search model on complete data\n\n"
            f"{md(summarize(rows, 'truth'))}\n\n## truth = matched pairs (model-free)\n\n"
            f"{md(summarize(rows, 'pair_truth'))}")
    print(text)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "audit.md"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        if rows:
            with open(os.path.join(a.out, "claims.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
