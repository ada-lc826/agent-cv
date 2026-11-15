"""Seed-level paired tests: the full method against its controls (2026-11-15).

For every run of the compared groups, three numbers:
  brier     Brier score of the run's final confidences against the local matched-pair truth
            (claims.csv of tools/audit_principles.py; claims whose local truth is "unclear" are skipped)
  verdicts  confirmed + refuted claims of the run
  auc       mean best improvement over rounds 1..max_rounds (as tools/budget_curve.py)
Runs are paired by (setting, search seed). For each control and metric: mean paired difference
(control - full), a bootstrap 95% interval of it, how many pairs favour the full method, and the exact
two-sided Wilcoxon signed-rank p-value (zero differences dropped, tied ranks averaged; exact enumeration of
the sign assignments). Holm's correction is applied over the family of pooled tests.

    python tools/paired_tests.py --out runs_nb201/analysis/paired_judge \\
        --setting module runs_main runs_main/analysis/audit_judge/claims.csv runs/score_table/score_table.json classification t30 \\
        --setting cifar100 runs_nb201 runs_nb201/analysis/audit_judge10_cifar100/claims.csv runs/score_table_nb201/nb201_cifar100.json nb201_cifar100 t50 \\
        ...
Each --setting: name, runs root, claims.csv, score table, task, round tag of the full method's variant.
Groups: full = variant "<tag>", D-13 = "pmllm_evidence-<tag>", D-14 = "pmllm_confidence-<tag>".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from budget_curve import run_stats  # noqa: E402

GROUPS = {"full": "{t}", "D-13": "pmllm_evidence-{t}", "D-14": "pmllm_confidence-{t}"}
METRICS = ("brier", "verdicts", "auc")
BETTER = {"brier": "lower", "verdicts": "higher", "auc": "higher"}


# ---------------------------------------------------------------- statistics
def wilcoxon_exact(diffs: List[float]) -> Tuple[Optional[float], int]:
    """Exact two-sided Wilcoxon signed-rank p-value. Zeros dropped, ties get average ranks; the null
    distribution of the positive rank sum is enumerated over all 2^n sign assignments by dynamic
    programming on doubled ranks (so average ranks stay integers). Returns (p, n_used)."""
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return None, 0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(d[order[j + 1]]) - abs(d[order[i]])) < 1e-12:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    r2 = [int(round(2 * r)) for r in ranks]
    total = sum(r2)
    dist = [0] * (total + 1)
    dist[0] = 1
    for r in r2:
        for s in range(total, r - 1, -1):
            dist[s] += dist[s - r]
    w_plus = sum(r for r, x in zip(r2, d) if x > 0)
    mean = total / 2
    dev = abs(w_plus - mean)
    extreme = sum(c for s, c in enumerate(dist) if abs(s - mean) >= dev - 1e-9)
    return min(1.0, extreme / 2 ** n), n


def bootstrap_ci(diffs: List[float], reps: int = 10000, seed: int = 0) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.mean(rng.choice(diffs) for _ in range(n)) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def holm(ps: List[Optional[float]]) -> List[Optional[float]]:
    idx = sorted((i for i, p in enumerate(ps) if p is not None), key=lambda i: ps[i])
    m, out, run = len(idx), [None] * len(ps), 0.0
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (m - k) * ps[i]))
        out[i] = run
    return out


# ---------------------------------------------------------------- data
def seed_of(run: str) -> int:
    return int(re.search(r"_s(\d+)$", run).group(1))


def claim_metrics(claims_csv: str) -> Dict[Tuple[str, int], Dict[str, float]]:
    """(variant, seed) -> brier, verdicts from the audit's per-claim table."""
    per = defaultdict(list)
    with open(claims_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per[(r["variant"], seed_of(r["run"]))].append(r)
    out = {}
    for key, rows in per.items():
        dec = [r for r in rows if r["local_truth"] in ("holds", "fails")]
        out[key] = {"brier": statistics.mean((float(r["search_conf"]) - (r["local_truth"] == "holds")) ** 2
                                             for r in dec) if dec else None,
                    "verdicts": float(sum(r["status"] in ("confirmed", "refuted") for r in rows))}
    return out


def auc_metrics(runs_root: str, task: str, score_table: str, variants: List[str]) -> Dict[Tuple[str, int], float]:
    table = json.load(open(score_table, encoding="utf-8"))
    optimum = {s: max(r["score"] for r in rows.values() if r.get("status") == "ok")
               for s, rows in table["seeds"].items()}
    out = {}
    for v in variants:
        prefix = f"{task}_agent_reflect_{v}_s"
        for d in sorted(os.listdir(runs_root)):
            if d.startswith(prefix) and d[len(prefix):].isdigit():
                st = run_stats(os.path.join(runs_root, d), optimum, [5])
                if st:
                    out[(v, int(d[len(prefix):]))] = st["auc"]
    return out


def collect(settings: List[List[str]]) -> List[Dict]:
    """One row per (setting, group, seed)."""
    rows = []
    for name, root, claims_csv, table, task, tag in settings:
        variants = {g: pat.format(t=tag) for g, pat in GROUPS.items()}
        cm = claim_metrics(claims_csv)
        am = auc_metrics(root, task, table, list(variants.values()))
        for g, v in variants.items():
            for (vv, seed), auc in sorted(am.items()):
                if vv != v:
                    continue
                c = cm.get((v, seed), {})
                rows.append({"setting": name, "group": g, "seed": seed, "auc": auc,
                             "brier": c.get("brier"), "verdicts": c.get("verdicts")})
    return rows


def compare(rows: List[Dict], control: str, metric: str, setting: Optional[str]) -> Optional[Dict]:
    full = {(r["setting"], r["seed"]): r[metric] for r in rows if r["group"] == "full"}
    ctrl = {(r["setting"], r["seed"]): r[metric] for r in rows if r["group"] == control}
    keys = sorted(k for k in full if k in ctrl and full[k] is not None and ctrl[k] is not None
                  and (setting is None or k[0] == setting))
    if not keys:
        return None
    diffs = [ctrl[k] - full[k] for k in keys]
    p, n_used = wilcoxon_exact(diffs)
    lo, hi = bootstrap_ci(diffs)
    better_full = sum((d > 0) if BETTER[metric] == "lower" else (d < 0) for d in diffs)
    return {"setting": setting or "pooled", "control": control, "metric": metric, "pairs": len(keys),
            "full_mean": statistics.mean(full[k] for k in keys), "control_mean": statistics.mean(ctrl[k] for k in keys),
            "diff": statistics.mean(diffs), "ci_lo": lo, "ci_hi": hi, "full_better": better_full, "p": p,
            "n_nonzero": n_used}


def fmt(x: Optional[float], d: int = 3) -> str:
    return "-" if x is None else f"{x:.{d}f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", nargs=6, action="append", required=True,
                   metavar=("NAME", "RUNS_ROOT", "CLAIMS_CSV", "SCORE_TABLE", "TASK", "TAG"))
    p.add_argument("--out", default="")
    a = p.parse_args()
    rows = collect(a.setting)
    names = [s[0] for s in a.setting]
    res = []
    for metric in METRICS:
        for control in ("D-13", "D-14"):
            for s in names + [None]:
                r = compare(rows, control, metric, s)
                if r:
                    res.append(r)
    pooled = [r for r in res if r["setting"] == "pooled"]
    for r, q in zip(pooled, holm([r["p"] for r in pooled])):
        r["p_holm"] = q
    head = ["metric", "control", "setting", "pairs", "full_mean", "control_mean", "diff", "ci95",
            "full_better", "p", "p_holm"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in res:
        d = 1 if r["metric"] == "verdicts" else 3
        lines.append("| " + " | ".join([
            r["metric"] + f" ({BETTER[r['metric']]} is better)", r["control"], r["setting"], str(r["pairs"]),
            fmt(r["full_mean"], d), fmt(r["control_mean"], d), fmt(r["diff"], d),
            f"[{fmt(r['ci_lo'], d)}, {fmt(r['ci_hi'], d)}]", f"{r['full_better']}/{r['pairs']}",
            fmt(r["p"], 4), fmt(r.get("p_holm"), 4)]) + " |")
    n_runs = defaultdict(int)
    for r in rows:
        n_runs[(r["setting"], r["group"])] += 1
    text = ("runs per setting and group: " + ", ".join(f"{s}/{g}={n}" for (s, g), n in sorted(n_runs.items()))
            + "\ndiff = control - full; p = exact two-sided Wilcoxon signed-rank; p_holm = Holm over the "
              f"{len(pooled)} pooled tests\n\n" + "\n".join(lines))
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out + ".md", "w", encoding="utf-8") as f:
            f.write(text + "\n")
        with open(a.out + "_runs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["setting", "group", "seed", "brier", "verdicts", "auc"])
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
