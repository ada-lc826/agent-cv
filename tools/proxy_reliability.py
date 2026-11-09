"""Is the short proxy training a reliable stand-in for full training? (precondition for every search run)

Advisor (2026-10-13): use 12-16 architectures rather than 8, report the correlation with its
uncertainty, and above all check whether the proxy's top picks keep their lead after full training.

1) Pick the architectures (seeded; the baseline is always included as the reference point):
    python tools/proxy_reliability.py sample --n 16 --seed 0 > runs/proxy_check/archs.txt
2) Train each one with the proxy recipe and with the full recipe, then fill a CSV with columns
   signature,proxy,full  (one row per architecture; mean over seeds if several were run).
3) Analyse:
    python tools/proxy_reliability.py analyze --csv runs/proxy_check/scores.csv --top_k 3

Reports Kendall tau-b and Spearman rho with bootstrap 95% intervals, the overlap of the proxy top-k
with the full-training top-k, the full-training rank of the proxy's best pick, and its regret (best
full score minus the full score of the proxy's pick). Decision rule from the research plan: tau > 0.6
and the proxy top-k keeps its lead; otherwise lengthen the proxy before any large search.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visionevoagent.search_space import BASELINE, all_architectures, architecture_signature  # noqa: E402


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    n, conc, disc, tx, ty = len(x), 0, 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    denom = ((conc + disc + tx) * (conc + disc + ty)) ** 0.5
    return (conc - disc) / denom if denom else float("nan")


def _ranks(v: Sequence[float]) -> List[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx and sy else float("nan")


def bootstrap_ci(x, y, fn, n_boot=5000, seed=0):
    rng, n, vals = random.Random(seed), len(x), []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = fn([x[i] for i in idx], [y[i] for i in idx])
        if v == v:  # skip NaN (all-tied resamples)
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def cmd_sample(a) -> None:
    rng = random.Random(f"proxy-check-{a.seed}")
    base = architecture_signature(BASELINE)
    others = [architecture_signature(x) for x in all_architectures(a.task)]
    others = sorted(s for s in set(others) if s != base)
    print(base)
    for s in rng.sample(others, a.n - 1):
        print(s)


def cmd_analyze(a) -> None:
    rows: List[Dict[str, str]] = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    skipped = [r["signature"] for r in rows if not r.get("proxy") or not r.get("full")]
    rows = [r for r in rows if r.get("proxy") and r.get("full")]
    if skipped:
        print(f"skipped (missing proxy or full score): {', '.join(skipped)}")
    sig = [r["signature"] for r in rows]
    px, fx = [float(r["proxy"]) for r in rows], [float(r["full"]) for r in rows]
    n, k = len(rows), min(a.top_k, len(rows))
    tau, rho = kendall_tau_b(px, fx), spearman(px, fx)
    tau_ci, rho_ci = bootstrap_ci(px, fx, kendall_tau_b), bootstrap_ci(px, fx, spearman)
    top_p = sorted(range(n), key=lambda i: -px[i])[:k]
    top_f = sorted(range(n), key=lambda i: -fx[i])[:k]
    full_rank = {i: r + 1 for r, i in enumerate(sorted(range(n), key=lambda i: -fx[i]))}
    pick = top_p[0]
    print(f"n = {n} architectures")
    print(f"Kendall tau-b = {tau:.3f}  (bootstrap 95% CI {tau_ci[0]:.3f} .. {tau_ci[1]:.3f})")
    print(f"Spearman rho  = {rho:.3f}  (bootstrap 95% CI {rho_ci[0]:.3f} .. {rho_ci[1]:.3f})")
    print(f"top-{k} overlap (proxy vs full) = {len(set(top_p) & set(top_f))}/{k}")
    print(f"proxy top-{k} -> full-training ranks: " + ", ".join(f"{sig[i]} #{full_rank[i]}" for i in top_p))
    print(f"proxy's best pick {sig[pick]}: full rank #{full_rank[pick]}, regret {max(fx) - fx[pick]:.2f}")
    ok = tau > 0.6 and full_rank[pick] <= k
    print("decision:", "proxy usable" if ok else "lengthen the proxy before large-scale search")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=16)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--task", default="classification")
    z = sub.add_parser("analyze")
    z.add_argument("--csv", required=True)
    z.add_argument("--top_k", type=int, default=3)
    a = p.parse_args()
    cmd_sample(a) if a.cmd == "sample" else cmd_analyze(a)


if __name__ == "__main__":
    main()
