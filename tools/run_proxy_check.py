"""Train the proxy-check architectures under the proxy schedule and the full (official) recipe.

Feeds tools/proxy_reliability.py. Every (architecture, schedule) pair is one runner call, so a finished
pair is never retrained (result.json cache) and the script can simply be re-run after an interruption.
Several trainings can share one GPU (--jobs); CIFAR-100 ResNet-50 at batch 128 uses a small part of 48 GB.

    python tools/proxy_reliability.py sample --n 16 --seed 0 > runs/proxy_check/archs.txt
    python tools/run_proxy_check.py --config configs/cls_resnet50_cifar100_c93.yaml \
        --archs runs/proxy_check/archs.txt --schedule both --jobs 4
    python tools/proxy_reliability.py analyze --csv runs/proxy_check/scores.csv

Timing only (one architecture, proxy schedule):
    python tools/run_proxy_check.py --config ... --archs runs/proxy_check/archs.txt --schedule proxy --only 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from visionevoagent.runners.mmpretrain_runner import MMPretrainRunner, MMPretrainRunnerConfig  # noqa: E402
from visionevoagent.search_space import from_signature  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--archs", required=True, help="one signature per line (tools/proxy_reliability.py sample)")
    p.add_argument("--schedule", choices=("proxy", "full", "both"), default="both")
    p.add_argument("--jobs", type=int, default=1, help="trainings running at the same time on the GPU")
    p.add_argument("--only", type=int, default=0, help="use only the first N architectures (timing runs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full_amp", action="store_true",
                   help="mixed precision for the full recipe too (faster; the reproduction used fp32)")
    p.add_argument("--out", default="runs/proxy_check")
    a = p.parse_args()

    rc = (yaml.safe_load(open(a.config, encoding="utf-8")) or {}).get("runner", {})
    sigs = [s.strip() for s in open(a.archs, encoding="utf-8") if s.strip() and "__" in s]
    if a.only:
        sigs = sigs[: a.only]
    schedules = ["proxy", "full"] if a.schedule == "both" else [a.schedule]
    runners = {
        "proxy": MMPretrainRunner(MMPretrainRunnerConfig(**{**rc, "seed": a.seed, "schedule": "proxy",
                                                            "work_root": os.path.join(a.out, "proxy")})),
        "full": MMPretrainRunner(MMPretrainRunnerConfig(**{**rc, "seed": a.seed, "schedule": "official",
                                                           "amp": a.full_amp, "timeout_s": 24 * 3600,
                                                           "work_root": os.path.join(a.out, "full")})),
    }
    jobs = [(s, sch) for sch in schedules for s in sigs]  # all proxies first: they finish early

    def run(job):
        sig, sch = job
        res = runners[sch].run(from_signature(sig), sig)
        print(f"{sch:5s} {sig}: {res.status} {res.score} ({res.duration_s / 60:.1f} min)", flush=True)
        return sig, sch, res

    results = {}
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for sig, sch, res in ex.map(run, jobs):
            results[(sig, sch)] = res

    rows = []
    for s in sigs:
        pr, fu = results.get((s, "proxy")), results.get((s, "full"))
        rows.append({"signature": s,
                     "proxy": pr.score if pr and pr.status == "ok" else "",
                     "full": fu.score if fu and fu.status == "ok" else "",
                     "proxy_min": round(pr.duration_s / 60, 1) if pr else "",
                     "full_min": round(fu.duration_s / 60, 1) if fu else "",
                     "params_m": (pr or fu).params if (pr or fu) else ""})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    failed = [f"{k[1]} {k[0]}: {v.failure_reason}" for k, v in results.items() if v.status != "ok"]
    with open(os.path.join(a.out, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"jobs": a.jobs, "failed": failed, "rows": rows}, f, indent=2)
    print(f"wrote {os.path.join(a.out, 'scores.csv')}; failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
