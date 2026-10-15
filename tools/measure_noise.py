"""Train the baseline under the proxy schedule with several seeds and report the score std.

Use the result as `noise_level` / `min_improvement` in the search config.
    python tools/measure_noise.py --config configs/cls_resnet50_cifar100.yaml --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from visionevoagent.runners.mmpretrain_runner import MMPretrainRunner, MMPretrainRunnerConfig  # noqa: E402
from visionevoagent.search_space import BASELINE  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=0, help="proxy length to measure (default: the yaml's)")
    p.add_argument("--out", default="runs/noise")
    a = p.parse_args()
    with open(a.config, encoding="utf-8") as f:
        rcfg = (yaml.safe_load(f) or {}).get("runner", {})
    if a.epochs:
        rcfg = {**rcfg, "epochs": a.epochs}
        a.out = f"{a.out}_e{a.epochs}"
    scores = []
    for s in a.seeds:
        runner = MMPretrainRunner(MMPretrainRunnerConfig(**{**rcfg, "seed": s, "work_root": a.out}))
        res = runner.run(BASELINE, f"baseline_seed{s}")
        print(f"seed {s}: {res.status} {res.score}")
        if res.status == "ok":
            scores.append(res.score)
    out = {"scores": scores, "mean": statistics.mean(scores) if scores else None,
           "std": statistics.stdev(scores) if len(scores) > 1 else None}
    print(json.dumps(out, indent=2))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "noise.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
