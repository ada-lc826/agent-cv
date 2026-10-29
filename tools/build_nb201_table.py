"""NAS-Bench-201 score tables (second set of experiments, 2026-10-30) from Syne Tune's blackbox files
(HuggingFace dataset synetune/blackbox-repository, folder nasbench201; ~788 MB):

  objectives_evaluations.npy  float32 [dataset 3, architecture 15625, seed 3, epoch 200, objective 7]
  hyperparameters.parquet     hp_x0 .. hp_x5 = operation of each edge, in NAS-Bench-201 string order
                              |x0~0|+|x1~0|x2~1|+|x3~0|x4~1|x5~2|  (checked: the best cells give 94.37 /
                              73.50 on CIFAR-10 / -100 and the ResNet cell 93.91 / 70.70 / 44.17, as published)
  metadata.json               objectives: valid_error, train_error, runtime, elapsed_time, latency, flops, params

One table per dataset in the format of tools/build_score_table.py, read by run_search.py --score_table
--task nb201_<dataset>: score = 100 * (1 - error after 200 epochs), one entry per training seed (0, 1, 2),
params in millions, duration_s = elapsed training time (the nominal cost of the trial).

    python tools/build_nb201_table.py --src /root/autodl-tmp/nb201/synetune --out runs/score_table_nb201
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionevoagent.search_space import NB201_EDGES  # noqa: E402

TASKS = {"cifar10": "nb201_cifar10", "cifar100": "nb201_cifar100", "ImageNet16-120": "nb201_imagenet16"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--out", default="runs/score_table_nb201")
    p.add_argument("--epoch", type=int, default=200)
    a = p.parse_args()

    import pyarrow.parquet as pq
    meta = json.load(open(os.path.join(a.src, "metadata.json"), encoding="utf-8"))
    obj = {n: i for i, n in enumerate(meta["objectives_names"])}
    hps = pq.read_table(os.path.join(a.src, "hyperparameters.parquet")).to_pylist()
    sigs = ["__".join(h[f"hp_x{k}"] for k in range(len(NB201_EDGES))) for h in hps]
    ev = np.load(os.path.join(a.src, "objectives_evaluations.npy"), mmap_mode="r")
    fid = list(np.load(os.path.join(a.src, "fidelities_values.npy")))
    e = fid.index(a.epoch)
    os.makedirs(a.out, exist_ok=True)
    for k, name in enumerate(meta["task_names"]):
        block = np.asarray(ev[k, :, :, e, :])                       # [arch, seed, objective]
        table = {"meta": {"source": "synetune/blackbox-repository nasbench201", "dataset": name,
                          "epoch": a.epoch, "score": "100 * (1 - metric_valid_error)",
                          "edges": NB201_EDGES},
                 "seeds": {}}
        for s in range(block.shape[1]):
            table["seeds"][str(s)] = {
                sig: {"status": "ok", "score": round(float(100 * (1 - block[i, s, obj["metric_valid_error"]])), 3),
                      "params": round(float(block[i, s, obj["metric_params"]]), 4),
                      "duration_s": round(float(block[i, s, obj["metric_elapsed_time"]]), 1)}
                for i, sig in enumerate(sigs)}
        path = os.path.join(a.out, f"{TASKS[name]}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(table, f)
        acc = 100 * (1 - block[:, :, obj["metric_valid_error"]].mean(1))
        noise = float(np.median(100 * block[:, :, obj["metric_valid_error"]].std(1, ddof=1)))
        print(f"{path}: {len(sigs)} cells x {block.shape[1]} seeds; best mean {acc.max():.2f}, "
              f"median {np.median(acc):.2f}; median seed sd {noise:.3f} points")


if __name__ == "__main__":
    main()
