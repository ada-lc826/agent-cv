"""Do any two configurations of the search space build the same network? (advisor, 2026-10-13)

Signature-level de-duplication only knows the rule in search_space.canonicalize(). This tool builds
all raw combinations (no canonicalisation) with the real MMPretrain config (CPU, one subprocess) and
fingerprints the resulting module graph: the module tree with every layer's hyper-parameters
(str(model)) plus the names and shapes of all parameters and buffers. Two configurations with the same
fingerprint are the same network. The de-duplication is correct iff "same signature" and "same
fingerprint" define exactly the same groups. It also reports the size of the effective search space:
all distinct networks minus the baseline itself.

    python tools/check_equivalence.py --config configs/cls_resnet50_cifar100_c93.yaml --out runs/equivalence
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import visionevoagent.runners.mmpretrain_runner as mr  # noqa: E402
from visionevoagent.search_space import (BASELINE, DIMENSIONS, architecture_signature,  # noqa: E402
                                         get_search_space)

CODE = r"""
import sys, json, hashlib
from mmengine.config import Config
from mmengine.registry import init_default_scope
import visionevoagent.plugins.mmpretrain_evo
from mmpretrain.registry import MODELS
init_default_scope('mmpretrain')
out = {}
for sig, path in json.loads(open(sys.argv[1]).read()).items():
    m = MODELS.build(Config.fromfile(path).model)
    shapes = [(n, tuple(p.shape)) for n, p in m.named_parameters()] + \
             [(n, tuple(b.shape)) for n, b in m.named_buffers()]
    text = repr(m) + json.dumps(shapes)
    out[sig] = {"fingerprint": hashlib.sha256(text.encode()).hexdigest(),
                "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 3)}
    del m
print(json.dumps(out))
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--task", default="classification")
    p.add_argument("--out", default="runs/equivalence")
    a = p.parse_args()
    import yaml
    yr = (yaml.safe_load(open(a.config, encoding="utf-8")) or {}).get("runner", {})
    os.makedirs(a.out, exist_ok=True)
    runner = mr.MMPretrainRunner(mr.MMPretrainRunnerConfig(**{**yr, "work_root": a.out}))
    mr.canonicalize = lambda arch: dict(arch)  # build exactly what each raw combination asks for

    paths, sig_of = {}, {}
    cfg_dir = os.path.join(os.path.abspath(a.out), "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    space = get_search_space(a.task)
    for combo in itertools.product(*(space[d] for d in DIMENSIONS)):
        arch = dict(zip(DIMENSIONS, combo))
        raw = "__".join(combo)
        sig_of[raw] = architecture_signature(arch)
        paths[raw] = os.path.join(cfg_dir, f"{raw}.py")
        with open(paths[raw], "w", encoding="utf-8") as f:
            f.write(runner.build_config(arch))
    lst = os.path.join(cfg_dir, "_paths.json")
    with open(lst, "w", encoding="utf-8") as f:
        json.dump(paths, f)

    env = runner._env()
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run([runner.cfg.python, "-c", CODE, lst], cwd=runner.cfg.mmpretrain_root, env=env,
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        sys.exit(proc.stderr[-3000:])
    res = json.loads(proc.stdout.strip().splitlines()[-1])

    by_fp, by_sig = defaultdict(set), defaultdict(set)
    for raw, r in res.items():
        by_fp[r["fingerprint"]].add(raw)
        by_sig[sig_of[raw]].add(raw)
    sig_groups = sorted(sorted(g) for g in by_sig.values())
    # same signature but different networks -> de-duplication merges things that differ
    over_merged = [sorted(g) for g in by_sig.values() if len({res[r]["fingerprint"] for r in g}) > 1]
    # same network but different signatures -> de-duplication misses an equivalence
    missed = [sorted(g) for g in by_fp.values() if len({sig_of[r] for r in g}) > 1]
    base_raw = "__".join(BASELINE[d] for d in DIMENSIONS)
    base_fp = res[base_raw]["fingerprint"]
    same_as_base = sorted(sig_of[r] for r in by_fp[base_fp] if sig_of[r] != architecture_signature(BASELINE))
    report = {
        "n_raw_combinations": len(res),
        "n_signatures": len(by_sig),
        "n_distinct_networks": len(by_fp),
        "dedup_correct": not over_merged and not missed,
        "over_merged_signatures": over_merged,
        "missed_equivalences": missed,
        "baseline": architecture_signature(BASELINE),
        "signatures_identical_to_baseline": sorted(set(same_as_base)),
        "n_effective_candidates_excluding_baseline": len(by_fp) - 1,
        "groups_merged_by_signature": [g for g in sig_groups if len(g) > 1],
        "params_m": {sig_of[r]: res[r]["params_m"] for r in sorted(res)},
    }
    with open(os.path.join(a.out, "equivalence.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"{len(res)} raw combinations -> {len(by_sig)} signatures -> {len(by_fp)} distinct networks")
    print("de-duplication correct:", report["dedup_correct"])
    print("over-merged (same signature, different network):", over_merged or "none")
    print("missed (different signatures, same network):", missed or "none")
    print("identical to the baseline:", report["signatures_identical_to_baseline"] or "none")
    print(f"effective search space (excluding the baseline): {len(by_fp) - 1} candidates")


if __name__ == "__main__":
    main()
