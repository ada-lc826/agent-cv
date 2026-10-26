"""Train every architecture of the search space once per seed under the proxy schedule and store the
results in one score table (read by runners/table_runner.py via run_search.py --score_table).

    # import results that already exist (same proxy schedule and seed), then train the rest
    python tools/build_score_table.py --config configs/cls_resnet50_cifar100_c93.yaml --seeds 0 1 2 --jobs 4 \\
        --import "runs_b/*/trials/*" --import "runs/proxy_check/proxy_e60/*"
    # several machines: each trains a share, then merge the tables
    python tools/build_score_table.py --config ... --seeds 0 1 2 --jobs 4 --shard 1/3 --out runs/score_table_1
    python tools/build_score_table.py --merge runs/score_table_*/score_table.json --out runs/score_table

Re-running the same command resumes: finished trainings are cached (runner result.json) and already in
the table. Failed trainings are stored as failures (a search sees them as it would online) but are
retried by the next run of this script, in case the failure was transient.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from visionevoagent.search_space import all_architectures, architecture_signature, from_signature  # noqa: E402

TASK = "classification"


def _meta(rc: dict) -> dict:
    keys = ("base_config", "schedule", "epochs", "warmup_epochs", "batch_size", "lr", "amp")
    return {k: rc.get(k) for k in keys}


def _save(table: dict, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "score_table.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(table, f, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)


def _load(out: str, meta: dict) -> dict:
    path = os.path.join(out, "score_table.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            table = json.load(f)
        if table.get("meta") != meta:
            raise SystemExit(f"{path} was built with {table.get('meta')}, not {meta}")
        return table
    return {"meta": meta, "seeds": {}}


def import_results(patterns, seed_epochs: int, amp: bool, table: dict) -> int:
    """Copy result.json of earlier trainings whose trial_config.py matches the table's proxy schedule
    (max_epochs, AMP) into the table, keyed by the seed written in the config."""
    n = 0
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            rp, cp = os.path.join(d, "result.json"), os.path.join(d, "trial_config.py")
            if not (os.path.exists(rp) and os.path.exists(cp)):
                continue
            cfg = open(cp, encoding="utf-8").read()
            m_seed = re.search(r"randomness = dict\(seed=(\d+)", cfg)
            if (not m_seed or f"max_epochs={seed_epochs}," not in cfg
                    or ("AmpOptimWrapper" in cfg) != bool(amp)):
                continue
            m_sig = re.search(r"([a-z]+__[a-z]+__[a-z]+__[a-z]+)$", os.path.basename(os.path.normpath(d)))
            if not m_sig:
                continue
            sig = architecture_signature(from_signature(m_sig.group(1)))
            res = json.load(open(rp, encoding="utf-8"))
            rows = table["seeds"].setdefault(m_seed.group(1), {})
            cur = rows.get(sig, {})
            if cur.get("status") != "ok":
                rows[sig] = {**res, "source": d}
                n += 1
            elif res.get("status") == "ok" and res.get("score") != cur.get("score"):
                # the table assumes training is deterministic for a fixed seed; report any exception
                print(f"WARNING: {sig} seed {m_seed.group(1)}: {cur.get('score')} ({cur.get('source')}) "
                      f"vs {res.get('score')} ({d})")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--jobs", type=int, default=4, help="trainings running at the same time on the GPU")
    p.add_argument("--import", dest="imports", action="append", default=[],
                   help="glob of earlier trial work dirs (with result.json + trial_config.py)")
    p.add_argument("--shard", default="1/1", help="i/n: train only every n-th missing architecture, offset i-1")
    p.add_argument("--merge", nargs="+", help="score_table.json files to merge into --out (no training)")
    p.add_argument("--out", default="runs/score_table")
    a = p.parse_args()

    if a.merge:
        tables = [json.load(open(f, encoding="utf-8")) for f in a.merge]
        metas = {json.dumps(t["meta"], sort_keys=True) for t in tables}
        if len(metas) != 1:
            raise SystemExit(f"tables were built with different settings: {metas}")
        merged = {"meta": tables[0]["meta"], "seeds": {}}
        for t in tables:
            for s, rows in t["seeds"].items():
                for sig, r in rows.items():
                    cur = merged["seeds"].setdefault(s, {}).get(sig)
                    if cur is None or (cur.get("status") != "ok" and r.get("status") == "ok"):
                        merged["seeds"][s][sig] = r
        _save(merged, a.out)
        for s, rows in sorted(merged["seeds"].items()):
            print(f"seed {s}: {sum(r['status'] == 'ok' for r in rows.values())} ok, "
                  f"{sum(r['status'] != 'ok' for r in rows.values())} failed, of {len(all_architectures(TASK))}")
        return

    from visionevoagent.runners.mmpretrain_runner import MMPretrainRunner, MMPretrainRunnerConfig
    rc = (yaml.safe_load(open(a.config, encoding="utf-8")) or {}).get("runner", {})
    rc = {**rc, "schedule": "proxy"}
    meta = _meta(rc)
    table = _load(a.out, meta)
    if a.imports:
        print(f"imported {import_results(a.imports, rc['epochs'], rc.get('amp', False), table)} results")
        _save(table, a.out)

    i, n = (int(x) for x in a.shard.split("/"))
    sigs = sorted(architecture_signature(x) for x in all_architectures(TASK))
    jobs = []
    for seed in a.seeds:
        rows = table["seeds"].setdefault(str(seed), {})
        missing = [s for s in sigs if rows.get(s, {}).get("status") != "ok"]
        jobs += [(seed, s) for k, s in enumerate(missing) if k % n == i - 1]
    print(f"{len(jobs)} trainings to run (shard {a.shard})", flush=True)
    runners = {seed: MMPretrainRunner(MMPretrainRunnerConfig(**{**rc, "seed": seed,
                                                               "work_root": os.path.join(a.out, f"seed{seed}")}))
               for seed in a.seeds}
    lock = threading.Lock()

    def work(job):
        seed, sig = job
        res = runners[seed].run(from_signature(sig), sig)
        with lock:
            table["seeds"][str(seed)][sig] = dict(res.__dict__)
            _save(table, a.out)
        print(f"seed {seed} {sig}: {res.status} {res.score} ({res.duration_s / 60:.1f} min)", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        list(ex.map(work, jobs))
    for seed in a.seeds:
        rows = table["seeds"][str(seed)]
        print(f"seed {seed}: {sum(r['status'] == 'ok' for r in rows.values())} ok, "
              f"{sum(r['status'] != 'ok' for r in rows.values())} failed, of {len(sigs)}")


if __name__ == "__main__":
    main()
