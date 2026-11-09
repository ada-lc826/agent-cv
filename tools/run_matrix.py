"""Run a whole experiment matrix (methods x settings x seeds) against a proxy score table, several
searches at a time. Finished runs (search_summary.json with stop_reason != "running") are skipped,
so re-running the same command resumes. No GPU is needed: every trial is answered from the table.

    python tools/run_matrix.py --config configs/cls_resnet50_cifar100_c93.yaml \\
        --score_table runs/score_table/score_table.json --set main ablation --seeds 0 1 2 \\
        --max_rounds 30 --parallel 6 --runs_root runs_main
    python tools/run_matrix.py ... --list          # show the jobs and their status only

Sets (final_experiment_plan sections 4.1-4.2):
  main       random reg_evo tpe genius evoprompting llmatic lapt nader agent agent_reflect
  ablation   agent_reflect with D-1 ... D-12 (D-5 = reflect every 2 and 4 rounds)
  thinking   the 7 LLM methods with --thinking on
  nader_orig NADER with its original 90%-of-parent reflection threshold
  nb201_ablation  agent_reflect with D-9 random / D-10 no_update / D-11 / D-12 (NAS-Bench-201, 2026-10-30;
             D-8 needs hand-written principles for the space and is not run there)
  transfer   agent_reflect started from random principles, or from the principles that the agent_reflect
             run of the same seed learned on another task (--transfer_from <runs dir of that task>)

NAS-Bench-201 (2026-10-30): --task nb201_cifar10 | nb201_cifar100 | nb201_imagenet16 with the matching
config and table; --table_seeds 3 lets search seeds 0..4 read table seeds seed % 3.

    python tools/run_matrix.py --task nb201_cifar100 --config configs/nb201_cifar100.yaml \
        --score_table runs/score_table_nb201/nb201_cifar100.json --set main nb201_ablation \
        --seeds 0 1 2 3 4 --table_seeds 3 --max_rounds 50 --parallel 6 --runs_root runs_nb201
    python tools/run_matrix.py --task nb201_cifar100 ... --set transfer --transfer_from runs_nb201 \
        --source_task nb201_cifar10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM_METHODS = ["genius", "evoprompting", "llmatic", "lapt", "nader", "agent", "agent_reflect"]
SETS = {
    "main": [(m, []) for m in ["random", "reg_evo", "tpe", "genius", "evoprompting", "llmatic", "lapt", "nader",
                               "agent", "agent_reflect"]],
    "ablation": [("agent_reflect", f) for f in (
        ["--no_failure_memory"],                 # D-1
        ["--no_principle_memory"],               # D-2
        ["--mutation_budget", "4"],              # D-3
        ["--explore_every", "0"],                # D-4
        ["--reflect_every_n", "2"],              # D-5
        ["--reflect_every_n", "4"],              # D-5
        ["--reflection_style", "rule"],          # D-6
        ["--reflection_style", "immediate"],     # D-7
        ["--principle_mode", "fixed"],           # D-8
        ["--principle_mode", "random"],          # D-9
        ["--principle_mode", "no_update"],       # D-10
        ["--selection", "llm"],                  # D-11
        ["--reflection_version", "v3"],          # D-12
    )],
    "thinking": [(m, ["--thinking", "on"]) for m in LLM_METHODS],
    "nader_orig": [("nader", ["--nader_threshold", "orig"])],
    "nb201_ablation": [("agent_reflect", f) for f in (
        ["--principle_mode", "random"],          # D-9
        ["--principle_mode", "no_update"],       # D-10
        ["--selection", "llm"],                  # D-11
        ["--reflection_version", "v3"],          # D-12
    )],
    "transfer": [("agent_reflect", ["--init_principles", "random"]),
                 ("agent_reflect", ["--init_principles", "{source}"])],
}


def run_dir_of(mode, flags, seed, max_rounds, runs_root, task="classification"):
    sys.path.insert(0, ROOT)
    import run_search  # noqa: E402
    a = run_search.parse_args(["--mode", mode, "--seed", str(seed), "--max_rounds", str(max_rounds)] + flags)
    v = run_search.variant_tag(a)
    return os.path.join(runs_root, f"{task}_{mode}{'_' + v if v else ''}_s{seed}")


def fill(flags, seed, a):
    """{source}: principle memory of the same seed's agent_reflect run on --source_task (transfer set)."""
    src = os.path.join(a.transfer_from or a.runs_root,
                       f"{a.source_task}_agent_reflect_t{a.max_rounds}_s{seed}", "principle_memory.json")
    return [f.replace("{source}", src) for f in flags]


def status(run_dir):
    p = os.path.join(run_dir, "search_summary.json")
    if not os.path.exists(p):
        return "todo"
    try:
        stop = json.load(open(p, encoding="utf-8")).get("stop_reason")
    except (OSError, ValueError):
        return "todo"
    return "done" if stop and stop != "running" else "partial"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--score_table", required=True)
    p.add_argument("--set", nargs="+", default=["main"], choices=sorted(SETS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max_rounds", type=int, default=30)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--runs_root", default="runs_main")
    p.add_argument("--task", default="classification")
    p.add_argument("--table_seeds", type=int, default=0,
                   help="number of training seeds in the table; search seed s reads table seed s %% N (0: same)")
    p.add_argument("--transfer_from", default="", help="runs dir of the source task (transfer set)")
    p.add_argument("--source_task", default="nb201_cifar10")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()

    jobs = []
    for s in a.set:
        for mode, flags in SETS[s]:
            for seed in a.seeds:  # seed-major order would finish whole methods first; interleave seeds instead
                f = fill(flags, seed, a)
                d = run_dir_of(mode, f, seed, a.max_rounds, a.runs_root, a.task)
                jobs.append((s, mode, f, seed, d))
    for s, mode, flags, seed, d in jobs:
        print(f"{status(d):8s} {s:10s} {mode:14s} {' '.join(flags):32s} s{seed}  {d}")
    if a.list:
        return
    todo = [j for j in jobs if status(j[4]) != "done"]
    print(f"{len(todo)} of {len(jobs)} runs to do, {a.parallel} at a time", flush=True)
    os.makedirs(os.path.join(a.runs_root, "logs"), exist_ok=True)

    def work(job):
        s, mode, flags, seed, d = job
        log = os.path.join(a.runs_root, "logs", os.path.basename(d) + ".log")
        cmd = [sys.executable, os.path.join(ROOT, "run_search.py"), "--config", a.config, "--mode", mode,
               "--task", a.task, "--seed", str(seed), "--max_rounds", str(a.max_rounds),
               "--score_table", a.score_table, "--runs_root", a.runs_root, "--log_level", "WARNING"] + flags
        if a.table_seeds:
            cmd += ["--table_seed", str(seed % a.table_seeds)]
        t0 = time.time()
        with open(log, "a", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"{'ok ' if rc == 0 else 'ERR'} {os.path.basename(d)} ({(time.time() - t0) / 60:.1f} min, exit {rc})",
              flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=max(1, a.parallel)) as ex:
        codes = list(ex.map(work, todo))
    print(f"finished: {sum(c == 0 for c in codes)} ok, {sum(c != 0 for c in codes)} failed")


if __name__ == "__main__":
    main()
