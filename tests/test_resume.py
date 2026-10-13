"""Resume test (dry run): a run interrupted after 4 rounds and resumed must equal an uninterrupted run.

Usage (repo root, PYTHONPATH=.): python tests/test_resume.py [mode ...]
"""
import json
import logging
import os
import shutil
import sys

sys.path.insert(0, os.getcwd())
logging.basicConfig(level=logging.ERROR)

import run_search  # noqa: E402
from visionevoagent.search import SearchEngine  # noqa: E402

ROOT = "/tmp/vea_resume"
shutil.rmtree(ROOT, ignore_errors=True)
modes = sys.argv[1:] or ["random", "reg_evo", "genius", "evoprompting", "llmatic", "lapt", "nader",
                         "agent", "agent_reflect"]


def sigs(d):
    return [t["signature"] for t in json.load(open(os.path.join(d, "trial_memory.json")))]


orig_run = SearchEngine.run
for mode in modes:
    full, part = f"{ROOT}/{mode}_full", f"{ROOT}/{mode}_part"
    base = ["--config", "configs/cls_resnet50_cifar100.yaml", "--mode", mode, "--dry_run", "--seed", "3",
            "--log_level", "ERROR"]
    run_search.main(base + ["--run_dir", full])

    # interrupted run: stop after round 4 by shrinking the budget for the first engine only
    def short_run(self):
        self.cfg.max_rounds = 4
        return orig_run(self)
    SearchEngine.run = short_run
    run_search.main(base + ["--run_dir", part])
    SearchEngine.run = orig_run
    run_search.main(base + ["--run_dir", part])

    a, b = sigs(full), sigs(part)
    status = "OK " if a == b else "MISMATCH"
    print(f"{status} {mode:14s} {len(a)} trials", "" if a == b else f"\n  full={a}\n  part={b}")
