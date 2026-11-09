"""NAS-Bench-201 space (2026-10-30): switching the active space leaves the module-insertion prompts
unchanged, every NB201 phrase replacement matches the prompt it is meant for, and every method runs end to
end in the NB201 space against a (synthetic) score table with the mock LLM. Offline.

Usage (repo root): python tests/test_nb201.py
"""
import json
import os
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, os.getcwd())

from visionevoagent import baseline_prompts, prompts, search_space  # noqa: E402
from visionevoagent.search_space import (BASELINE, DIMENSIONS, activate, all_architectures,  # noqa: E402
                                         architecture_signature)

# 1. modules space by default; its prompt pieces are exactly what the finished runs used
assert search_space.ACTIVE == "modules" and DIMENSIONS == ["attention", "conv", "fusion", "target_stage"]
rows = [{"round": 1, "architecture": {"attention": "eca", "conv": "standard", "fusion": "none", "target_stage": "late"},
         "status": "ok", "score": 78.5, "delta_vs_baseline": 0.4, "params": 23.7, "param_ratio": 1.0}]
h = prompts._fmt_history(rows)
assert h.splitlines()[0] == ("round | attention | conv | fusion | target_stage | status | score | delta_vs_baseline | "
                             "params(M) | param_ratio | note"), h
assert h.splitlines()[1].startswith("1 | eca | standard | none | late | ok | 78.50"), h
assert prompts._sig_format() == "attention__conv__fusion__target_stage"
assert prompts.adapt_text(prompts.DESIGN_SYSTEM) == prompts.DESIGN_SYSTEM
assert prompts._fmt_space(search_space.get_search_space("classification")).startswith("- attention:")

# 2. NB201: every replacement phrase occurs in some prompt text, and the space switches in place
texts = [v for m in (prompts, baseline_prompts) for v in vars(m).values() if isinstance(v, str)]
texts += baseline_prompts.LLMATIC_INSTRUCTIONS
for old, _ in prompts.NB201_PHRASES:
    assert any(old in t for t in texts), f"phrase not found in any prompt: {old[:60]}"
activate("nb201_cifar100")
assert DIMENSIONS[0] == "edge_0_1" and BASELINE["edge_2_3"] == "skip_connect"
assert len(all_architectures("nb201_cifar100")) == 15625
adapted = prompts.adapt_text(prompts.REFLECTOR_SYSTEM_V4)
assert "attention" not in adapted and "edge_0_2" in adapted, adapted
assert "NAS-Bench-201 cell" in prompts._fmt_space(search_space.get_search_space("nb201_cifar100"))
activate("classification")
assert DIMENSIONS[0] == "attention" and len(all_architectures("classification")) == 86

# 3. end to end in the NB201 space (mock LLM, synthetic table)
d = tempfile.mkdtemp()
activate("nb201_cifar10")
rng = random.Random(0)
seeds = {}
for s in range(3):
    seeds[str(s)] = {architecture_signature(a): {"status": "ok", "score": round(80 + 10 * rng.random(), 2),
                                                 "params": round(0.1 + rng.random(), 3), "duration_s": 3000.0}
                     for a in all_architectures("nb201_cifar10")}
table = os.path.join(d, "t.json")
json.dump({"meta": {}, "seeds": seeds}, open(table, "w"))
activate("classification")
modes = ["random", "reg_evo", "tpe", "genius", "evoprompting", "llmatic", "lapt", "nader", "agent", "agent_reflect"]
try:
    import optuna  # noqa: F401
except ImportError:
    modes.remove("tpe")
    print("tpe skipped (optuna not installed here)")
for mode in modes:
    cmd = [sys.executable, "run_search.py", "--config", "configs/nb201_cifar10.yaml", "--task", "nb201_cifar10",
           "--mode", mode, "--seed", "4", "--table_seed", "1", "--max_rounds", "6", "--score_table", table,
           "--provider", "mock", "--runs_root", d, "--log_level", "WARNING"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, (mode, r.stdout[-2000:], r.stderr[-2000:])
    s = json.load(open(os.path.join(d, f"nb201_cifar10_{mode}_t6_s4", "search_summary.json")))
    assert s["stop_reason"] == "max_rounds" and s["n_trials"] == 6, (mode, s["stop_reason"], s["n_trials"])
    assert s["baseline"]["signature"] == "nor_conv_3x3__none__nor_conv_3x3__skip_connect__none__skip_connect"
    assert s["baseline"]["score"] == seeds["1"][s["baseline"]["signature"]]["score"]  # table seed 1
    print(f"{mode:14s} ok  best {s['best']['score']}")
print("nb201: all checks passed")
