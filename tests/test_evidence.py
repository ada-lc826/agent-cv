"""Reflection v3: principle confidence computed from the trials (visionevoagent/evidence.py).
Checks the Bayesian linear model against closed forms and a synthetic ground truth. Offline.

Usage (repo root): python tests/test_evidence.py
"""
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.getcwd())

from visionevoagent.evidence import fit, validate_claim  # noqa: E402
from visionevoagent.memory import ArchitectureMemory  # noqa: E402
from visionevoagent.search_space import BASELINE, all_architectures  # noqa: E402

TASK, SD = "classification", 0.25


def row(arch, score):
    return {"architecture": arch, "status": "ok", "score": score}


def claim(cond, direction="increase", m=SD, ref=()):
    return {"condition": [{"dim": d, "value": v} for d, v in cond],
            "reference": [{"dim": d, "value": v} for d, v in ref], "direction": direction, "min_effect": m}


ECA = {**BASELINE, "attention": "eca"}
base = row(BASELINE, 78.0)

# 1. no data beyond the baseline: the prior decides, P(effect > 0.25 | N(0, 1)) = 1 - Phi(0.25)
m = fit([], base, TASK, SD)
c = m.confidence(claim([("attention", "eca")]))["confidence"]
assert abs(c - (1 - 0.5 * (1 + math.erf(0.25 / math.sqrt(2))))) < 1e-3, c

# 2. flat prior, one paired comparison: effect = y1 - y0, sd = sqrt(2) * noise
m = fit([row(ECA, 78.5)], base, TASK, SD, prior_sd=1e3)
mu, sd = m.effect([("attention", "eca")])
assert abs(mu - 0.5) < 1e-3 and abs(sd - math.sqrt(2) * SD) < 1e-3, (mu, sd)
assert m.confidence(claim([("attention", "eca")]))["n_direct"] == 1

# 3. synthetic ground truth with an interaction: eca +1.0, weighted +0.3, eca x weighted +0.6, depthwise -1.0
TRUE = {("attention", "eca"): 1.0, ("fusion", "weighted"): 0.3, ("conv", "depthwise"): -1.0}
rng = random.Random(0)


def truth(a):
    y = 78.0 + sum(v for (d, val), v in TRUE.items() if a[d] == val)
    return y + (0.6 if a["attention"] == "eca" and a["fusion"] == "weighted" else 0.0)


rows = [row(a, truth(a) + rng.gauss(0, SD)) for a in all_architectures(TASK) if a != BASELINE]
inter = claim([("attention", "eca"), ("fusion", "weighted")], m=1.5)
m = fit(rows, base, TASK, SD, claims=[inter])
for (d, v), eff in TRUE.items():
    mu, sd = m.effect([(d, v)])
    assert abs(mu - eff) < 3 * sd + 0.05, (d, v, mu, sd)
mu, sd = m.effect([("attention", "eca"), ("fusion", "weighted")])
assert abs(mu - 1.9) < 3 * sd + 0.05, (mu, sd)                     # 1.0 + 0.3 + 0.6
assert m.confidence(inter)["confidence"] > 0.9                       # joint effect > 1.5: true
assert m.confidence(claim([("fusion", "weighted")], m=1.0))["confidence"] < 0.05     # weighted alone > 1.0: false
assert m.confidence(claim([("conv", "depthwise")], "decrease"))["confidence"] > 0.95
# comparison against another value instead of the baseline value: eca beats se by > 0.25 (se has no effect)
assert m.confidence(claim([("attention", "eca")], ref=[("attention", "se")]))["confidence"] > 0.95
# the model predicts unseen architectures: leave one out
held = {**BASELINE, "attention": "eca", "fusion": "weighted", "conv": "depthwise", "target_stage": "all"}
m2 = fit([r for r in rows if r["architecture"] != held], base, TASK, SD, claims=[inter])
mu, sd = m2.predict(held)
assert abs(mu - (truth(held) - 78.0)) < 3 * sd + 0.1, (mu, sd)

# 4. claim validation
assert validate_claim(claim([("attention", "eca")]), TASK) is None
assert validate_claim(claim([("target_stage", "late")]), TASK) is not None            # the baseline itself
assert validate_claim(claim([("attention", "eca")], ref=[("conv", "dilated")]), TASK) is not None
assert validate_claim(claim([("attention", "eca"), ("attention", "se")]), TASK) is not None
assert validate_claim(claim([("attention", "transformer")]), TASK) is not None
assert validate_claim({**claim([("attention", "eca")]), "direction": "up"}, TASK) is not None

# 5. memory: confidence recomputed from the data, status follows, restating a claim adds nothing
mem = ArchitectureMemory(tempfile.mkdtemp())
f = {"if": "attention=eca", "scope": "s", "then": "t", "because": "b", "disproved_if": "d"}
p_true = mem.add_principle(f, TASK, 1, claim=claim([("attention", "eca")]), restate_bonus=False)
p_false = mem.add_principle({**f, "if": "dw"}, TASK, 1, claim=claim([("conv", "depthwise")]), restate_bonus=False)
again = mem.add_principle({**f, "because": "other words"}, TASK, 2, claim=claim([("attention", "eca")]),
                          restate_bonus=False)
assert again.id == p_true.id and len(mem.principles) == 2 and p_true.confidence == 0.5
mem.recompute_confidence(m, 3)
assert p_true.status == "confirmed" and p_false.status == "refuted", (p_true.confidence, p_false.confidence)
assert p_true.evidence[-1]["type"] == "computed" and p_true.stats["n_direct"] > 0
print("evidence model: all checks passed")
