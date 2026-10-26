"""Hypothesis-guided framework (2026-10-26): selection terms, parent -> child pair evidence, candidate rules.
Offline, uses the mock provider.

Usage (repo root): python tests/test_selection.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.getcwd())

from visionevoagent.agent_client import AgentClient, AgentClientConfig  # noqa: E402
from visionevoagent.agents.design_agent import DesignAgent, DesignContext  # noqa: E402
from visionevoagent.evidence import fit, pair_verdict, predicted_sign  # noqa: E402
from visionevoagent.memory import ArchitectureMemory, Trial  # noqa: E402
from visionevoagent.search_space import BASELINE, architecture_signature as sig  # noqa: E402
from visionevoagent.selection import choose, combine, expected_improvement, score_candidates  # noqa: E402

TASK, SD = "classification", 0.25
ECA = {**BASELINE, "attention": "eca"}
SE = {**BASELINE, "attention": "se"}
ECA_W = {**ECA, "fusion": "weighted"}
row = lambda a, s: {"architecture": a, "status": "ok", "score": s}  # noqa: E731
base = row(BASELINE, 78.0)
claim_w = {"condition": [{"dim": "fusion", "value": "weighted"}], "reference": [], "direction": "increase",
           "min_effect": 0.25}

# 1. expected improvement and the LLM prior
assert abs(expected_improvement(0.0, 1.0, 0.0) - 0.3989) < 1e-3
assert expected_improvement(1.0, 0.0, 0.5) == 0.5
mu, sd = combine(0.0, 1.0, 1.0, 1.0)
assert abs(mu - 0.5) < 1e-9 and abs(sd - 0.7071) < 1e-3          # equal weights
mu, sd = combine(0.4, 0.1, 2.0, 0.75)
assert abs(mu - 0.4) < 0.05                                        # data dominate once precise

# 2. verification value: training an architecture that has never been seen sharpens an untested claim
m = fit([row(ECA, 78.5), row(SE, 78.2)], base, TASK, SD, claims=[claim_w])
assert m.verification_value({**BASELINE, "fusion": "weighted"}, [claim_w]) > 0.3
assert m.verification_value(SE, [claim_w]) < 0.05                  # tells nothing about fusion

# 3. selection prefers the informative candidate in an explore round, the promising one in exploit
cands = [{"architecture": {**SE, "target_stage": "early"}, "parent": sig(SE), "expected_improvement": 0.3},
         {"architecture": ECA_W, "parent": sig(ECA), "expected_improvement": 0.0}]
lib = [{"id": "p1", "confidence": 0.5, "claim": claim_w}]
deltas = {sig(BASELINE): 0.0, sig(ECA): 0.5, sig(SE): 0.2}
ex = score_candidates(cands, m, lib, deltas, 0.5, SD, "explore")
assert ex[1]["verification"] > ex[0]["verification"] and ex[1]["principles_touched"] == ["p1"]
assert choose(ex, "guided") == 1 and choose(ex, "llm") == 0

# 4. pair evidence: clean tests only; support / counterexample / inconclusive
assert predicted_sign(claim_w, ECA, ECA_W) == 1 and predicted_sign(claim_w, SE, ECA_W) is None
assert pair_verdict(claim_w, 1, 0.4, SD) == "support"
assert pair_verdict(claim_w, 1, -0.3, SD) == "counterexample"
assert pair_verdict(claim_w, 1, 0.1, SD) == "inconclusive"
mem = ArchitectureMemory(tempfile.mkdtemp())
f = {"if": "fusion=weighted", "scope": "s", "then": "t", "because": "b", "disproved_if": "d"}
p = mem.add_principle(f, TASK, 1, claim=claim_w, restate_bonus=False)
t = Trial(2, ECA_W, sig(ECA_W), "ok", 78.1, parent=sig(ECA), delta_vs_parent=-0.4)
mem.record_pair_evidence(t, ECA, SD)
assert p.contradict_count == 1 and p.evidence[-1]["type"] == "counterexample"

# 5. candidate rules: parent must be offered; budget counted from the parent; tried/repeated dropped
agent = DesignAgent(AgentClient(AgentClientConfig(provider="mock")), seed=0)
answer = {"candidates": [
    {"parent": sig(ECA), "architecture": ECA_W, "expected_improvement": 0.2, "rationale": "ok", "cited_principles": []},
    {"parent": sig(ECA), "architecture": ECA_W, "expected_improvement": 0.2, "rationale": "repeat", "cited_principles": []},
    {"parent": "nope", "architecture": SE, "expected_improvement": 0.1, "rationale": "bad parent", "cited_principles": []},
    {"parent": sig(BASELINE), "architecture": ECA, "expected_improvement": 0.1, "rationale": "tried", "cited_principles": []},
    {"parent": sig(BASELINE), "architecture": {**ECA_W, "conv": "depthwise"}, "expected_improvement": 0.1,
     "rationale": "3 dims", "cited_principles": []}]}
agent._mock_candidates = lambda ctx, attempt: answer
ctx = DesignContext(task=TASK, metric="top1", baseline={"architecture": BASELINE, "score": 78.0, "params": 23.7},
                    best={"architecture": ECA, "score": 78.5, "params": 23.7}, history=[],
                    tried={sig(BASELINE), sig(ECA)}, failed=set(), search_mode="exploit", mutation_budget=1,
                    noise_level=SD, reflection_context={"design_principles": []}, round=2, show_history=False,
                    parents=[{"signature": sig(ECA), "architecture": ECA, "score": 78.5, "delta": 0.5},
                             {"signature": sig(BASELINE), "architecture": BASELINE, "score": 78.0, "delta": 0.0}],
                    n_candidates=4)
rec = []
out = agent.propose_candidates(ctx, record=rec)
assert [sig(c["architecture"]) for c in out] == [sig(ECA_W)], out
assert rec == ["ok", "invalid", "invalid", "duplicate", "invalid"], rec
print("hypothesis-guided selection: all checks passed")
