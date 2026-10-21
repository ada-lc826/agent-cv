"""Fixes after the B run (2026-10-21): the Reflector is asked again when it proposes tested architectures
(unusable ones are dropped and recorded on the last attempt); the Design Agent gets a positive list of
untried candidates; the calibration has a direction criterion. Offline, uses the mock provider.

Usage (repo root): python tests/test_b_fixes.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))

from principle_calibration import predicted_sign  # noqa: E402
from visionevoagent.agent_client import AgentClient, AgentClientConfig  # noqa: E402
from visionevoagent.agents.reflector import ReflectionContext, ReflectorAgent  # noqa: E402
from visionevoagent.prompts import build_design_prompt  # noqa: E402
from visionevoagent.search_space import BASELINE, architecture_signature, get_search_space  # noqa: E402

TASK = "classification"
ECA = {**BASELINE, "attention": "eca"}
ECA_W = {**ECA, "fusion": "weighted"}
ECA_E = {**ECA, "target_stage": "early"}
info = lambda a: {"architecture": a, "score": 78.5, "params": 23.7}  # noqa: E731
tried = sorted(architecture_signature(a) for a in (BASELINE, ECA, ECA_W))


def reflection(hyps):
    return {"effective_patterns": [], "ineffective_patterns": [], "synergies": [], "new_principles": [],
            "next_search_direction": [], "data_sufficiency": "limited",
            "untested_combinations": [{"architecture": h, "rationale": "r", "principle_ids": [], "predicted_delta": 0.3}
                                      for h in hyps]}


ctx = ReflectionContext(task=TASK, metric="top1", baseline=info(BASELINE), best=info(ECA), history=[],
                        new_trials=[], failed=[], noise_level=0.25, previous_principles=[],
                        propose_hypotheses=True, tried=tried, computed_evidence=True, evidence_table=[])

# 1. a tested hypothesis -> asked again; the second answer is used
calls = []
agent = ReflectorAgent(AgentClient(AgentClientConfig(provider="mock")))
answers = [reflection([ECA_W, ECA_E]), reflection([ECA_E])]
agent._mock = lambda c: (calls.append(1), answers[len(calls) - 1])[1]
out = agent.reflect(ctx)
assert len(calls) == 2 and [architecture_signature(h["architecture"]) for h in out["untested_combinations"]] == \
    [architecture_signature(ECA_E)], out
assert "rejected_hypotheses" not in out

# 2. still unusable on the last attempt -> keep the reflection, drop and record the unusable ones
calls = []
answers = [reflection([ECA_W, ECA_E]), reflection([ECA_W, ECA_E])]
out = agent.reflect(ctx)
assert len(calls) == 2 and len(out["untested_combinations"]) == 1, out
assert out["rejected_hypotheses"] == [{"signature": architecture_signature(ECA_W), "reason": "already tried"}]

# 3. design prompt: positive candidate list
p = build_design_prompt(task=TASK, metric="top1", space=get_search_space(TASK), baseline=info(BASELINE),
                        history=[], best=info(ECA), failed=[], search_mode="exploit", mutation_budget=1,
                        noise_level=0.25, reflection_context={}, show_history=False, tried=tried,
                        candidates=["se__standard__none__late"])
assert "UNTRIED CANDIDATES" in p and "se__standard__none__late" in p

# 4. direction criterion of the calibration
c = {"condition": [{"dim": "fusion", "value": "weighted"}], "reference": [], "direction": "increase", "min_effect": 0.25}
assert predicted_sign(c, ECA, ECA_W) == 1            # parent -> child adds weighted: predicts a rise
assert predicted_sign(c, ECA_W, ECA) == -1           # removes it: predicts a fall
assert predicted_sign(c, ECA, ECA_E) is None         # changes another dimension: not applicable
assert predicted_sign(c, BASELINE, {**ECA_W}) is None  # confounded (attention changed too)
c2 = {"condition": [{"dim": "attention", "value": "cbam"}], "reference": [{"dim": "attention", "value": "eca"}],
      "direction": "decrease", "min_effect": 0.25}
assert predicted_sign(c2, ECA, {**ECA, "attention": "cbam"}) == -1
assert predicted_sign(c2, {**ECA, "attention": "se"}, {**ECA, "attention": "cbam"}) is None
print("B-run fixes: all checks passed")
