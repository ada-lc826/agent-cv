"""Reflection v2 (2026-10-20): the Design Agent sees the reflection instead of the raw history, the
Reflector's untested combinations may differ from the best in 2 dimensions, and non-neutral evidence moves
the confidence with eta 0.35 and a strength floor of 0.5. Offline, uses the mock provider.

Usage (repo root): python tests/test_reflection_v2.py
"""
import os
import tempfile
import sys

sys.path.insert(0, os.getcwd())

from visionevoagent.agent_client import AgentClient, AgentClientConfig  # noqa: E402
from visionevoagent.agents.design_agent import DesignAgent, DesignContext  # noqa: E402
from visionevoagent.memory import ArchitectureMemory  # noqa: E402
from visionevoagent.prompts import build_design_prompt  # noqa: E402
from visionevoagent.schemas import validate_reflection  # noqa: E402
from visionevoagent.search_space import BASELINE, architecture_signature, get_search_space  # noqa: E402

TASK = "classification"
ECA = {**BASELINE, "attention": "eca"}
TWO = {**ECA, "fusion": "weighted", "target_stage": "early"}       # 2 dims from ECA
THREE = {**TWO, "conv": "depthwise"}                                # 3 dims from ECA
info = lambda a: {"architecture": a, "score": 78.0, "params": 23.7}  # noqa: E731
tried = {architecture_signature(BASELINE), architecture_signature(ECA)}
history = [{"round": 1, "architecture": ECA, "status": "ok", "score": 78.5, "delta_vs_baseline": 0.5}]

# prompt: reflection_only hides the history table but keeps the do-not-repeat list and the hypotheses
kw = dict(task=TASK, metric="top1", space=get_search_space(TASK), baseline=info(BASELINE), history=history,
          best=info(ECA), failed=[], search_mode="exploit", mutation_budget=1, noise_level=0.25,
          reflection_context={"design_principles": []})
full = build_design_prompt(**kw)
only = build_design_prompt(**kw, show_history=False, tried=sorted(tried),
                           hypotheses=[{"architecture": TWO, "rationale": "r", "principle_ids": []}])
assert "HISTORY:" in full and "78.50" in full
assert "HISTORY:" not in only and "78.50" not in only, only
assert "TRIED ARCHITECTURES" in only and architecture_signature(ECA) in only and "HYPOTHESES_TO_TEST" in only

# Design Agent: a 2-dim change is legal only when it is one of the hypotheses
agent = DesignAgent(AgentClient(AgentClientConfig(provider="mock")), seed=0)


def ctx(hyps):
    return DesignContext(task=TASK, metric="top1", baseline=info(BASELINE), best=info(ECA), history=[],
                         tried=tried, failed=set(), search_mode="exploit", mutation_budget=1, noise_level=0.25,
                         reflection_context={"design_principles": []}, round=2, show_history=False,
                         hypotheses=hyps)


hyp = [{"architecture": TWO, "rationale": "weighted + early never tried together", "principle_ids": []}]
arch, note, _ = agent.propose(ctx(hyp))
assert architecture_signature(arch) == architecture_signature(TWO) and note.startswith("llm[hypothesis]"), note
agent._mock = lambda c, attempt: {"architecture": TWO, "rationale": "test", "cited_principles": []}
rec = []
arch, note, _ = agent.propose(ctx(None), record=rec)                # same answer without the hypothesis
assert rec == ["invalid"] * 3 and note.startswith("fallback"), (rec, note)
agent._mock = lambda c, attempt: {"architecture": THREE, "rationale": "test", "cited_principles": []}
rec = []
agent.propose(ctx([{"architecture": THREE, "rationale": "r", "principle_ids": []}]), record=rec)
assert rec == ["invalid"] * 3, rec                                  # 3 dims is too far even as a hypothesis

# schema: v2 reflections must carry 1-3 valid untested combinations
base = {"effective_patterns": [], "ineffective_patterns": [], "synergies": [], "principle_updates": [],
        "new_principles": [], "next_search_direction": [], "data_sufficiency": "limited"}
assert validate_reflection(base, TASK) is None
assert validate_reflection(base, TASK, with_hypotheses=True) is not None
assert validate_reflection({**base, "untested_combinations": hyp}, TASK, with_hypotheses=True) is None
bad = [{"architecture": {**TWO, "attention": "transformer"}, "rationale": "", "principle_ids": []}]
assert validate_reflection({**base, "untested_combinations": bad}, TASK, with_hypotheses=True) is not None

# confidence: v1 step 0.2 * 0.2 * 0.5 = 0.02; v2 floors the strength at 0.5 -> 0.35 * 0.5 * 0.5 = 0.0875
mem = ArchitectureMemory(tempfile.mkdtemp())
fields = {"if": "attention=eca", "scope": "s", "then": "t", "because": "b", "disproved_if": "d"}
p1, p2 = mem.add_principle(fields, TASK, 1), mem.add_principle({**fields, "if": "fusion=weighted"}, TASK, 1)
mem.update_principle(p1.id, "supports", 0.2, 2)
mem.update_principle(p2.id, "supports", 0.2, 2, eta=0.35, floor=0.5)
assert abs(p1.confidence - 0.52) < 1e-6 and abs(p2.confidence - 0.5875) < 1e-6, (p1.confidence, p2.confidence)
mem.update_principle(p2.id, "neutral", 0.0, 3, eta=0.35, floor=0.5)   # neutral never moves it
assert abs(p2.confidence - 0.5875) < 1e-6
print("reflection v2: all checks passed")
