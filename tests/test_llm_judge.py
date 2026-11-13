"""Control "who judges the evidence" (2026-11-12): principle_mode llm_evidence (D-13) / llm_confidence (D-14).
The hypothesis-guided framework is unchanged except that the Reflector, not the evidence model, sets each
principle's confidence. Offline, uses the mock provider.

Usage (repo root): python tests/test_llm_judge.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.getcwd())

from visionevoagent.agent_client import AgentClient, AgentClientConfig  # noqa: E402
from visionevoagent.agents import DesignAgent, ReflectorAgent  # noqa: E402
from visionevoagent.controllers.agent_search import (LLM_JUDGE_MODES, PRINCIPLE_MODES, AgentSearchConfig,  # noqa: E402
                                                     AgentSearchController)
from visionevoagent.memory import ArchitectureMemory, Trial  # noqa: E402
from visionevoagent.prompts import REFLECTOR_SYSTEM_V4, REFLECTOR_SYSTEM_V4_JUDGE, build_reflection_prompt  # noqa: E402
from visionevoagent.schemas import reflection_schema, validate_reflection  # noqa: E402
from visionevoagent.search_space import BASELINE, architecture_signature as sig, get_search_space  # noqa: E402

TASK, SD = "classification", 0.25
ECA = {**BASELINE, "attention": "eca"}
ECA_W = {**ECA, "fusion": "weighted"}
claim_eca = {"condition": [{"dim": "attention", "value": "eca"}], "reference": [], "direction": "increase",
             "min_effect": SD}
claim_w = {"condition": [{"dim": "fusion", "value": "weighted"}], "reference": [], "direction": "increase",
           "min_effect": SD}
fields = {"if": "attention=eca", "scope": "s", "then": "t", "because": "b", "disproved_if": "d"}

# 1. prompts: only the judging paragraph differs from the v4 Reflector prompt
assert "You do NOT judge evidence" in REFLECTOR_SYSTEM_V4
for mode, field in (("evidence", "principle_updates"), ("confidence", "principle_confidences")):
    s = REFLECTOR_SYSTEM_V4_JUDGE[mode]
    assert "You do NOT judge evidence" not in s and "You JUDGE the evidence" in s and field in s, mode
    assert "Bayesian" not in s and "EVIDENCE_TABLE" not in s, mode           # no evidence model in the control
    assert "VERIFIABLE PRINCIPLE LIBRARY" in s and "machine-checkable claim" in s and s.endswith("Return JSON only.")
    assert s.split("You JUDGE the evidence")[0] == REFLECTOR_SYSTEM_V4.split("You do NOT judge evidence")[0]
    assert s.split("Every new principle must pass all of:")[1] == \
        REFLECTOR_SYSTEM_V4.split("Every new principle must pass all of:")[1]
assert set(LLM_JUDGE_MODES) <= set(PRINCIPLE_MODES)

# 2. schema: the default asks for no verdicts; the controls ask for theirs
props = lambda j: reflection_schema(TASK, computed_evidence=True, judge=j)  # noqa: E731
assert "principle_updates" not in props("")["properties"] and "principle_confidences" not in props("")["properties"]
assert "principle_updates" in props("evidence")["required"]
assert "principle_confidences" in props("confidence")["required"]
assert "principle_updates" not in props("confidence")["properties"]
assert "confidence" in props("confidence")["properties"]["new_principles"]["items"]["required"]
assert "confidence" not in props("evidence")["properties"]["new_principles"]["items"]["properties"]

# 3. validation
base = {"effective_patterns": [], "ineffective_patterns": [], "synergies": [], "new_principles": [],
        "next_search_direction": [], "data_sufficiency": "limited"}
ok = lambda extra, j: validate_reflection({**base, **extra}, TASK, computed_evidence=True, judge=j)  # noqa: E731
assert ok({}, "") is None                                                    # default: nothing to report
assert ok({}, "evidence") is not None                                        # verdicts are required
assert ok({"principle_updates": []}, "evidence") is None
assert ok({}, "confidence") is not None
pc = {"principle_id": "p1", "confidence": 0.8, "reason": "r"}
assert ok({"principle_confidences": [pc]}, "confidence") is None
assert ok({"principle_confidences": [{**pc, "confidence": 1.3}]}, "confidence") is not None
new = {**fields, "claim": claim_eca}
assert ok({"principle_confidences": [], "new_principles": [{**new, "confidence": 0.7}]}, "confidence") is None
assert ok({"principle_confidences": [], "new_principles": [new]}, "confidence") is None       # unstated: stays 0.5
assert ok({"principle_confidences": [], "new_principles": [{**new, "confidence": -0.1}]}, "confidence") is not None

# 4. user prompt: no output of the evidence model reaches the judge
kw = dict(task=TASK, metric="top1", space=get_search_space(TASK),
          baseline={"architecture": BASELINE, "score": 78.0, "params": 23.7}, history=[], new_trials=[],
          best={"architecture": ECA, "score": 78.6, "params": 23.7}, failed=[], noise_level=SD,
          previous_principles=[])
assert "EVIDENCE_TABLE" in build_reflection_prompt(**kw, evidence_table=[])
for mode in ("evidence", "confidence"):
    assert "EVIDENCE_TABLE" not in build_reflection_prompt(**kw, evidence_table=None, judge=mode)
assert "report your confidence for them by id" in build_reflection_prompt(**kw, judge="confidence")
assert "report evidence for them by id" in build_reflection_prompt(**kw, judge="evidence")

# 5. memory: stated confidence; verdicts that do not touch the pair counters
mem = ArchitectureMemory(tempfile.mkdtemp())
p = mem.add_principle(fields, TASK, 1, claim=claim_eca, restate_bonus=False)
assert mem.set_confidence(p.id, 0.9, 2, reason="r").status == "confirmed" and p.evidence[-1]["type"] == "llm_confidence"
assert mem.set_confidence(p.id, 1.7, 3).confidence == 0.99                   # clamped to CONF_BOUNDS
assert mem.set_confidence("nope", 0.5, 3) is None and mem.set_confidence(p.id, "high", 3) is None
mem.set_confidence(p.id, 0.5, 4)
mem.update_principle(p.id, "supports", 1.0, 5, eta=0.35, floor=0.5, count=False)
assert abs(p.confidence - 0.675) < 1e-6 and p.support_count == 0, (p.confidence, p.support_count)


# 6. controller: the same reflection moves the confidence by the judge's rule, not by the model
def setup(mode):
    m = ArchitectureMemory(tempfile.mkdtemp())
    m.baseline = Trial(0, dict(BASELINE), sig(BASELINE), "ok", 78.0, params=23.7, delta_vs_baseline=0.0)
    m.add_trial(Trial(1, dict(ECA), sig(ECA), "ok", 78.6, params=23.7, delta_vs_baseline=0.6, delta_vs_previous=0.6,
                      is_best=True, parent=sig(BASELINE), delta_vs_parent=0.6))
    m.add_trial(Trial(2, dict(ECA_W), sig(ECA_W), "ok", 78.2, params=23.7, delta_vs_baseline=0.2,
                      delta_vs_previous=-0.4, parent=sig(ECA), delta_vs_parent=-0.4))
    pr = m.add_principle(fields, TASK, 1, claim=claim_eca, restate_bonus=False)
    client = AgentClient(AgentClientConfig(provider="mock"))
    c = AgentSearchController(TASK, 0, DesignAgent(client, seed=0), ReflectorAgent(client),
                              AgentSearchConfig(principle_mode=mode), metric="top1", noise_level=SD)
    return m, pr, c


m, pr, c = setup("online")
assert c._judge() == ""
out = c._apply_reflection(m, {"new_principles": [], "principle_updates": []}, 2)
assert out["evidence"] == "computed" and out["updated"] == [pr.id] and pr.stats, out
posterior = pr.confidence
assert posterior > 0.6, posterior                                            # eca: +0.6 alone, about +0.5 jointly

m, pr, c = setup("llm_evidence")
assert c._judge() == "evidence"
up = {"principle_id": pr.id, "evidence": "contradicts", "strength": 1.0, "trials": "r2", "reason": "x"}
out = c._apply_reflection(m, {"new_principles": [], "principle_updates": [up, up, {**up, "principle_id": "nope"}]}, 2)
assert out["evidence"] == "llm_evidence" and out["updated"] == [pr.id] and out["ignored"] == [pr.id, "nope"], out
assert abs(pr.confidence - 0.325) < 1e-6 and not pr.stats, (pr.confidence, pr.stats)   # 0.5 - 0.35 * 1 * 0.5
assert pr.contradict_count == 0 and abs(pr.confidence - posterior) > 0.2      # the model never touched it

m, pr, c = setup("llm_confidence")
assert c._judge() == "confidence"
ref = {"new_principles": [{**fields, "if": "fusion=weighted", "claim": claim_w, "confidence": 0.3}],
       "principle_updates": [], "principle_confidences": [{"principle_id": pr.id, "confidence": 0.9, "reason": "r"}]}
out = c._apply_reflection(m, ref, 2)
assert out["evidence"] == "llm_confidence" and out["updated"] == [pr.id] and len(out["added"]) == 1, out
assert pr.confidence == 0.9 and pr.status == "confirmed" and not pr.stats
q = m.principle(out["added"][0])
assert q.confidence == 0.3 and q.evidence[0]["type"] == "llm_confidence", (q.confidence, q.evidence)
assert any(e["type"] == "counterexample" for e in q.evidence), q.evidence      # eca -> eca+weighted: -0.4 (backfilled)

# 7. a whole reflection round through the mock Reflector, for both controls
for mode in LLM_JUDGE_MODES:
    m, pr, c = setup(mode)
    ref = c.after_trial(m, 2)
    assert ref is not None and ref["applied"]["evidence"] == mode, (mode, ref)
    assert m.reflections and all(not x.stats for x in m.principles), mode       # the model scored nothing
    entry = c._principle_entry(pr)
    assert "claim" in entry and "effect_estimate" not in entry, entry

# 8. the controls are defined for the hypothesis-guided framework only
for bad in (dict(selection="none"), dict(evidence_mode="llm"), dict(use_principle_memory=False),
            dict(reflection_style="immediate")):
    client = AgentClient(AgentClientConfig(provider="mock"))
    try:
        AgentSearchController(TASK, 0, DesignAgent(client, seed=0), ReflectorAgent(client),
                              AgentSearchConfig(principle_mode="llm_evidence", **bad), metric="top1", noise_level=SD)
    except ValueError:
        continue
    raise AssertionError(f"llm_evidence accepted with {bad}")
print("LLM-judged controls: all checks passed")
