"""Re-ask rule (decision 2026-10-13): one ask plus at most two re-asks, then a random untried fallback;
every attempt's outcome is recorded for the duplicate rate. Offline, uses the mock provider.

Usage (repo root): python tests/test_reask.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())

from visionevoagent.agent_client import AgentClient, AgentClientConfig  # noqa: E402
from visionevoagent.agents.design_agent import DesignAgent, DesignContext  # noqa: E402
from visionevoagent.controllers.llm_common import ask_architecture  # noqa: E402
from visionevoagent.schemas import MAX_PROPOSAL_ATTEMPTS  # noqa: E402
from visionevoagent.search_space import BASELINE, architecture_signature  # noqa: E402

client = AgentClient(AgentClientConfig(provider="mock"))
TRIED = {"none": dict(BASELINE), "eca": {**BASELINE, "attention": "eca"}}
NEW = {**BASELINE, "attention": "se"}
exclude = {architecture_signature(a) for a in TRIED.values()}
assert MAX_PROPOSAL_ATTEMPTS == 3


def answers(seq):
    return lambda att: {"architecture": seq[min(att, len(seq)) - 1], "rationale": "test"}


# duplicate, duplicate, new -> accepted on the 3rd answer
rec = []
arch, _, _ = ask_architecture(client, "t", "s", "p", "classification", exclude,
                              answers([TRIED["none"], TRIED["eca"], NEW]), record=rec)
assert architecture_signature(arch) == architecture_signature(NEW), arch
assert rec == ["duplicate", "duplicate", "ok"], rec

# three duplicates -> None (the controller then falls back to a random untried architecture)
rec = []
arch, _, _ = ask_architecture(client, "t", "s", "p", "classification", exclude,
                              answers([TRIED["eca"]]), record=rec)
assert arch is None and rec == ["duplicate"] * 3, rec

# illegal value counts as invalid, not duplicate
rec = []
ask_architecture(client, "t", "s", "p", "classification", exclude,
                 answers([{**BASELINE, "attention": "transformer"}, NEW]), record=rec)
assert rec == ["invalid", "ok"], rec

# Design Agent: same limit, same bookkeeping, fallback after three duplicates
agent = DesignAgent(client, seed=0)
agent._mock = lambda ctx, attempt: {"architecture": TRIED["eca"], "rationale": "test"}
ctx = DesignContext(task="classification", metric="top1", baseline={"architecture": BASELINE, "score": 78.0,
                    "params": 23.7}, best={"architecture": BASELINE, "score": 78.0, "params": 23.7},
                    history=[], tried=exclude, failed=set(), search_mode="exploit", mutation_budget=1,
                    noise_level=0.2, reflection_context=None, round=1)
rec = []
arch, note, _ = agent.propose(ctx, record=rec)
assert rec == ["duplicate"] * 3, rec
assert note.startswith("fallback") and architecture_signature(arch) not in exclude, (arch, note)
print("re-ask rule: all checks passed")
