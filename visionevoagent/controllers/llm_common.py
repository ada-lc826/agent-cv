"""Shared plumbing for the LLM-driven comparison methods (GENIUS, EvoPrompting, LLMatic, LAPT, NADER).

All of them ask the LLM for one architecture from the same SEARCH_SPACE, with the same structured
output schema as the Design Agent, and fall back to a legal random choice when the answer is unusable,
so a method never stalls the search.
"""
from __future__ import annotations

import json
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..agent_client import AgentClient, AgentClientError, AgentClientFatal
from ..memory import ArchitectureMemory, Trial
from ..schemas import MAX_PROPOSAL_ATTEMPTS, attempt_outcome, design_schema, validate_design
from ..search_space import (DIMENSIONS, MODULE_NOTES, Architecture, architecture_signature, canonicalize,
                            get_search_space, random_local_mutation)

log = logging.getLogger(__name__)

Conversation = List[Dict[str, str]]


def fmt_arch(a: Architecture) -> str:
    return json.dumps({d: a[d] for d in DIMENSIONS})


def fmt_space(task: str) -> str:
    space = get_search_space(task)
    lines = []
    for d in DIMENSIONS:
        opts = ", ".join(f"{o} ({MODULE_NOTES[o]})" if o in MODULE_NOTES else o for o in space[d])
        lines.append(f"- {d}: {opts}")
    return "\n".join(lines)


def fmt_result(t: Trial, metric: str) -> str:
    if t.status != "ok":
        return f"failed to train ({t.failure_reason or 'unknown reason'})"
    return f"{metric} {t.score:.2f}, {t.params:.2f}M params" if t.params else f"{metric} {t.score:.2f}"


def ask_architecture(client: AgentClient, purpose: str, system: str, prompt: Union[str, Conversation],
                     task: str, exclude: set, mock_fn: Callable[[int], Dict[str, Any]],
                     max_attempts: int = MAX_PROPOSAL_ATTEMPTS, effort: Optional[str] = None,
                     temperature: Optional[float] = None, record: Optional[List[str]] = None
                     ) -> Tuple[Optional[Architecture], str, Union[str, Conversation]]:
    """Ask for one new architecture. Returns (architecture or None, rationale, prompt incl. feedback).

    A rejected answer (illegal value, already evaluated) is re-asked with feedback, up to max_attempts
    answers in total: for a single prompt the feedback is appended to it, for a conversation it is added
    as an assistant/user turn pair. The outcome of every attempt is appended to `record`
    (ok | duplicate | invalid | call_error).
    """
    convo = prompt
    record = record if record is not None else []
    for attempt in range(1, max_attempts + 1):
        try:
            out = client.complete_json(purpose, system, convo, design_schema(task),
                                       mock_fn=lambda: mock_fn(attempt), effort=effort, temperature=temperature)
        except AgentClientFatal:
            raise
        except AgentClientError as e:
            log.warning("%s call failed (attempt %d): %s", purpose, attempt, e)
            record.append("call_error")
            continue
        reason = validate_design(out, task)
        arch = None
        if reason is None:
            arch = canonicalize(out["architecture"])
            if architecture_signature(arch) in exclude:
                reason = f"{architecture_signature(arch)} was already evaluated"
        record.append(attempt_outcome(reason))
        if reason is None:
            return arch, out["rationale"], convo
        log.info("%s answer rejected (attempt %d): %s", purpose, attempt, reason)
        feedback = f"That answer was rejected: {reason}. Propose a different architecture."
        if isinstance(convo, str):
            convo = f"{prompt}\n\n{feedback}"
        else:
            convo = list(convo) + [{"role": "assistant", "content": json.dumps(out)},
                                   {"role": "user", "content": feedback}]
    return None, "", convo


def mock_neighbour(arch: Architecture, task: str, exclude: set, rng: random.Random) -> Dict[str, Any]:
    """Dry-run stand-in for an LLM answer: a random untried one-dimension change of `arch`."""
    for budget in range(1, len(DIMENSIONS) + 1):
        cand = random_local_mutation(arch, task, rng, exclude, budget)
        if cand is not None:
            return {"architecture": cand, "rationale": "mock: random neighbour"}
    return {"architecture": arch, "rationale": "mock: nothing left"}


def ok_trials_with_baseline(memory: ArchitectureMemory) -> List[Trial]:
    base = [memory.baseline] if memory.baseline and memory.baseline.status == "ok" else []
    return base + memory.ok_trials()
