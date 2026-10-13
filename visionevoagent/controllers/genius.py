"""GENIUS (Zheng et al., arXiv 2304.10970): the LLM as a black-box optimiser in one conversation.

Follows the NAS-Bench-201 script of the official repo (the variant matching Algorithm 1 of the paper):
turn 1 encodes the problem, every later turn reports the last result and asks for a better model. No
reflection, no memory structure beyond the conversation itself.

Deviations: (1) the original sets temperature=0; Claude Opus 5.5 rejects `temperature`, so the default
sampling is used. (2) The original script needs a human to type the proposed operation IDs; here the
answer is structured JSON. (3) The persona naming a real researcher is replaced by a generic expert.
The conversation is rebuilt from memory each round (proposer_note keeps the rationale), so resume is
exact and no controller state is needed.
"""
from __future__ import annotations

import json

from ..agent_client import AgentClient
from ..baseline_prompts import GENIUS_FEEDBACK, GENIUS_PROBLEM, GENIUS_SYSTEM
from ..memory import ArchitectureMemory
from .base import BaseController, Proposal
from .llm_common import ask_architecture, fmt_arch, fmt_result, fmt_space, mock_neighbour


class GeniusController(BaseController):
    name = "genius"

    def __init__(self, task: str, seed: int, client: AgentClient, metric: str):
        super().__init__(task, seed)
        self.client = client
        self.metric = metric

    def _conversation(self, memory: ArchitectureMemory):
        b = memory.baseline
        convo = [{"role": "user", "content": GENIUS_PROBLEM.format(
            metric=self.metric, task=self.task, space=fmt_space(self.task),
            baseline=f"{fmt_arch(b.architecture)} with {fmt_result(b, self.metric)}")}]
        for t in sorted(memory.trials, key=lambda t: t.round):
            rationale = t.proposer_note[len("genius: "):] if t.proposer_note.startswith("genius: ") else t.proposer_note
            convo.append({"role": "assistant",
                          "content": json.dumps({"architecture": t.architecture, "rationale": rationale})})
            convo.append({"role": "user", "content": GENIUS_FEEDBACK.format(result=fmt_result(t, self.metric))})
        return convo

    def propose(self, memory: ArchitectureMemory, round_: int) -> Proposal:
        tried = memory.tried_signatures()
        best = memory.best_trial() or memory.baseline
        arch, why, _ = ask_architecture(
            self.client, "genius", GENIUS_SYSTEM, self._conversation(memory), self.task, tried,
            mock_fn=lambda att: mock_neighbour(best.architecture, self.task, tried, self.rng(round_, f"mock{att}")))
        if arch is None:
            return Proposal(self.random_untried(memory, round_), note="genius-fallback: random untried")
        return Proposal(arch, note=f"genius: {why}")
