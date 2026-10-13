"""EvoPrompting (Chen, Dohan & So, NeurIPS 2023), scaled to the 10-trial budget.

Original: T=10 rounds x m=10 prompts x n=16 samples, k=2 parents per prompt drawn uniformly from the
top p=10, fitness = -(model size x validation error), target metrics in the last line of the prompt
(size = 90% of the smallest parent, accuracy = 102% of the best parent), selected parents are retired,
soft prompt-tuning between rounds, mixed sampling temperature.

Here: generations of `children_per_gen` trials (2 x 5 for a 10-trial budget), one sample per prompt,
k=2 parents drawn from the top p=4 of everything evaluated before the generation started (baseline
included), fitness = -(params x (100 - top1)). Not reproduced: soft prompt-tuning (API model, no
weights) and temperature mixing (Claude Opus 5.5 rejects `temperature`); diversity comes from the
different parent pairs. Retired parents are kept in controller state.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..agent_client import AgentClient
from ..baseline_prompts import EVOPROMPTING_SYSTEM, evoprompting_prompt
from ..memory import ArchitectureMemory, Trial
from .base import BaseController, Proposal
from .llm_common import ask_architecture, fmt_arch, fmt_space, mock_neighbour, ok_trials_with_baseline


def fitness(t: Trial) -> float:
    return -(t.params or 0.0) * (100.0 - t.score)


class EvoPromptingController(BaseController):
    name = "evoprompting"

    def __init__(self, task: str, seed: int, client: AgentClient, children_per_gen: int = 5,
                 top_p: int = 4, k: int = 2):
        super().__init__(task, seed)
        self.client = client
        self.gen_size = children_per_gen
        self.top_p = top_p
        self.k = k

    def _parents_for_generation(self, memory: ArchitectureMemory, gen: int) -> List[Trial]:
        key = str(gen)
        gens: Dict[str, Any] = self.state.setdefault("generations", {})
        pool = [t for t in ok_trials_with_baseline(memory) if t.round <= gen * self.gen_size]
        if key not in gens:
            retired = set(self.state.setdefault("retired", []))
            cands = [t for t in pool if t.signature not in retired] or pool
            top = sorted(cands, key=fitness, reverse=True)[: self.top_p]
            gens[key] = [t.signature for t in top]
            self.state["retired"] = sorted(retired | set(gens[key]))   # GetTop removes them from G
        by_sig = {t.signature: t for t in pool}
        return [by_sig[s] for s in gens[key] if s in by_sig]

    def propose(self, memory: ArchitectureMemory, round_: int) -> Proposal:
        gen = (round_ - 1) // self.gen_size
        top = self._parents_for_generation(memory, gen)
        rng = self.rng(round_, "parents")
        parents = rng.sample(top, min(self.k, len(top)))
        target = {"params": round(0.9 * min(p.params for p in parents), 2),
                  "score": round(min(100.0, 1.02 * max(p.score for p in parents)), 2)}
        prompt = evoprompting_prompt(
            self.task, fmt_space(self.task),
            [{"params": p.params, "score": p.score, "arch": fmt_arch(p.architecture)} for p in parents], target)
        tried = memory.tried_signatures()
        arch, why, _ = ask_architecture(
            self.client, "evoprompting", EVOPROMPTING_SYSTEM, prompt, self.task, tried,
            mock_fn=lambda att: mock_neighbour(parents[0].architecture, self.task, tried,
                                               self.rng(round_, f"mock{att}")))
        if arch is None:
            return Proposal(self.random_untried(memory, round_), note="evoprompting-fallback: random untried")
        names = ", ".join(f"r{p.round:03d}" for p in parents)
        return Proposal(arch, note=f"evoprompting gen {gen} parents [{names}]: {why}")
