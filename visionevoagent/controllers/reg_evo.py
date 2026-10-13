"""Regularized (aging) Evolution (Real et al., AAAI 2019), scaled to the 10-trial budget.

Original: population P=100, tournament S=25, one random mutation per child, remove the OLDEST
individual. Here P=4, S=2 (LAPT uses a similarly small REA on NAS-Bench-201): the first P trials are
random, every later trial mutates the tournament winner in one dimension.

The population is the P most recently evaluated trials, which is exactly the aging queue, so the
state is rebuilt from memory and resume needs nothing extra.
"""
from __future__ import annotations

from ..memory import ArchitectureMemory
from ..search_space import DIMENSIONS, random_local_mutation
from .base import BaseController, Proposal


class RegularizedEvolutionController(BaseController):
    name = "reg_evo"

    def __init__(self, task: str, seed: int, population_size: int = 4, tournament_size: int = 2):
        super().__init__(task, seed)
        self.P = population_size
        self.S = tournament_size

    def propose(self, memory: ArchitectureMemory, round_: int) -> Proposal:
        trials = sorted(memory.trials, key=lambda t: t.round)
        if len(trials) < self.P:
            return Proposal(self.random_untried(memory, round_, "init"), note="reg_evo: random initial population")
        population = trials[-self.P:]                              # aging: oldest dropped first
        alive = [t for t in population if t.status == "ok" and t.score is not None]
        if not alive:
            return Proposal(self.random_untried(memory, round_, "dead"), note="reg_evo: no trainable parent")
        rng = self.rng(round_, "tournament")
        sample = [rng.choice(alive) for _ in range(self.S)]           # with replacement, as in the paper
        parent = max(sample, key=lambda t: t.score)
        excluded = memory.tried_signatures()
        for budget in range(1, len(DIMENSIONS) + 1):                  # widen only if every 1-change is tried
            child = random_local_mutation(parent.architecture, self.task, self.rng(round_, "mutate"),
                                          excluded, budget)
            if child is not None:
                return Proposal(child, note=f"reg_evo: mutate r{parent.round:03d} (budget {budget})")
        return Proposal(self.random_untried(memory, round_), note="reg_evo: no untried mutation, random")
