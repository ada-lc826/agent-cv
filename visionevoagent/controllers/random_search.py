"""Random Search baseline (Li & Talwalkar, UAI 2019): uniform sampling without replacement.

Uses no history, no reflection. Resumable: the sample for round r depends only on (seed, r)
and the set of signatures already tried.
"""
from __future__ import annotations

import random

from ..memory import ArchitectureMemory
from ..search_space import all_architectures, architecture_signature
from .base import BaseController, Proposal


class RandomSearchController(BaseController):
    name = "random"

    def propose(self, memory: ArchitectureMemory, round_: int) -> Proposal:
        tried = memory.tried_signatures()
        pool = [a for a in all_architectures(self.task) if architecture_signature(a) not in tried]
        if not pool:
            raise RuntimeError("search space exhausted")
        rng = random.Random(f"random-{self.seed}-{round_}")
        return Proposal(rng.choice(pool), note="random sample")
