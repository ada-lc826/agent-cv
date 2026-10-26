"""Controller interface shared by VisionEvoAgent and every comparison method.

Fairness contract: all controllers search the same SEARCH_SPACE, get the same trial budget
(max_rounds), are evaluated by the same runner, and are seeded.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..memory import ArchitectureMemory
from ..search_space import Architecture, all_architectures, architecture_signature


@dataclass
class Proposal:
    architecture: Architecture
    note: str = ""
    search_mode: str = ""
    # principles the proposal relies on, snapshot at proposal time: [{"id", "confidence", "status"}]
    cited_principles: List[Dict[str, Any]] = field(default_factory=list)
    # predictions made before training (reflection v3), stored with the trial
    predictions: Dict[str, Any] = field(default_factory=dict)
    # hypothesis-guided framework: parent signature and the selection record
    parent: str = ""
    selection: Dict[str, Any] = field(default_factory=dict)


class BaseController(ABC):
    name = "base"

    def __init__(self, task: str, seed: int):
        self.task = task
        self.seed = seed
        # Controller-private state that cannot be rebuilt from ArchitectureMemory (archives, learned
        # principles, experience stores, ...). The search engine saves it after every round to
        # controller_state.json and restores it on resume. Must stay JSON-serialisable.
        self.state: Dict[str, Any] = {}
        # Outcome of every LLM answer behind the current proposal (ok | duplicate | invalid | call_error);
        # cleared by the search engine before each propose() and stored with the trial.
        self.proposal_attempts: List[str] = []

    @abstractmethod
    def propose(self, memory: ArchitectureMemory, round_: int) -> Proposal:
        """Return the next architecture. Must not return an already-tried signature."""

    def after_trial(self, memory: ArchitectureMemory, round_: int) -> Optional[dict]:
        """Hook called after each trial (e.g. reflection). Return a record to log, or None."""
        return None

    # ------------------------------------------------------------------ helpers
    def rng(self, round_: int, purpose: str = "") -> random.Random:
        """Deterministic per-round RNG, so a resumed run makes the same random choices."""
        return random.Random(f"{self.name}-{self.seed}-{round_}-{purpose}")

    def random_untried(self, memory: ArchitectureMemory, round_: int, purpose: str = "fallback") -> Architecture:
        tried = memory.tried_signatures()
        pool = [a for a in all_architectures(self.task) if architecture_signature(a) not in tried]
        if not pool:
            raise RuntimeError("search space exhausted")
        return self.rng(round_, purpose).choice(pool)
