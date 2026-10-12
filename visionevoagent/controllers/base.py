"""Controller interface shared by VisionEvoAgent and every comparison method.

Fairness contract: all controllers search the same SEARCH_SPACE, get the same trial budget
(max_rounds), are evaluated by the same runner, and are seeded.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

from ..memory import ArchitectureMemory
from ..search_space import Architecture


@dataclass
class Proposal:
    architecture: Architecture
    note: str = ""
    search_mode: str = ""


class BaseController(ABC):
    name = "base"

    def __init__(self, task: str, seed: int):
        self.task = task
        self.seed = seed

    @abstractmethod
    def propose(self, memory: ArchitectureMemory, round_: int) -> Proposal:
        """Return the next architecture. Must not return an already-tried signature."""

    def after_trial(self, memory: ArchitectureMemory, round_: int) -> Optional[dict]:
        """Hook called after each trial (e.g. reflection). Return a record to log, or None."""
        return None
