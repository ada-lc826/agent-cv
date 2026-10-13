"""Runner interface: train + evaluate one architecture under the proxy protocol."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from ..search_space import Architecture


@dataclass
class TrialResult:
    status: str                    # ok | failed
    score: Optional[float] = None
    params: Optional[float] = None  # millions
    failure_reason: str = ""       # oom | crash | nan | timeout | missing_result
    duration_s: float = 0.0
    work_dir: str = ""


class BaseRunner(ABC):
    metric = "score"

    @abstractmethod
    def run(self, arch: Architecture, tag: str) -> TrialResult:
        """Train/evaluate `arch`. Must never raise: failures are returned as status='failed'."""

    def param_table(self, task: str) -> Optional[Dict[str, float]]:
        """Parameters (millions) of every architecture in the search space, keyed by signature,
        computed without training. None if the runner cannot provide it; agents then work
        without the cost table."""
        return None
