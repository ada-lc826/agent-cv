"""ArchitectureMemory: trial / failure / principle memories persisted as JSON files.

Independent of any agent, so every controller (agent or baseline) writes the same records
and the whole search can be reconstructed from disk.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .search_space import Architecture, architecture_signature

log = logging.getLogger(__name__)


@dataclass
class Trial:
    round: int
    architecture: Architecture
    signature: str
    status: str                      # "ok" | "failed"
    score: Optional[float] = None
    params: Optional[float] = None   # millions
    delta_vs_baseline: Optional[float] = None
    delta_vs_previous: Optional[float] = None
    param_ratio: Optional[float] = None
    failure_reason: str = ""
    search_mode: str = ""            # "exploit" | "explore" | "" (non-agent)
    proposer_note: str = ""          # rationale / how the candidate was chosen
    is_best: bool = False
    duration_s: float = 0.0
    task: str = ""


@dataclass
class Failure:
    architecture_signature: str
    reason: str                      # score_drop | oom | crash | nan | timeout | invalid
    delta: Optional[float] = None
    round: int = -1


@dataclass
class Principle:
    principle: str
    confidence: float
    support_count: int = 1
    task: str = ""
    first_round: int = -1
    last_round: int = -1


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass
class ArchitectureMemory:
    root: str
    trials: List[Trial] = field(default_factory=list)
    failures: List[Failure] = field(default_factory=list)
    principles: List[Principle] = field(default_factory=list)
    reflections: List[Dict[str, Any]] = field(default_factory=list)
    baseline: Optional[Trial] = None

    # ---------------- persistence ----------------
    def _path(self, name: str) -> str:
        return os.path.join(self.root, name)

    def save(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        payloads = {
            "trial_memory.json": [asdict(t) for t in self.trials],
            "failure_memory.json": [asdict(f) for f in self.failures],
            "principle_memory.json": [asdict(p) for p in self.principles],
            "reflections.json": self.reflections,
            "baseline.json": asdict(self.baseline) if self.baseline else None,
        }
        for name, data in payloads.items():
            tmp = self._path(name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self._path(name))  # atomic: a crash never leaves a half-written file

    @classmethod
    def load(cls, root: str) -> "ArchitectureMemory":
        mem = cls(root=root)

        def read(name: str, default: Any) -> Any:
            p = mem._path(name)
            if not os.path.exists(p):
                return default
            with open(p, encoding="utf-8") as f:
                return json.load(f)

        mem.trials = [Trial(**t) for t in read("trial_memory.json", [])]
        mem.failures = [Failure(**x) for x in read("failure_memory.json", [])]
        mem.principles = [Principle(**x) for x in read("principle_memory.json", [])]
        mem.reflections = read("reflections.json", [])
        b = read("baseline.json", None)
        mem.baseline = Trial(**b) if b else None
        return mem

    def load_principles_from(self, path: str, task: str = "") -> int:
        """Cold-start from another run's principle memory (principle-transfer experiments)."""
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        n = 0
        for x in items:
            p = Principle(**x)
            p.principle = f"[transferred{(' from ' + p.task) if p.task else ''}] {p.principle}"
            p.task = task or p.task
            self.principles.append(p)
            n += 1
        return n

    # ---------------- queries ----------------
    def tried_signatures(self) -> set:
        sigs = {t.signature for t in self.trials}
        if self.baseline:
            sigs.add(self.baseline.signature)
        return sigs

    def failed_signatures(self) -> set:
        return {f.architecture_signature for f in self.failures}

    def ok_trials(self) -> List[Trial]:
        return [t for t in self.trials if t.status == "ok" and t.score is not None]

    def best_trial(self) -> Optional[Trial]:
        best = [t for t in self.trials if t.is_best]
        return best[-1] if best else None

    def last_round(self) -> int:
        return max((t.round for t in self.trials), default=0)

    def find(self, arch: Architecture) -> Optional[Trial]:
        sig = architecture_signature(arch)
        for t in self.trials:
            if t.signature == sig:
                return t
        return None

    # ---------------- updates ----------------
    def add_trial(self, trial: Trial) -> None:
        self.trials.append(trial)

    def add_failure(self, failure: Failure) -> None:
        if failure.architecture_signature not in self.failed_signatures():
            self.failures.append(failure)

    def merge_principles(self, texts: List[str], confidence: float, task: str, round_: int) -> None:
        """Add principles; repeated principles (same normalised text) gain support instead of duplicating."""
        index = {_norm(p.principle): p for p in self.principles}
        for text in texts:
            key = _norm(text)
            if not key:
                continue
            if key in index:
                p = index[key]
                p.support_count += 1
                p.confidence = round(max(p.confidence, confidence), 3)
                p.last_round = round_
            else:
                p = Principle(text, round(confidence, 3), 1, task, round_, round_)
                self.principles.append(p)
                index[key] = p

    def add_reflection(self, round_: int, reflection: Dict[str, Any]) -> None:
        self.reflections.append({"round": round_, **reflection})

    def latest_reflection(self) -> Optional[Dict[str, Any]]:
        return self.reflections[-1] if self.reflections else None
