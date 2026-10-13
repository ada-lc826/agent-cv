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
    # Principles the Design Agent said it relied on, with their confidence/status at proposal time:
    # [{"id", "confidence", "status"}]. Used for the calibration analysis (tools/principle_calibration.py).
    cited_principles: List[Dict[str, Any]] = field(default_factory=list)
    # Outcome of every LLM answer behind this proposal, in order: ok | duplicate | invalid | call_error.
    # Empty for methods that do not ask an LLM for the architecture. Used for the duplicate rate.
    proposal_attempts: List[str] = field(default_factory=list)


@dataclass
class Failure:
    architecture_signature: str
    reason: str                      # score_drop | oom | crash | nan | timeout | invalid
    delta: Optional[float] = None
    round: int = -1


# Evidence-based confidence (after HypoExplore, Koo et al. 2026, Eq. 1):
#   supports:    c <- c + ETA * s * (1 - c)        contradicts: c <- c - ETA * s * c
# with evidence strength s in [0, 1]. New principles start at maximal uncertainty (0.5).
ETA = 0.2
INITIAL_CONFIDENCE = 0.5
CONFIRMED_AT = 0.75
REFUTED_AT = 0.25
CONF_BOUNDS = (0.01, 0.99)
PRINCIPLE_FIELDS = ("if", "scope", "then", "because", "disproved_if")


def render_principle(f: Dict[str, str]) -> str:
    """Falsifiable form: IF <choice> IN <scope>, THEN <predicted effect>, BECAUSE <mechanism>. DISPROVED IF <...>."""
    return (f"IF {f.get('if', '').strip()} IN {f.get('scope', '').strip()}, THEN {f.get('then', '').strip()}, "
            f"BECAUSE {f.get('because', '').strip()}. DISPROVED IF {f.get('disproved_if', '').strip()}.")


@dataclass
class Principle:
    principle: str                       # rendered statement (see render_principle)
    confidence: float = INITIAL_CONFIDENCE
    support_count: int = 0               # number of supporting evidence updates
    task: str = ""
    first_round: int = -1
    last_round: int = -1
    id: str = ""
    fields: Dict[str, str] = field(default_factory=dict)   # if / scope / then / because / disproved_if
    contradict_count: int = 0
    neutral_count: int = 0
    status: str = "uncertain"            # uncertain | confirmed | refuted
    # [{"round", "type": supports|contradicts|neutral, "strength", "trials", "reason", "before", "after"}]
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    transferred_from: str = ""           # task (or run) the principle was learned in, for transferred ones


def _status(conf: float) -> str:
    return "confirmed" if conf > CONFIRMED_AT else ("refuted" if conf < REFUTED_AT else "uncertain")


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
        mem._ensure_ids()
        mem.reflections = read("reflections.json", [])
        b = read("baseline.json", None)
        mem.baseline = Trial(**b) if b else None
        return mem

    def load_principles_from(self, path: str, task: str = "", mark_transferred: bool = True) -> int:
        """Cold-start from another run's principle memory (principle-transfer experiments), or from a
        hand-written file (fixed-principle control, mark_transferred=False).

        Refuted principles are not transferred. The others keep their confidence and evidence log, get a
        fresh id in this run and `transferred_from`; new evidence in this task keeps updating them."""
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        n = 0
        for x in items:
            if "principle" not in x and "fields" in x:  # hand-written: fields only
                x = {**x, "principle": render_principle(x["fields"])}
            p = Principle(**x)
            if p.status == "refuted":
                continue
            if mark_transferred:
                p.transferred_from = p.transferred_from or p.task or os.path.basename(os.path.dirname(path))
            p.task = task or p.task
            p.id = self._next_principle_id()
            self.principles.append(p)
            n += 1
        return n

    def add_principles(self, items: List[Principle]) -> int:
        for p in items:
            p.id = self._next_principle_id()
            self.principles.append(p)
        return len(items)

    def _next_principle_id(self) -> str:
        nums = [int(p.id[1:]) for p in self.principles if re.fullmatch(r"p\d+", p.id or "")]
        return f"p{max(nums, default=0) + 1}"

    def _ensure_ids(self) -> None:
        """Principle files written before ids existed get ids in list order."""
        for p in self.principles:
            if not p.id:
                p.id = self._next_principle_id()

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

    def principle(self, pid: str) -> Optional[Principle]:
        return next((p for p in self.principles if p.id == pid), None)

    def update_principle(self, pid: str, evidence: str, strength: float, round_: int,
                         reason: str = "", trials: str = "") -> Optional[Principle]:
        """Apply one piece of evidence. Returns None for an unknown id. Neutral evidence is logged only."""
        p = self.principle(pid)
        if p is None or evidence not in ("supports", "contradicts", "neutral"):
            return None
        s = min(1.0, max(0.0, float(strength)))
        before = p.confidence
        if evidence == "supports":
            p.confidence = before + ETA * s * (1.0 - before)
            p.support_count += 1
        elif evidence == "contradicts":
            p.confidence = before - ETA * s * before
            p.contradict_count += 1
        else:
            p.neutral_count += 1
        p.confidence = round(min(CONF_BOUNDS[1], max(CONF_BOUNDS[0], p.confidence)), 4)
        p.status = _status(p.confidence)
        p.last_round = round_
        p.evidence.append({"round": round_, "type": evidence, "strength": round(s, 3), "trials": trials,
                           "reason": reason, "before": before, "after": p.confidence})
        return p

    def add_principle(self, fields: Dict[str, str], task: str, round_: int,
                      already_updated: Optional[set] = None) -> Principle:
        """Add a new falsifiable principle at confidence 0.5. If the same statement already exists
        (normalised text), the restatement counts as moderate supporting evidence instead, unless that
        principle already received evidence in this reflection (`already_updated`)."""
        text = render_principle(fields)
        key = _norm(text)
        for p in self.principles:
            if _norm(p.principle) == key:
                if p.id not in (already_updated or set()):
                    self.update_principle(p.id, "supports", 0.5, round_, reason="restated by the reflector")
                return p
        p = Principle(text, INITIAL_CONFIDENCE, 0, task, round_, round_, id=self._next_principle_id(),
                      fields={k: str(fields.get(k, "")) for k in PRINCIPLE_FIELDS})
        self.principles.append(p)
        return p

    def active_principles(self) -> List[Principle]:
        """Principles that may guide design: not refuted, most confident first."""
        live = [p for p in self.principles if p.status != "refuted"]
        return sorted(live, key=lambda p: (p.confidence, p.support_count - p.contradict_count), reverse=True)

    def add_reflection(self, round_: int, reflection: Dict[str, Any]) -> None:
        self.reflections.append({"round": round_, **reflection})

    def latest_reflection(self) -> Optional[Dict[str, Any]]:
        return self.reflections[-1] if self.reflections else None
