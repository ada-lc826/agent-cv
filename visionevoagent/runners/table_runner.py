"""TableRunner: answer every trial from a precomputed proxy score table (NAS-Bench style).

The search space has 86 distinct architectures and a proxy training with a fixed seed is deterministic
(the B run trained the same architecture in three different searches and on different days and got
identical scores), so one training per (architecture, seed) gives exactly the result every search would
get by training it again. tools/build_score_table.py trains them once; all searches with the same seed
read the same entries. Nothing about a method changes: it sees the same score, parameter count and
failures it would see online, and gpu_hours stays the nominal cost (the stored training duration).
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..search_space import Architecture, architecture_signature, canonicalize
from .base import BaseRunner, TrialResult


class ScoreTableMissing(RuntimeError):
    """The table has no entry for an architecture: a setup error, never a trial failure."""


class TableRunner(BaseRunner):
    metric = "top1"

    def __init__(self, path: str, seed: int):
        with open(path, encoding="utf-8") as f:
            table = json.load(f)
        self.meta: Dict[str, Any] = table.get("meta", {})
        self.rows: Dict[str, Dict[str, Any]] = table.get("seeds", {}).get(str(seed), {})
        self.seed = seed
        self.path = path
        if not self.rows:
            raise ScoreTableMissing(f"{path} has no entries for seed {seed}")

    def run(self, arch: Architecture, tag: str) -> TrialResult:
        sig = architecture_signature(canonicalize(arch))
        row = self.rows.get(sig)
        if row is None:
            raise ScoreTableMissing(f"{sig} (seed {self.seed}) is not in {self.path}; finish build_score_table first")
        return TrialResult(status=row["status"], score=row.get("score"), params=row.get("params"),
                           failure_reason=row.get("failure_reason", ""), duration_s=float(row.get("duration_s", 0.0)),
                           work_dir=f"table:{sig}")

    def param_table(self, task: str) -> Optional[Dict[str, float]]:
        return {sig: r["params"] for sig, r in self.rows.items() if r.get("params") is not None}
