"""Synthetic runner for dry runs: a hand-written score landscape plus seeded noise.

Only for testing the search logic end-to-end without GPUs. Its numbers are fake and must
never appear in the paper. The landscape deliberately contains a synergy (ECA + dilated)
and a costly-but-useless module (CBAM + depthwise) so reflection has something to find.
"""
from __future__ import annotations

import random

from ..search_space import Architecture, architecture_signature, canonicalize
from .base import BaseRunner, TrialResult

_EFFECT = {
    ("attention", "se"): 0.3, ("attention", "eca"): 0.8, ("attention", "cbam"): 0.4, ("attention", "coordatt"): 0.5,
    ("conv", "dilated"): 0.3, ("conv", "depthwise"): -0.4,
    ("fusion", "weighted"): 0.3,
    ("target_stage", "early"): -0.2, ("target_stage", "all"): 0.0, ("target_stage", "late"): 0.1,
}
_PARAMS = {("attention", "se"): 2.5, ("attention", "eca"): 0.001, ("attention", "cbam"): 2.6,
           ("attention", "coordatt"): 1.4, ("conv", "depthwise"): -9.0, ("fusion", "weighted"): 2.1}


class MockRunner(BaseRunner):
    metric = "top1 (synthetic)"

    def __init__(self, baseline_score: float = 78.0, baseline_params: float = 23.71,
                 noise_std: float = 0.15, seed: int = 0, fail_signatures: tuple = ()):
        self.baseline_score = baseline_score
        self.baseline_params = baseline_params
        self.noise_std = noise_std
        self.seed = seed
        self.fail_signatures = set(fail_signatures)

    def run(self, arch: Architecture, tag: str) -> TrialResult:
        a = canonicalize(arch)
        sig = architecture_signature(a)
        if sig in self.fail_signatures:
            return TrialResult("failed", failure_reason="oom", work_dir=tag)
        score = self.baseline_score
        if a != canonicalize({"attention": "none", "conv": "standard", "fusion": "none", "target_stage": "late"}):
            for d in ("attention", "conv", "fusion"):
                score += _EFFECT.get((d, a[d]), 0.0)
            if a["attention"] != "none" or a["conv"] != "standard":
                score += _EFFECT.get(("target_stage", a["target_stage"]), 0.0)
            if a["attention"] == "eca" and a["conv"] == "dilated":
                score += 0.4
            if a["attention"] == "cbam" and a["conv"] == "depthwise":
                score -= 0.5
        scale = {"early": 0.3, "late": 1.0, "all": 1.3}[a["target_stage"]]
        params = self.baseline_params + sum(_PARAMS.get((d, a[d]), 0.0) for d in ("attention", "conv")) * scale \
            + _PARAMS.get(("fusion", a["fusion"]), 0.0)
        rng = random.Random(f"mock-{self.seed}-{sig}")
        score += rng.gauss(0.0, self.noise_std)
        return TrialResult("ok", score=round(score, 2), params=round(params, 3), duration_s=0.0, work_dir=tag)
