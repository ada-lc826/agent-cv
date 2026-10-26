"""Hypothesis-Guided Selection (framework agreed with the advisor, 2026-10-26).

The Design Agent proposes several candidates, each with a parent (the tried architecture it modifies)
and an explicit expected improvement over that parent. The selection does not take the LLM's favourite;
it scores every candidate on three terms and trains the best-scoring one:

  gain          expected improvement over the current best, EI(mu, s; best) / noise, where (mu, s) combine
                the evidence model's posterior prediction with the Design Agent's stated expectation used
                as a Gaussian prior (sd = llm_prior_sd): the LLM's knowledge counts most while there are
                few data and the data take over as trials accumulate
  uncertainty   mean 4c(1-c) over the principles the candidate bears on (1 at confidence 0.5, 0 when
                decided): testing principles that are still open
  verification  sum over all principle claims of the relative reduction of the claimed effect's posterior
                sd if this candidate were trained (EvidenceModel.verification_value): how decisively the
                experiment would confirm or refute the library

  score = w_gain * gain + w_uncertainty * uncertainty + w_verification * verification

with separate weights for exploit and explore rounds. selection="llm" (ablation) trains the Design
Agent's first valid candidate instead; the three terms are still recorded for it.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from .evidence import EvidenceModel, predicted_sign
from .search_space import DIMENSIONS, from_signature

DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "exploit": {"gain": 1.0, "uncertainty": 0.5, "verification": 0.5},
    "explore": {"gain": 0.5, "uncertainty": 1.0, "verification": 1.5},
}
SELECTION_MODES = ("guided", "llm", "none")   # none = no candidate set (classic single proposal)


def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def expected_improvement(mu: float, sd: float, best: float) -> float:
    if sd <= 1e-9:
        return max(mu - best, 0.0)
    z = (mu - best) / sd
    return (mu - best) * _norm_cdf(z) + sd * _norm_pdf(z)


def combine(mu: float, sd: float, prior_mean: Optional[float], prior_sd: float) -> (float, float):
    """Posterior of the model prediction combined with the LLM's expectation as a Gaussian prior."""
    if prior_mean is None or prior_sd <= 0:
        return mu, sd
    sd = max(sd, 1e-6)
    prec = 1.0 / sd ** 2 + 1.0 / prior_sd ** 2
    return (mu / sd ** 2 + prior_mean / prior_sd ** 2) / prec, math.sqrt(1.0 / prec)


def bears_on(claim: Dict[str, Any], parent: Dict[str, str], child: Dict[str, str]) -> bool:
    """The candidate tests the claim cleanly (parent -> child), or at least realises its condition."""
    if predicted_sign(claim, parent, child) is not None:
        return True
    cond = claim.get("condition") or []
    return bool(cond) and all(child.get(t["dim"]) == t["value"] for t in cond)


def score_candidates(candidates: List[Dict[str, Any]], model: EvidenceModel,
                     principles: Sequence[Dict[str, Any]], parent_deltas: Dict[str, float],
                     best_delta: float, noise: float, mode: str, llm_prior_sd: float = 0.75,
                     weights: Optional[Dict[str, Dict[str, float]]] = None) -> List[Dict[str, Any]]:
    """candidates: [{"architecture", "parent" (signature), "expected_improvement" (vs parent)}].
    principles: [{"id", "confidence", "claim"}] (active, with claims). parent_deltas: signature ->
    score delta vs baseline of every scored architecture (baseline 0). Returns one record per candidate
    (same order) with the prediction and the three terms."""
    w = (weights or DEFAULT_WEIGHTS).get(mode, DEFAULT_WEIGHTS["exploit"])
    claims = [p["claim"] for p in principles if p.get("claim")]
    out = []
    for c in candidates:
        arch, psig = c["architecture"], c["parent"]
        pdelta = parent_deltas.get(psig, 0.0)
        parent = from_signature(psig)
        mu, sd = model.predict(arch)
        e = c.get("expected_improvement")
        prior = pdelta + float(e) if isinstance(e, (int, float)) else None
        mu_c, sd_c = combine(mu, sd, prior, llm_prior_sd)
        gain = expected_improvement(mu_c, sd_c, best_delta) / max(noise, 1e-6)
        touched = [p for p in principles if p.get("claim") and bears_on(p["claim"], parent, arch)]
        unc = (sum(4 * p["confidence"] * (1 - p["confidence"]) for p in touched) / len(touched)) if touched else 0.0
        ver = model.verification_value(arch, claims)
        score = w["gain"] * gain + w["uncertainty"] * unc + w["verification"] * ver
        out.append({"signature": "__".join(arch[d] for d in DIMENSIONS), "parent": psig,
                    "design_expected": e, "model_mean": round(mu - pdelta, 3), "model_sd": round(sd, 3),
                    "combined_mean": round(mu_c - pdelta, 3), "combined_sd": round(sd_c, 3),
                    "gain": round(gain, 3), "uncertainty": round(unc, 3), "verification": round(ver, 3),
                    "principles_touched": [p["id"] for p in touched], "score": round(score, 3)})
    return out


def choose(scored: List[Dict[str, Any]], selection: str) -> int:
    if not scored:
        raise ValueError("no candidates")
    if selection == "llm":
        return 0
    return max(range(len(scored)), key=lambda i: (scored[i]["score"], -i))
