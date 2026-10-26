"""Reflection v3 (2026-10-20, advisor-approved): principle confidence computed from the real trials.

The LLM conjectures, the experiments refute. Every principle carries a machine-checkable claim

    {"condition": [{"dim", "value"}, ...1-2],      the architectural choice
     "reference": [{"dim", "value"}, ...],         what it is compared with (same dims; [] = baseline values)
     "direction": "increase" | "decrease",
     "min_effect": float}                          size of the claimed effect (default: the noise level)

and its confidence is the posterior probability that the claim holds under a Bayesian linear model of
all scored trials (the baseline included):

    score = b0 + sum of main effects (one per non-baseline value) + interaction terms + noise,

with Gaussian priors N(0, prior_sd^2) on main effects, N(0, interaction_prior_sd^2) on the interaction
terms that 2-term claims introduce, and the measured seed noise as the (known) noise std. Every round
the confidence is recomputed from all data, so there is no learning rate and nothing is counted twice.
No numpy: the framework does not depend on it and the systems are tiny (<= ~30 unknowns).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .search_space import BASELINE, DIMENSIONS, canonicalize, get_search_space

Term = Tuple[str, str]           # (dim, value)
Feature = Tuple[Term, ...]       # (term,) main effect, (term, term) interaction

DIRECTIONS = ("increase", "decrease")
MAX_CONDITION_TERMS = 2


def _terms(items: Any) -> List[Term]:
    return [(str(x["dim"]), str(x["value"])) for x in (items or [])]


def validate_claim(claim: Any, task: str) -> Optional[str]:
    """None if the claim is well formed, else a reason (used by the reflection validator)."""
    if not isinstance(claim, dict):
        return "claim must be an object"
    space = get_search_space(task)
    try:
        cond, ref = _terms(claim.get("condition")), _terms(claim.get("reference"))
    except (KeyError, TypeError):
        return "claim condition / reference items need dim and value"
    if not 1 <= len(cond) <= MAX_CONDITION_TERMS:
        return f"claim condition must have 1-{MAX_CONDITION_TERMS} terms"
    dims = [d for d, _ in cond]
    if len(set(dims)) != len(dims):
        return "claim condition repeats a dimension"
    for d, v in cond + ref:
        if d not in space or v not in space[d]:
            return f"claim term {d}={v} is not in the search space"
    if any(d not in dims for d, _ in ref):
        return "claim reference may only use the condition's dimensions"
    if dict(ref) and all(dict(ref).get(d, BASELINE[d]) == v for d, v in cond):
        return "claim reference equals its condition"
    if all(BASELINE[d] == v for d, v in cond) and not ref:
        return "claim condition is the baseline itself"
    if claim.get("direction") not in DIRECTIONS:
        return f"claim direction must be one of {DIRECTIONS}"
    m = claim.get("min_effect")
    if not isinstance(m, (int, float)) or not 0 <= float(m) <= 10:
        return "claim min_effect must be a number in [0, 10]"
    return None


def claim_text(claim: Dict[str, Any]) -> str:
    cond = " AND ".join(f"{d}={v}" for d, v in _terms(claim["condition"]))
    ref = " AND ".join(f"{d}={v}" for d, v in _terms(claim.get("reference"))) or "baseline values"
    sign = ">" if claim["direction"] == "increase" else "<"
    m = float(claim["min_effect"])
    return f"[{cond}] vs [{ref}]: delta {sign} {'+' if sign == '>' else '-'}{m:.2f}"


def claim_key(claim: Dict[str, Any], with_direction: bool = True) -> tuple:
    """Canonical identity of a claim: sorted condition, the reference with omitted dims filled in with
    their baseline values ([] and the explicit baseline values are the same claim), and the direction.
    min_effect is ignored (the same comparison at another threshold is a restatement)."""
    cond = sorted(_terms(claim.get("condition")))
    given = dict(_terms(claim.get("reference")))
    ref = sorted((d, given.get(d, BASELINE[d])) for d, _ in cond)
    key = (tuple(cond), tuple(ref))
    return key + ((claim.get("direction"),) if with_direction else ())


def predicted_sign(claim: Dict[str, Any], parent: Dict[str, str], child: Dict[str, str]) -> Optional[int]:
    """Does the claim predict the direction of the parent -> child change? +1 / -1, or None when the
    pair is not a clean test: the two may differ only in the claim's dimensions, and one must satisfy
    the condition while the other satisfies the reference (default: the baseline values)."""
    if not claim or not parent or not child:
        return None
    p, c = canonicalize(parent), canonicalize(child)
    cond = _terms(claim.get("condition"))
    dims = {d for d, _ in cond}
    if not dims or p == c or any(p[d] != c[d] for d in DIMENSIONS if d not in dims):
        return None
    given = dict(_terms(claim.get("reference")))
    ref = [(d, given.get(d, BASELINE[d])) for d in dims]
    sat = lambda a, terms: all(a[d] == v for d, v in terms)  # noqa: E731
    sign = 1 if claim.get("direction") == "increase" else -1
    if sat(c, cond) and sat(p, ref):
        return sign
    if sat(p, cond) and sat(c, ref):
        return -sign
    return None


def pair_verdict(claim: Dict[str, Any], sign: int, delta: float, noise_sd: float) -> str:
    """Classify one parent -> child comparison for a claim that predicts `sign`:
    support (moved in the predicted direction by at least min_effect), counterexample (more than two
    noise sd below the claimed effect, i.e. clearly not the predicted change), else inconclusive."""
    m = float(claim.get("min_effect", noise_sd))
    moved = sign * delta
    if moved >= m:
        return "support"
    if moved <= m - 2 * noise_sd:
        return "counterexample"
    return "inconclusive"


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _solve_inverse(a: List[List[float]]) -> List[List[float]]:
    """Inverse of a symmetric positive definite matrix (Gauss-Jordan with partial pivoting)."""
    n = len(a)
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        piv = m[c][c]
        m[c] = [x / piv for x in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0.0:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [row[n:] for row in m]


@dataclass
class EvidenceModel:
    features: List[Feature]
    mean: List[float]              # posterior mean, index 0 = intercept
    cov: List[List[float]]
    n_obs: int
    noise_sd: float
    counts: Dict[Feature, int]     # number of scored architectures that have each feature

    # ------------------------------------------------------------------
    def _contrast(self, cond: Sequence[Term], ref: Sequence[Term]) -> List[float]:
        """phi(condition) - phi(reference); dimensions not named stay at their baseline values."""
        def phi(assign: Dict[str, str]) -> List[float]:
            arch = {**BASELINE, **assign}
            return [1.0] + [float(all(arch[d] == v for d, v in f)) for f in self.features]
        c_assign = dict(cond)
        r_assign = {d: dict(ref).get(d, BASELINE[d]) for d in c_assign}
        a, b = phi(c_assign), phi(r_assign)
        return [x - y for x, y in zip(a, b)]

    def effect(self, cond: Sequence[Term], ref: Sequence[Term] = ()) -> Tuple[float, float]:
        c = self._contrast(cond, ref)
        mu = sum(ci * mi for ci, mi in zip(c, self.mean))
        var = sum(c[i] * c[j] * self.cov[i][j] for i in range(len(c)) for j in range(len(c)) if c[i] and c[j])
        return mu, math.sqrt(max(var, 0.0))

    def predict(self, arch: Dict[str, str]) -> Tuple[float, float]:
        """Predicted score delta vs the baseline architecture, and its posterior sd."""
        a = canonicalize(arch)
        changed = [(d, a[d]) for d in DIMENSIONS if a[d] != BASELINE[d]]
        return self.effect(changed, []) if changed else (0.0, 0.0)

    def n_relevant(self, cond: Sequence[Term]) -> int:
        """Scored architectures that satisfy the whole condition (how directly it has been tested)."""
        key = tuple(sorted(cond))
        return self.counts.get(key, 0)

    def confidence(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        cond, ref = _terms(claim["condition"]), _terms(claim.get("reference"))
        mu, sd = self.effect(cond, ref)
        m = float(claim.get("min_effect", self.noise_sd))
        if sd <= 0:
            p = 1.0 if (mu > m if claim["direction"] == "increase" else mu < -m) else 0.0
        elif claim["direction"] == "increase":
            p = 1.0 - _phi((m - mu) / sd)
        else:
            p = _phi((-m - mu) / sd)
        return {"confidence": p, "effect_mean": round(mu, 3), "effect_sd": round(sd, 3),
                "n_direct": self.n_relevant(cond), "n_obs": self.n_obs}

    def _phi_arch(self, arch: Dict[str, str]) -> List[float]:
        a = canonicalize(arch)
        return [1.0] + [float(all(a[d] == v for d, v in f)) for f in self.features]

    def _quad(self, u: List[float], v: List[float]) -> float:
        nz_u = [i for i, x in enumerate(u) if x]
        nz_v = [j for j, x in enumerate(v) if x]
        return sum(u[i] * v[j] * self.cov[i][j] for i in nz_u for j in nz_v)

    def verification_value(self, arch: Dict[str, str], claims: Sequence[Dict[str, Any]],
                           weights: Optional[Sequence[float]] = None) -> float:
        """How much training `arch` once would sharpen the claims: sum over claims of the relative
        reduction of the posterior sd of the claimed effect, 1 - sd_after / sd_now (Gaussian update with
        one more observation of `arch` at the known noise), times the claim's weight (the caller passes
        4c(1-c), so decided claims add little). Claims that compare the same contrast are counted once.
        Claims that are not modelled score 0."""
        x = self._phi_arch(arch)
        denom = self.noise_sd ** 2 + self._quad(x, x)
        total, seen = 0.0, set()
        for i, c in enumerate(claims):
            try:
                key = claim_key(c, with_direction=False)
                k = self._contrast(_terms(c["condition"]), _terms(c.get("reference")))
            except (KeyError, TypeError):
                continue
            if key in seen:
                continue
            seen.add(key)
            var = self._quad(k, k)
            if var <= 1e-12:
                continue
            after = max(var - self._quad(k, x) ** 2 / denom, 0.0)
            total += (1.0 if weights is None else weights[i]) * (1.0 - math.sqrt(after / var))
        return total

    def table(self, task: str) -> List[Dict[str, Any]]:
        """Main effects (and modelled interactions) for prompts: estimate +- sd and how often observed."""
        out = []
        for f in self.features:
            mu, sd = self.effect(list(f), [])
            if len(f) == 2:  # report the interaction term itself, not the combined effect
                i = self.features.index(f) + 1
                mu, sd = self.mean[i], math.sqrt(self.cov[i][i])
            out.append({"term": " AND ".join(f"{d}={v}" for d, v in f),
                        "kind": "interaction" if len(f) == 2 else "main effect vs baseline value",
                        "estimate": round(mu, 2), "sd": round(sd, 2), "n_trials_with_it": self.counts.get(f, 0)})
        return out


def fit(rows: List[Dict[str, Any]], baseline: Optional[Dict[str, Any]], task: str, noise_sd: float,
        claims: Sequence[Dict[str, Any]] = (), prior_sd: float = 1.0,
        interaction_prior_sd: float = 0.5) -> EvidenceModel:
    """rows: trial dicts (architecture, status, score). Only scored trials are used."""
    space = get_search_space(task)
    mains: List[Feature] = [((d, v),) for d in DIMENSIONS for v in space[d] if v != BASELINE[d]]
    inter: List[Feature] = []
    for c in claims:
        cond = tuple(sorted(_terms(c.get("condition"))))
        if len(cond) == 2 and cond not in inter:
            inter.append(cond)
    feats = mains + inter
    data = [r for r in rows if r.get("status") == "ok" and r.get("score") is not None]
    if baseline and baseline.get("status") == "ok" and baseline.get("score") is not None:
        data = [baseline] + data
    xs, ys, counts = [], [], {}
    for r in data:
        a = canonicalize(r["architecture"])
        x = [1.0] + [float(all(a[d] == v for d, v in f)) for f in feats]
        xs.append(x)
        ys.append(float(r["score"]))
        for f, xi in zip(feats, x[1:]):
            if xi:
                counts[f] = counts.get(f, 0) + 1
    k = len(feats) + 1
    s2 = max(noise_sd, 1e-6) ** 2
    y0 = baseline["score"] if baseline and baseline.get("score") is not None else (ys[0] if ys else 0.0)
    prior_mean = [y0] + [0.0] * len(feats)
    prior_var = [100.0 ** 2] + [prior_sd ** 2] * len(mains) + [interaction_prior_sd ** 2] * len(inter)
    prec = [[(1.0 / prior_var[i] if i == j else 0.0) for j in range(k)] for i in range(k)]
    rhs = [prior_mean[i] / prior_var[i] for i in range(k)]
    for x, y in zip(xs, ys):
        for i in range(k):
            if x[i]:
                rhs[i] += x[i] * y / s2
                for j in range(k):
                    if x[j]:
                        prec[i][j] += x[i] * x[j] / s2
    cov = _solve_inverse(prec)
    mean = [sum(cov[i][j] * rhs[j] for j in range(k)) for i in range(k)]
    return EvidenceModel(feats, mean, cov, len(ys), noise_sd, counts)
