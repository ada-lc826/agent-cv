"""Reflector Agent (innovation 2): turns new experimental evidence into confidence updates for the
existing design principles and, when warranted, proposes new falsifiable principles."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..agent_client import AgentClient, AgentClientError, AgentClientFatal
from ..memory import render_principle
from ..prompts import (REFLECTOR_HYPOTHESES_ADDENDUM, REFLECTOR_SYSTEM, REFLECTOR_SYSTEM_V3, REFLECTOR_SYSTEM_V4,
                       build_reflection_prompt)
from ..schemas import reflection_schema, validate_reflection
from ..search_space import (DIMENSIONS, all_architectures, architecture_signature,
                            count_architecture_difference, get_search_space)

log = logging.getLogger(__name__)


@dataclass
class ReflectionContext:
    task: str
    metric: str
    baseline: Dict[str, Any]
    best: Dict[str, Any]
    history: List[Dict[str, Any]]
    new_trials: List[Dict[str, Any]]          # trials since the previous reflection: the only new evidence
    failed: List[Dict[str, Any]]
    noise_level: float
    previous_principles: List[Dict[str, Any]]
    # v2: also propose untested combinations (and judge evidence less conservatively)
    propose_hypotheses: bool = False
    tried: Optional[List[str]] = None
    # v3: confidence is computed by the system (evidence.py); the Reflector sees the evidence table and
    # writes machine-checkable claims and predicted deltas instead of judging evidence
    computed_evidence: bool = False
    evidence_table: Optional[List[Dict[str, Any]]] = None
    model: Any = None                         # evidence.EvidenceModel (used by the dry-run mock only)
    library: bool = False                     # v4: verifiable principle library (pair evidence, conditional refinement)


class ReflectorAgent:
    def __init__(self, client: AgentClient, max_attempts: int = 2):
        self.client = client
        self.max_attempts = max_attempts

    def reflect(self, ctx: ReflectionContext) -> Optional[Dict[str, Any]]:
        """Return a validated reflection dict, or None if every attempt failed (search continues).
        The returned dict also carries `design_principles`: the new principles rendered as text."""
        user = build_reflection_prompt(
            task=ctx.task, metric=ctx.metric, space=get_search_space(ctx.task), baseline=ctx.baseline,
            history=ctx.history, new_trials=ctx.new_trials, best=ctx.best, failed=ctx.failed,
            noise_level=ctx.noise_level, previous_principles=ctx.previous_principles,
            tried=ctx.tried if ctx.propose_hypotheses else None,
            evidence_table=(ctx.evidence_table or []) if ctx.computed_evidence else None)
        if ctx.computed_evidence:
            system = REFLECTOR_SYSTEM_V4 if ctx.library else REFLECTOR_SYSTEM_V3
        else:
            system = REFLECTOR_SYSTEM + (REFLECTOR_HYPOTHESES_ADDENDUM if ctx.propose_hypotheses else "")
        schema = reflection_schema(ctx.task, with_hypotheses=ctx.propose_hypotheses,
                                   computed_evidence=ctx.computed_evidence)
        feedback, pending = "", None   # pending: last valid reflection whose hypotheses were partly unusable
        for attempt in range(1, self.max_attempts + 1):
            try:
                out = self.client.complete_json("reflect", system, user + feedback, schema,
                                                mock_fn=lambda: self._mock(ctx))
            except AgentClientFatal:
                raise
            except AgentClientError as e:
                log.warning("reflector call failed (attempt %d): %s", attempt, e)
                continue
            reason = validate_reflection(out, ctx.task, with_hypotheses=ctx.propose_hypotheses,
                                         computed_evidence=ctx.computed_evidence)
            if reason is not None:
                log.warning("reflection rejected (attempt %d): %s", attempt, reason)
                continue
            bad = self._unusable_hypotheses(out, ctx) if ctx.propose_hypotheses else []
            if bad and attempt < self.max_attempts:
                # B run (2026-10-21): the Reflector kept re-proposing tested architectures; ask again
                log.info("reflection: %d unusable hypotheses, asking again", len(bad))
                pending = out
                feedback = ("\n\nYOUR PREVIOUS ANSWER WAS REJECTED: these untested_combinations are not usable "
                            "(already tried, identical to the current best, or more than 2 dimensions away): "
                            + ", ".join(sig for sig, _ in bad)
                            + ". Propose untested combinations that are NOT in TRIED ARCHITECTURES.")
                continue
            return self._finish(out, ctx)
        return self._finish(pending, ctx) if pending is not None else None

    def _finish(self, out: Dict[str, Any], ctx: ReflectionContext) -> Dict[str, Any]:
        if ctx.propose_hypotheses:  # keep only usable hypotheses; record the others
            bad = dict(self._unusable_hypotheses(out, ctx))
            if bad:
                out["rejected_hypotheses"] = [{"signature": s, "reason": r} for s, r in bad.items()]
                out["untested_combinations"] = [h for h in out["untested_combinations"]
                                                if architecture_signature(h["architecture"]) not in bad]
        out["design_principles"] = [render_principle(p) for p in out["new_principles"]]
        return out

    @staticmethod
    def _unusable_hypotheses(out: Dict[str, Any], ctx: ReflectionContext) -> List[Tuple[str, str]]:
        tried = set(ctx.tried or [])
        bad = []
        for h in out.get("untested_combinations", []):
            sig = architecture_signature(h["architecture"])
            diff = count_architecture_difference(h["architecture"], ctx.best["architecture"])
            if sig in tried:
                bad.append((sig, "already tried"))
            elif not 1 <= diff <= 2:
                bad.append((sig, f"{diff} dimensions from the current best"))
        return bad

    # ------------------------------------------------------------------
    @staticmethod
    def _mock(ctx: ReflectionContext) -> Dict[str, Any]:
        """Statistical stand-in for the LLM (dry runs only).

        New principles say "IF <dim>=<value> ..."; evidence for an existing principle compares the new
        trials that use <dim>=<value> with the noise level. Enough to exercise the confidence logic."""
        def stats(rows: List[Dict[str, Any]], key: str) -> Dict[Tuple[str, str], List[float]]:
            out: Dict[Tuple[str, str], List[float]] = {}
            for r in rows:
                if r["status"] == "ok" and r.get(key) is not None:
                    for d in DIMENSIONS:
                        v = r["architecture"][d]
                        if v not in ("none", "standard"):
                            out.setdefault((d, v), []).append(r[key])
            return out

        noise = ctx.noise_level
        # patterns vs the baseline; evidence vs the best-so-far before each new trial (can be negative)
        all_stats, new_stats = stats(ctx.history, "delta_vs_baseline"), stats(ctx.new_trials, "delta_vs_previous")
        eff, ineff = [], []
        for (d, v), vals in sorted(all_stats.items()):
            mean = sum(vals) / len(vals)
            item = {"pattern": f"{d}={v}", "evidence": f"mean delta {mean:+.2f} over {len(vals)} trials",
                    "confidence": round(min(0.95, 0.4 + 0.15 * len(vals)), 2)}
            (eff if mean > noise else ineff if mean < -noise else []).append(item)

        updates, covered = [], set()
        for p in ctx.previous_principles:
            m = re.search(r"(\w+)=(\w+)", p.get("fields", {}).get("if", "") or p.get("principle", ""))
            if not m:
                continue
            key = (m.group(1), m.group(2))
            covered.add(key)
            vals = new_stats.get(key)
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            kind = "supports" if mean > noise else "contradicts" if mean < -noise else "neutral"
            updates.append({"principle_id": p["id"], "evidence": kind,
                            "strength": round(min(1.0, abs(mean) / (2 * noise) if noise else 1.0), 2),
                            "trials": "new trials", "reason": f"mock: mean delta {mean:+.2f} for {key[0]}={key[1]}"})
        new = [{"if": f"{p['pattern']}", "scope": "this backbone at the proxy schedule",
                "then": "top-1 improves by more than the noise level", "because": "mock mechanism",
                "disproved_if": "trials with this choice do not beat the baseline by more than the noise level"}
               for p in eff if tuple(p["pattern"].split("=")) not in covered][:3]
        n = sum(len(v) for v in all_stats.values())
        out = {
            "effective_patterns": eff, "ineffective_patterns": ineff, "synergies": [],
            "principle_updates": updates, "new_principles": new,
            "next_search_direction": [f"Keep {p['pattern']} and vary another dimension" for p in eff[:2]],
            "data_sufficiency": "insufficient" if n < 4 else ("limited" if n < 12 else "adequate"),
        }
        if ctx.computed_evidence:  # v3: claims instead of evidence judgements; one interaction conjecture
            del out["principle_updates"]
            for p in out["new_principles"]:
                d, v = p["if"].split("=")
                p["claim"] = {"condition": [{"dim": d, "value": v}], "reference": [], "direction": "increase",
                              "min_effect": noise}
            singles = [tuple(p["pattern"].split("=")) for p in eff]
            pairs = [(a, b) for i, a in enumerate(singles) for b in singles[i + 1:] if a[0] != b[0]]
            known = {str(p.get("claim", {}).get("condition")) for p in ctx.previous_principles}
            for a, b in pairs:
                cond = [{"dim": a[0], "value": a[1]}, {"dim": b[0], "value": b[1]}]
                if str(cond) not in known and len(out["new_principles"]) < 3:
                    out["new_principles"].append({
                        "if": f"{a[0]}={a[1]} AND {b[0]}={b[1]}", "scope": "this backbone at the proxy schedule",
                        "then": "the joint change beats the baseline by more than either alone",
                        "because": "mock: complementary mechanisms", "disproved_if": "the joint change does not beat the baseline",
                        "claim": {"condition": cond, "reference": [], "direction": "increase", "min_effect": 2 * noise}})
                    break
        if ctx.propose_hypotheses:  # untried architectures within 2 dimensions of the best
            tried = set(ctx.tried or [])
            cands = [a for a in all_architectures(ctx.task) if architecture_signature(a) not in tried
                     and 1 <= count_architecture_difference(a, ctx.best["architecture"]) <= 2]
            if ctx.model is not None:  # mock "prediction": the evidence model's mean
                cands.sort(key=lambda a: -ctx.model.predict(a)[0])
            out["untested_combinations"] = [{"architecture": a, "rationale": "mock: untested nearby combination",
                                             "principle_ids": []} for a in cands[:2]] or [
                {"architecture": dict(ctx.best["architecture"]), "rationale": "mock: nothing left", "principle_ids": []}]
            if ctx.computed_evidence:
                for h in out["untested_combinations"]:
                    h["predicted_delta"] = round(ctx.model.predict(h["architecture"])[0], 2) if ctx.model else 0.0
        return out
