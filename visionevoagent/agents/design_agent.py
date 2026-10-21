"""Design Agent (innovation 1): proposes the next candidate architecture."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..agent_client import AgentClient, AgentClientError, AgentClientFatal
from ..prompts import (DESIGN_COMPUTED_ADDENDUM, DESIGN_REFLECTION_ADDENDUM, DESIGN_REFLECTION_ONLY_ADDENDUM,
                       DESIGN_SYSTEM, build_design_prompt)
from ..schemas import MAX_PROPOSAL_ATTEMPTS, attempt_outcome, design_schema, validate_design
from ..search_space import (DIMENSIONS, Architecture, architecture_signature, canonicalize,
                            count_architecture_difference, get_search_space, random_local_mutation)

log = logging.getLogger(__name__)


@dataclass
class DesignContext:
    task: str
    metric: str
    baseline: Dict[str, Any]                      # {"architecture", "score", "params"}
    best: Dict[str, Any]                          # {"architecture", "score", "params"}
    history: List[Dict[str, Any]]
    tried: set
    failed: set
    search_mode: str                              # exploit | explore
    mutation_budget: int
    noise_level: float
    reflection_context: Optional[Dict[str, Any]]  # None when reflection is disabled
    round: int
    candidate_costs: Optional[List[Dict[str, Any]]] = None  # [{"architecture", "params"}]; None = no cost table
    # v2 reflection: the raw history is hidden (REFLECTION_CONTEXT is the only account of the search) and
    # the Reflector's untested combinations are offered; those may differ from the best in 2 dimensions
    show_history: bool = True
    hypotheses: Optional[List[Dict[str, Any]]] = None       # [{"architecture", "rationale", "principle_ids"}]
    # v3: effect estimates computed from the trials; hypotheses then also carry "predicted_delta",
    # "model_prediction" and "tests" ([{"id", "confidence"}] of the principles they test)
    evidence_table: Optional[List[Dict[str, Any]]] = None
    candidates: Optional[List[str]] = None   # untried signatures within the budget (reflection-only view)

HYPOTHESIS_BUDGET = 2


class DesignAgent:
    def __init__(self, client: AgentClient, seed: int = 0, max_attempts: int = MAX_PROPOSAL_ATTEMPTS):
        self.client = client
        self.seed = seed
        self.max_attempts = max_attempts

    def propose(self, ctx: DesignContext, record: Optional[List[str]] = None
                ) -> Tuple[Architecture, str, List[str]]:
        """Return (architecture, note, cited principle ids). Never raises: after max_attempts rejected
        answers it falls back to a random untried legal local mutation (which cites nothing). The outcome
        of every attempt is appended to `record` (ok | duplicate | invalid | call_error)."""
        record = record if record is not None else []
        space = get_search_space(ctx.task)
        system = (DESIGN_SYSTEM + (DESIGN_REFLECTION_ADDENDUM if ctx.reflection_context is not None else "")
                  + ("" if ctx.show_history else DESIGN_REFLECTION_ONLY_ADDENDUM)
                  + (DESIGN_COMPUTED_ADDENDUM if ctx.evidence_table is not None else ""))
        hyp_sigs = {architecture_signature(h["architecture"]) for h in (ctx.hypotheses or [])}
        known_ids = {p.get("id") for p in (ctx.reflection_context or {}).get("design_principles", [])
                     if isinstance(p, dict) and p.get("id")}
        schema = design_schema(ctx.task, with_citations=bool(known_ids))
        feedback = ""
        for attempt in range(1, self.max_attempts + 1):
            user = build_design_prompt(
                task=ctx.task, metric=ctx.metric, space=space, baseline=ctx.baseline,
                history=ctx.history, best=ctx.best, failed=sorted(ctx.failed),
                search_mode=ctx.search_mode, mutation_budget=ctx.mutation_budget,
                noise_level=ctx.noise_level, reflection_context=ctx.reflection_context,
                candidate_costs=ctx.candidate_costs, feedback=feedback, show_history=ctx.show_history,
                tried=sorted(ctx.tried), hypotheses=ctx.hypotheses, evidence_table=ctx.evidence_table,
                candidates=ctx.candidates)
            try:
                out = self.client.complete_json("design", system, user, schema,
                                                mock_fn=lambda: self._mock(ctx, attempt))
            except AgentClientFatal:
                raise
            except AgentClientError as e:
                log.warning("design agent call failed (attempt %d): %s", attempt, e)
                record.append("call_error")
                feedback = ""
                continue
            reason = validate_design(out, ctx.task)
            if reason is None:
                arch = canonicalize(out["architecture"])
                reason = self._check_rules(arch, ctx, hyp_sigs)
            record.append(attempt_outcome(reason))
            if reason is None:
                cited = [c for c in dict.fromkeys(out.get("cited_principles", [])) if c in known_ids]
                tag = "llm[hypothesis]" if architecture_signature(arch) in hyp_sigs else "llm"
                return arch, f"{tag}: {out['rationale']}", cited
            log.info("design rejected (attempt %d): %s", attempt, reason)
            feedback = reason
        arch, note = self._fallback(ctx)
        return arch, note, []

    # ------------------------------------------------------------------
    @staticmethod
    def _check_rules(arch: Architecture, ctx: DesignContext, hyp_sigs: Optional[set] = None) -> Optional[str]:
        sig = architecture_signature(arch)
        if sig in ctx.failed:
            return f"{sig} is a known failed configuration"
        if sig in ctx.tried:
            return f"{sig} was already evaluated"
        budget = max(ctx.mutation_budget, HYPOTHESIS_BUDGET) if sig in (hyp_sigs or set()) else ctx.mutation_budget
        diff = count_architecture_difference(arch, ctx.best["architecture"])
        if diff > budget:
            return f"changes {diff} dimensions vs the current best, exceeding MUTATION_BUDGET={ctx.mutation_budget}"
        return None

    def _fallback(self, ctx: DesignContext) -> Tuple[Architecture, str]:
        rng = random.Random(f"{self.seed}-fallback-{ctx.round}")
        excluded = ctx.tried | ctx.failed
        for budget in range(ctx.mutation_budget, len(DIMENSIONS) + 1):
            arch = random_local_mutation(ctx.best["architecture"], ctx.task, rng, excluded, budget)
            if arch is not None:
                return arch, f"fallback: random legal mutation (budget {budget})"
        raise RuntimeError("search space exhausted")

    # ------------------------------------------------------------------
    def _mock(self, ctx: DesignContext, attempt: int) -> Dict[str, Any]:
        """Deterministic stand-in for the LLM (dry runs only): greedy on per-value mean delta."""
        rng = random.Random(f"{self.seed}-mock-{ctx.round}-{attempt}")
        if ctx.hypotheses and ctx.evidence_table is not None:
            # v3: exploit -> highest predicted gain; explore -> tests the principle closest to 0.5
            def informativeness(h):
                return min((abs(t["confidence"] - 0.5) for t in h.get("tests", [])), default=1.0)
            def model_mean(h):
                return float(str(h.get("model_prediction", "0")).split()[0])
            pick = (max(ctx.hypotheses, key=model_mean) if ctx.search_mode == "exploit"
                    else min(ctx.hypotheses, key=informativeness))
            return {"architecture": pick["architecture"], "rationale": f"mock: {ctx.search_mode} hypothesis",
                    "cited_principles": list(pick.get("principle_ids", []))}
        if ctx.hypotheses and ctx.search_mode == "exploit":  # v2: test the Reflector's first hypothesis
            h = ctx.hypotheses[0]
            return {"architecture": h["architecture"], "rationale": "mock: test the reflector's hypothesis",
                    "cited_principles": list(h.get("principle_ids", []))}
        space = get_search_space(ctx.task)
        stats: Dict[Tuple[str, str], List[float]] = {}
        for r in ctx.history:
            if r["status"] == "ok" and r.get("delta_vs_baseline") is not None:
                for d in DIMENSIONS:
                    stats.setdefault((d, r["architecture"][d]), []).append(r["delta_vs_baseline"])
        best = ctx.best["architecture"]
        cands = []
        for d in DIMENSIONS:
            for v in space[d]:
                if v == best[d]:
                    continue
                arch = canonicalize({**best, d: v})
                sig = architecture_signature(arch)
                if sig in ctx.tried or sig in ctx.failed:
                    continue
                vals = stats.get((d, v))
                prior = sum(vals) / len(vals) if vals else (0.5 if ctx.search_mode == "explore" else 0.0)
                cands.append((prior + rng.uniform(0, 0.05), arch, d, v))
        if not cands:
            return {"architecture": best, "rationale": "mock: no untried single mutation", "cited_principles": []}
        cands.sort(key=lambda x: x[0], reverse=True)
        _, arch, d, v = cands[0]
        principles = (ctx.reflection_context or {}).get("design_principles", [])
        cited = [p["id"] for p in principles if isinstance(p, dict) and f"{d}={v}" in p.get("principle", "")]
        return {"architecture": arch, "rationale": f"mock: set {d}={v} (best mean delta so far)",
                "cited_principles": cited}
