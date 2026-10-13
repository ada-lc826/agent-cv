"""Design Agent (innovation 1): proposes the next candidate architecture."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..agent_client import AgentClient, AgentClientError, AgentClientFatal
from ..prompts import DESIGN_REFLECTION_ADDENDUM, DESIGN_SYSTEM, build_design_prompt
from ..schemas import design_schema, validate_design
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


class DesignAgent:
    def __init__(self, client: AgentClient, seed: int = 0, max_attempts: int = 2):
        self.client = client
        self.seed = seed
        self.max_attempts = max_attempts

    def propose(self, ctx: DesignContext) -> Tuple[Architecture, str]:
        """Return (architecture, note). Never raises: falls back to a legal local mutation."""
        space = get_search_space(ctx.task)
        system = DESIGN_SYSTEM + (DESIGN_REFLECTION_ADDENDUM if ctx.reflection_context is not None else "")
        feedback = ""
        for attempt in range(1, self.max_attempts + 1):
            user = build_design_prompt(
                task=ctx.task, metric=ctx.metric, space=space, baseline=ctx.baseline,
                history=ctx.history, best=ctx.best, failed=sorted(ctx.failed),
                search_mode=ctx.search_mode, mutation_budget=ctx.mutation_budget,
                noise_level=ctx.noise_level, reflection_context=ctx.reflection_context,
                candidate_costs=ctx.candidate_costs, feedback=feedback)
            try:
                out = self.client.complete_json("design", system, user, design_schema(ctx.task),
                                                mock_fn=lambda: self._mock(ctx, attempt))
            except AgentClientFatal:
                raise
            except AgentClientError as e:
                log.warning("design agent call failed (attempt %d): %s", attempt, e)
                feedback = ""
                continue
            reason = validate_design(out, ctx.task)
            if reason is None:
                arch = canonicalize(out["architecture"])
                reason = self._check_rules(arch, ctx)
                if reason is None:
                    return arch, f"llm: {out['rationale']}"
            log.info("design rejected (attempt %d): %s", attempt, reason)
            feedback = reason
        return self._fallback(ctx)

    # ------------------------------------------------------------------
    @staticmethod
    def _check_rules(arch: Architecture, ctx: DesignContext) -> Optional[str]:
        sig = architecture_signature(arch)
        if sig in ctx.failed:
            return f"{sig} is a known failed configuration"
        if sig in ctx.tried:
            return f"{sig} was already evaluated"
        diff = count_architecture_difference(arch, ctx.best["architecture"])
        if diff > ctx.mutation_budget:
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
            return {"architecture": best, "rationale": "mock: no untried single mutation"}
        cands.sort(key=lambda x: x[0], reverse=True)
        _, arch, d, v = cands[0]
        return {"architecture": arch, "rationale": f"mock: set {d}={v} (best mean delta so far)"}
