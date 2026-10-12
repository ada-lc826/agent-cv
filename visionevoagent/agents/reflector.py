"""Reflector Agent (innovation 2): induces design knowledge from the experiment history."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..agent_client import AgentClient, AgentClientError, AgentClientFatal
from ..prompts import REFLECTOR_SYSTEM, build_reflection_prompt
from ..schemas import REFLECTION_SCHEMA, validate_reflection
from ..search_space import DIMENSIONS, get_search_space

log = logging.getLogger(__name__)


@dataclass
class ReflectionContext:
    task: str
    metric: str
    baseline: Dict[str, Any]
    best: Dict[str, Any]
    history: List[Dict[str, Any]]
    recent_n: int
    failed: List[Dict[str, Any]]
    noise_level: float
    previous_principles: List[Dict[str, Any]]


class ReflectorAgent:
    def __init__(self, client: AgentClient, max_attempts: int = 2):
        self.client = client
        self.max_attempts = max_attempts

    def reflect(self, ctx: ReflectionContext) -> Optional[Dict[str, Any]]:
        """Return a validated reflection dict, or None if every attempt failed (search continues)."""
        user = build_reflection_prompt(
            task=ctx.task, metric=ctx.metric, space=get_search_space(ctx.task), baseline=ctx.baseline,
            history=ctx.history, recent_n=ctx.recent_n, best=ctx.best, failed=ctx.failed,
            noise_level=ctx.noise_level, previous_principles=ctx.previous_principles)
        for attempt in range(1, self.max_attempts + 1):
            try:
                out = self.client.complete_json("reflect", REFLECTOR_SYSTEM, user, REFLECTION_SCHEMA,
                                                mock_fn=lambda: self._mock(ctx))
            except AgentClientFatal:
                raise
            except AgentClientError as e:
                log.warning("reflector call failed (attempt %d): %s", attempt, e)
                continue
            reason = validate_reflection(out)
            if reason is None:
                return out
            log.warning("reflection rejected (attempt %d): %s", attempt, reason)
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _mock(ctx: ReflectionContext) -> Dict[str, Any]:
        """Statistical stand-in for the LLM (dry runs only): mean delta per module value."""
        stats: Dict[Tuple[str, str], List[float]] = {}
        for r in ctx.history:
            if r["status"] == "ok" and r.get("delta_vs_baseline") is not None:
                for d in DIMENSIONS:
                    v = r["architecture"][d]
                    if v not in ("none", "standard"):
                        stats.setdefault((d, v), []).append(r["delta_vs_baseline"])
        eff, ineff = [], []
        for (d, v), vals in sorted(stats.items()):
            mean = sum(vals) / len(vals)
            conf = round(min(0.95, 0.4 + 0.15 * len(vals)), 2)
            item = {"pattern": f"{d}={v}", "evidence": f"mean delta {mean:+.2f} over {len(vals)} trials",
                    "confidence": conf}
            if mean > ctx.noise_level:
                eff.append(item)
            elif mean < -ctx.noise_level:
                ineff.append(item)
        principles = [f"Prefer {p['pattern']}" for p in eff[:3]] + [f"Avoid {p['pattern']}" for p in ineff[:2]]
        if not principles:
            principles = ["Evidence is still insufficient; keep exploring single-module changes."]
        n = sum(len(v) for v in stats.values())
        return {
            "effective_patterns": eff,
            "ineffective_patterns": ineff,
            "synergies": [],
            "design_principles": principles[:5],
            "next_search_direction": [f"Keep {p['pattern']} and vary another dimension" for p in eff[:2]],
            "data_sufficiency": "insufficient" if n < 4 else ("limited" if n < 12 else "adequate"),
        }
