"""VisionEvoAgent controller.

use_reflection=False -> innovation 1 only (Agent-Guided Module Exploration, history in the prompt).
use_reflection=True  -> + innovation 2 (Reflective Architecture Evolution).
Sub-switches support the ablations D-1..D-5 of the research plan.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from ..agents import DesignAgent, DesignContext, ReflectionContext, ReflectorAgent
from ..memory import ArchitectureMemory
from .base import BaseController, Proposal

log = logging.getLogger(__name__)


@dataclass
class AgentSearchConfig:
    use_reflection: bool = True
    use_failure_memory: bool = True
    use_principle_memory: bool = True
    reflect_every_n: int = 2
    recent_n: int = 4
    mutation_budget: int = 1
    stagnation_budget: int = 2          # budget used after `stagnation_rounds` rounds without a new best
    stagnation_rounds: int = 3
    explore_every: int = 4              # every k-th round is an exploration round (0 = never)
    max_principles_in_prompt: int = 8


def trial_rows(memory: ArchitectureMemory) -> List[Dict[str, Any]]:
    return [asdict(t) for t in memory.trials]


def best_info(memory: ArchitectureMemory) -> Dict[str, Any]:
    b = memory.best_trial() or memory.baseline
    return {"architecture": b.architecture, "score": b.score, "params": b.params}


class AgentSearchController(BaseController):
    name = "agent"

    def __init__(self, task: str, seed: int, design: DesignAgent, reflector: Optional[ReflectorAgent],
                 cfg: AgentSearchConfig, metric: str, noise_level: float):
        super().__init__(task, seed)
        self.design = design
        self.reflector = reflector
        self.cfg = cfg
        self.metric = metric
        self.noise_level = noise_level
        if cfg.use_reflection and reflector is None:
            raise ValueError("use_reflection=True requires a ReflectorAgent")
        self.name = "agent_reflect" if cfg.use_reflection else "agent"

    # ------------------------------------------------------------------
    def _search_mode(self, round_: int) -> str:
        k = self.cfg.explore_every
        return "explore" if k and round_ % k == 0 else "exploit"

    def _mutation_budget(self, memory: ArchitectureMemory, round_: int) -> int:
        best = memory.best_trial()
        last_improve = best.round if best else 0
        stagnated = round_ - 1 - last_improve >= self.cfg.stagnation_rounds
        return self.cfg.stagnation_budget if stagnated else self.cfg.mutation_budget

    def _reflection_context(self, memory: ArchitectureMemory) -> Optional[Dict[str, Any]]:
        if not self.cfg.use_reflection:
            return None
        ref = memory.latest_reflection() or {}
        ctx: Dict[str, Any] = {k: ref.get(k, []) for k in
                               ("effective_patterns", "ineffective_patterns", "synergies", "next_search_direction")}
        if self.cfg.use_principle_memory:
            ranked = sorted(memory.principles, key=lambda p: (p.support_count, p.confidence), reverse=True)
            ctx["design_principles"] = [asdict(p) for p in ranked[: self.cfg.max_principles_in_prompt]]
        else:
            ctx["design_principles"] = ref.get("design_principles", [])
        return ctx

    def propose(self, memory: ArchitectureMemory, round_: int) -> Proposal:
        mode = self._search_mode(round_)
        failed = memory.failed_signatures() if self.cfg.use_failure_memory else set()
        ctx = DesignContext(
            task=self.task, metric=self.metric,
            baseline={"architecture": memory.baseline.architecture, "score": memory.baseline.score,
                      "params": memory.baseline.params},
            best=best_info(memory), history=trial_rows(memory),
            tried=memory.tried_signatures(), failed=failed, search_mode=mode,
            mutation_budget=self._mutation_budget(memory, round_), noise_level=self.noise_level,
            reflection_context=self._reflection_context(memory), round=round_)
        arch, note = self.design.propose(ctx)
        return Proposal(arch, note=note, search_mode=mode)

    def after_trial(self, memory: ArchitectureMemory, round_: int) -> Optional[dict]:
        if not self.cfg.use_reflection or round_ % self.cfg.reflect_every_n != 0:
            return None
        ctx = ReflectionContext(
            task=self.task, metric=self.metric,
            baseline={"architecture": memory.baseline.architecture, "score": memory.baseline.score,
                      "params": memory.baseline.params},
            best=best_info(memory), history=trial_rows(memory), recent_n=self.cfg.recent_n,
            failed=[asdict(f) for f in memory.failures] if self.cfg.use_failure_memory else [],
            noise_level=self.noise_level,
            previous_principles=[asdict(p) for p in memory.principles] if self.cfg.use_principle_memory else [])
        ref = self.reflector.reflect(ctx)
        if ref is None:
            log.warning("round %d: reflection failed, keeping previous knowledge", round_)
            return None
        memory.add_reflection(round_, ref)
        if self.cfg.use_principle_memory:
            conf = {"insufficient": 0.4, "limited": 0.6, "adequate": 0.8}[ref["data_sufficiency"]]
            memory.merge_principles(ref["design_principles"], conf, self.task, round_)
        return ref
