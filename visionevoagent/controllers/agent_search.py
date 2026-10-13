"""VisionEvoAgent controller.

use_reflection=False -> innovation 1 only (Agent-Guided Module Exploration, history in the prompt).
use_reflection=True  -> + innovation 2 (Reflective Architecture Evolution).
Sub-switches support the ablations of the research plan:
  D-1 use_failure_memory=False      D-2 use_principle_memory=False
  D-3 mutation_budget = 4           D-4 explore_every=0
  D-5 reflect_every_n in {1, 2, 4}
  D-6 reflection_style="rule"       LLMENAS-style trigger: "best did not improve -> change direction,
                                    improved -> refine", no LLM reflection, no principles
  D-7 reflection_style="immediate"  RZ-NAS-style: reflect on the last trial only, every round,
                                    nothing accumulated across rounds
  show_param_costs=False            Design Agent without the per-candidate parameter-cost table
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from ..agents import DesignAgent, DesignContext, ReflectionContext, ReflectorAgent
from ..memory import ArchitectureMemory
from ..search_space import architecture_signature, local_mutations
from .base import BaseController, Proposal

log = logging.getLogger(__name__)

REFLECTION_STYLES = ("full", "rule", "immediate")


@dataclass
class AgentSearchConfig:
    use_reflection: bool = True
    use_failure_memory: bool = True
    use_principle_memory: bool = True
    reflection_style: str = "full"      # full | rule (D-6) | immediate (D-7)
    reflect_every_n: int = 2
    mutation_budget: int = 1
    stagnation_budget: int = 2          # budget used after `stagnation_rounds` rounds without a new best
    stagnation_rounds: int = 3
    explore_every: int = 4              # every k-th round is an exploration round (0 = never)
    max_principles_in_prompt: int = 8
    show_param_costs: bool = True       # per-candidate parameter-cost table in the design prompt


def trial_rows(memory: ArchitectureMemory) -> List[Dict[str, Any]]:
    return [asdict(t) for t in memory.trials]


def best_info(memory: ArchitectureMemory) -> Dict[str, Any]:
    b = memory.best_trial() or memory.baseline
    return {"architecture": b.architecture, "score": b.score, "params": b.params}


def baseline_info(memory: ArchitectureMemory) -> Dict[str, Any]:
    b = memory.baseline
    return {"architecture": b.architecture, "score": b.score, "params": b.params}


class AgentSearchController(BaseController):
    name = "agent"

    def __init__(self, task: str, seed: int, design: DesignAgent, reflector: Optional[ReflectorAgent],
                 cfg: AgentSearchConfig, metric: str, noise_level: float,
                 param_table_fn: Optional[Callable[[], Optional[Dict[str, float]]]] = None):
        super().__init__(task, seed)
        if cfg.reflection_style not in REFLECTION_STYLES:
            raise ValueError(f"reflection_style must be one of {REFLECTION_STYLES}")
        self.design = design
        self.reflector = reflector
        self.cfg = cfg
        self.metric = metric
        self.noise_level = noise_level
        self._param_table_fn = param_table_fn
        self._param_table: Optional[Dict[str, float]] = None
        if cfg.use_reflection and cfg.reflection_style != "rule" and reflector is None:
            raise ValueError("this reflection style requires a ReflectorAgent")
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

    def _candidate_costs(self, memory: ArchitectureMemory, best_arch, budget: int,
                         excluded: set) -> Optional[List[Dict[str, Any]]]:
        if not self.cfg.show_param_costs or self._param_table_fn is None:
            return None
        if self._param_table is None:
            self._param_table = self._param_table_fn() or {}
        if not self._param_table:
            return None
        out = []
        for a in local_mutations(best_arch, self.task, budget):
            sig = architecture_signature(a)
            if sig not in excluded and sig in self._param_table:
                out.append({"architecture": a, "params": self._param_table[sig]})
        return out

    def _reflection_context(self, memory: ArchitectureMemory) -> Optional[Dict[str, Any]]:
        if not self.cfg.use_reflection:
            return None
        ref = memory.latest_reflection() or {}
        if self.cfg.reflection_style == "rule":
            return {"next_search_direction": ref.get("next_search_direction", [])}
        ctx: Dict[str, Any] = {k: ref.get(k, []) for k in
                               ("effective_patterns", "ineffective_patterns", "synergies", "next_search_direction")}
        if self._uses_principle_memory():
            ctx["design_principles"] = [
                {"id": p.id, "principle": p.principle, "confidence": round(p.confidence, 3), "status": p.status,
                 "supports": p.support_count, "contradicts": p.contradict_count,
                 **({"transferred_from": p.transferred_from} if p.transferred_from else {})}
                for p in memory.active_principles()[: self.cfg.max_principles_in_prompt]]
        else:  # immediate style or principle memory ablated: only the latest reflection's new principles
            ctx["design_principles"] = ref.get("design_principles", [])
        return ctx

    def _uses_principle_memory(self) -> bool:
        return self.cfg.reflection_style == "full" and self.cfg.use_principle_memory

    def propose(self, memory: ArchitectureMemory, round_: int) -> Proposal:
        mode = self._search_mode(round_)
        failed = memory.failed_signatures() if self.cfg.use_failure_memory else set()
        best = best_info(memory)
        budget = self._mutation_budget(memory, round_)
        ctx = DesignContext(
            task=self.task, metric=self.metric, baseline=baseline_info(memory),
            best=best, history=trial_rows(memory),
            tried=memory.tried_signatures(), failed=failed, search_mode=mode,
            mutation_budget=budget, noise_level=self.noise_level,
            reflection_context=self._reflection_context(memory), round=round_,
            candidate_costs=self._candidate_costs(memory, best["architecture"], budget,
                                                  memory.tried_signatures() | failed))
        arch, note, cited_ids = self.design.propose(ctx)
        cited = []
        for pid in cited_ids:  # snapshot the confidence the Design Agent saw (calibration analysis)
            p = memory.principle(pid)
            if p is not None:
                cited.append({"id": p.id, "confidence": round(p.confidence, 4), "status": p.status})
        return Proposal(arch, note=note, search_mode=mode, cited_principles=cited)

    # ------------------------------------------------------------------ reflection
    def after_trial(self, memory: ArchitectureMemory, round_: int) -> Optional[dict]:
        if not self.cfg.use_reflection:
            return None
        style = self.cfg.reflection_style
        every = 1 if style == "immediate" else self.cfg.reflect_every_n
        if round_ % every != 0:
            return None
        if style == "rule":
            ref = self._rule_reflection(memory, round_)
            memory.add_reflection(round_, ref)
            return ref

        immediate = style == "immediate"
        history = trial_rows(memory)
        last_ref = memory.latest_reflection()
        since = last_ref["round"] if last_ref else 0
        new_trials = history[-1:] if immediate else [r for r in history if r["round"] > since]
        keep = self._uses_principle_memory()
        ctx = ReflectionContext(
            task=self.task, metric=self.metric, baseline=baseline_info(memory),
            best=best_info(memory), history=history[-1:] if immediate else history, new_trials=new_trials,
            failed=[] if immediate or not self.cfg.use_failure_memory else [asdict(f) for f in memory.failures],
            noise_level=self.noise_level,
            previous_principles=[asdict(p) for p in memory.active_principles()] if keep else [])
        ref = self.reflector.reflect(ctx)
        if ref is None:
            log.warning("round %d: reflection failed, keeping previous knowledge", round_)
            return None
        if keep:
            ref["applied"] = self._apply_reflection(memory, ref, round_)
        memory.add_reflection(round_, ref)
        return ref

    def _apply_reflection(self, memory: ArchitectureMemory, ref: Dict[str, Any], round_: int) -> Dict[str, Any]:
        """Evidence updates (one per principle per reflection, unknown ids ignored), then new principles."""
        updated, ignored = [], []
        for u in ref["principle_updates"]:
            pid = u["principle_id"]
            if pid in updated:
                ignored.append(pid)
                continue
            p = memory.update_principle(pid, u["evidence"], u["strength"], round_,
                                        reason=u.get("reason", ""), trials=u.get("trials", ""))
            (updated if p is not None else ignored).append(pid)
        added = [memory.add_principle(f, self.task, round_, already_updated=set(updated)).id
                 for f in ref["new_principles"]]
        return {"updated": updated, "ignored": ignored, "added": added}

    def _rule_reflection(self, memory: ArchitectureMemory, round_: int) -> Dict[str, Any]:
        """D-6: LLMENAS's trigger (Lai et al., TEVC 2026) without any induced knowledge. Compare the best
        score now with the best score `reflect_every_n` rounds ago and emit one fixed directive."""
        window = self.cfg.reflect_every_n
        before = [t for t in memory.trials if t.is_best and t.round <= round_ - window]
        best_before = before[-1].score if before else memory.baseline.score
        best_now = (memory.best_trial() or memory.baseline).score
        last = memory.trials[-1]
        change = f"last tried {last.signature} ({'score ' + format(last.score, '.2f') if last.score is not None else 'failed'})"
        if best_now > best_before:
            directive = (f"Refine: the best score improved over the last {window} rounds ({best_before:.2f} -> "
                         f"{best_now:.2f}). Keep the current direction and make a small change around the "
                         f"current best; {change}.")
        else:
            directive = (f"Change direction: the best score did not improve over the last {window} rounds "
                         f"(still {best_now:.2f}). Try a dimension or module family different from the recent "
                         f"rounds; {change}.")
        return {"style": "rule", "improved": best_now > best_before, "next_search_direction": [directive]}
