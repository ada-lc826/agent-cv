"""VisionEvoAgent controller.

use_reflection=False -> innovation 1 only (Agent-Guided Module Exploration, history in the prompt).
use_reflection=True  -> + innovation 2 (Reflective Architecture Evolution).
Sub-switches support the ablations of the research plan:
  D-1 use_failure_memory=False      D-2 use_principle_memory=False
  D-3 mutation_budget = 4           D-4 explore_every=0
  D-5 reflect_every_n in {1, 2, 4} (default 1)
  D-6 reflection_style="rule"       LLMENAS-style trigger: "best did not improve -> change direction,
                                    improved -> refine", no LLM reflection, no principles
  D-7 reflection_style="immediate"  RZ-NAS-style: reflect on the last trial only, every round,
                                    nothing accumulated across rounds
  show_param_costs=False            Design Agent without the per-candidate parameter-cost table
Controls for "do the principles matter, and does online updating matter" (advisor, 2026-10-13):
  principle_mode="fixed"      a fixed principle set loaded at the start (generic prior knowledge),
                              never updated, no new principles
  principle_mode="random"     randomly generated principles in the same format, never updated
  principle_mode="no_update"  principles are induced as usual but their confidence never moves
Reflection v2 (2026-10-20, after the pilot showed Agent and Agent+Reflector making the same choices;
full style only):
  design_view="reflection_only"  the Design Agent sees the Reflector's summary instead of the raw history
                                 (only a "do not repeat" list of tried signatures)
  propose_hypotheses=True        the Reflector must propose 1-3 untested combinations (up to 2 dimensions
                                 from the best); the Design Agent may take them beyond MUTATION_BUDGET
  confidence_eta / min_evidence_strength   larger step and a strength floor for non-neutral evidence
v1 = full_history, no hypotheses, eta 0.2, no floor (run_search.py --reflection_version v1).
Reflection v3 (2026-10-20, advisor-approved; default): evidence_mode="computed" -- the LLM conjectures,
the experiments refute. Principles carry machine-checkable claims; their confidence is the posterior
probability of the claim under a Bayesian linear model of all scored trials with the measured noise
(evidence.py), recomputed every round (no eta). The Reflector no longer judges evidence; it sees the
evidence table, writes claims and predicts the delta of each untested combination; predictions are stored
with the trial. evidence_mode="llm" is v2 (--reflection_version v2): LLM-judged evidence, eta update.
Hypothesis-guided framework (v4, 2026-10-26, advisor; default; requires v3 evidence):
  1 Design Agent      proposes n_candidates candidates, each with a parent (a tried architecture) and an
                      explicit expected improvement over it
  2 Proxy Evaluator   trains the selected one; the trial records parent and delta_vs_parent
  3 Reflector         verifiable principle library: conditional claims; every parent -> child comparison is
                      logged as support / counterexample per principle; confidence = model posterior
  4 Selection         selection.py: expected gain (model posterior + the LLM's expectation as a prior),
                      principle uncertainty and verification value (selection="llm": the Design Agent's
                      first candidate, ablation)
  5 next round        new evidence revises the library
--reflection_version v3 = the B run (single proposal, Reflector proposes untested combinations).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from ..agents import DesignAgent, DesignContext, ReflectionContext, ReflectorAgent
from ..agents.design_agent import HYPOTHESIS_BUDGET
from ..evidence import EvidenceModel
from ..evidence import fit as fit_evidence
from ..memory import ArchitectureMemory
from ..search_space import (architecture_signature, canonicalize, count_architecture_difference, from_signature,
                            local_mutations)
from ..selection import DEFAULT_WEIGHTS, SELECTION_MODES, choose, score_candidates
from .base import BaseController, Proposal

log = logging.getLogger(__name__)

REFLECTION_STYLES = ("full", "rule", "immediate")
PRINCIPLE_MODES = ("online", "fixed", "random", "no_update")
DESIGN_VIEWS = ("full_history", "reflection_only")
EVIDENCE_MODES = ("computed", "llm")


@dataclass
class AgentSearchConfig:
    use_reflection: bool = True
    use_failure_memory: bool = True
    use_principle_memory: bool = True
    reflection_style: str = "full"      # full | rule (D-6) | immediate (D-7)
    reflect_every_n: int = 1            # v2: the Design Agent sees only the reflection (v1: 2)
    mutation_budget: int = 1
    stagnation_budget: int = 2          # budget used after `stagnation_rounds` rounds without a new best
    stagnation_rounds: int = 3
    explore_every: int = 4              # every k-th round is an exploration round (0 = never)
    max_principles_in_prompt: int = 8
    show_param_costs: bool = True       # per-candidate parameter-cost table in the design prompt
    principle_mode: str = "online"      # online | fixed | random | no_update (see module docstring)
    # reflection v2 (see module docstring); v1 = full_history, False, None (memory.ETA = 0.2), 0.0
    design_view: str = "reflection_only"     # full_history | reflection_only
    propose_hypotheses: bool = False         # v2/v3: the Reflector proposes untested combinations
    confidence_eta: Optional[float] = 0.35
    min_evidence_strength: float = 0.5
    # reflection v3 (see module docstring); "llm" = v2
    evidence_mode: str = "computed"          # computed | llm
    prior_sd: float = 1.0                    # prior sd of a main effect (metric points)
    interaction_prior_sd: float = 0.5        # prior sd of an interaction term named by a 2-term claim
    # hypothesis-guided framework (v4); "none" = v3 single proposal
    selection: str = "guided"                # guided | llm | none
    n_candidates: int = 4
    n_parent_options: int = 5                # best tried architectures offered as parents (+ the baseline)
    llm_prior_sd: float = 0.75               # sd of the Design Agent's expectation used as a prior
    selection_weights: Optional[Dict[str, Dict[str, float]]] = None   # None = selection.DEFAULT_WEIGHTS


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
        if cfg.principle_mode not in PRINCIPLE_MODES:
            raise ValueError(f"principle_mode must be one of {PRINCIPLE_MODES}")
        if cfg.design_view not in DESIGN_VIEWS:
            raise ValueError(f"design_view must be one of {DESIGN_VIEWS}")
        if cfg.evidence_mode not in EVIDENCE_MODES:
            raise ValueError(f"evidence_mode must be one of {EVIDENCE_MODES}")
        if cfg.selection not in SELECTION_MODES:
            raise ValueError(f"selection must be one of {SELECTION_MODES}")
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
            ctx["design_principles"] = [self._principle_entry(p)
                                        for p in memory.active_principles()[: self.cfg.max_principles_in_prompt]]
        else:  # immediate style or principle memory ablated: only the latest reflection's new principles
            ctx["design_principles"] = ref.get("design_principles", [])
        return ctx

    def _principle_entry(self, p) -> Dict[str, Any]:
        e = {"id": p.id, "principle": p.principle, "confidence": round(p.confidence, 3), "status": p.status}
        if self._computed() and p.claim:
            from ..evidence import claim_text
            e["claim"] = claim_text(p.claim)
            if p.stats:
                e["effect_estimate"] = f"{p.stats['effect_mean']:+.2f} +- {p.stats['effect_sd']:.2f}"
                e["direct_trials"] = p.stats.get("n_direct", 0)
        else:
            e.update({"supports": p.support_count, "contradicts": p.contradict_count})
        if p.transferred_from:
            e["transferred_from"] = p.transferred_from
        return e

    def _v2(self) -> bool:
        return self.cfg.use_reflection and self.cfg.reflection_style == "full"

    def _computed(self) -> bool:
        """v3: confidence computed from the trials, evidence table shown to both agents."""
        return self._v2() and self.cfg.evidence_mode == "computed"

    def _guided(self) -> bool:
        """Hypothesis-guided framework: candidate set + selection + pair evidence."""
        return self._computed() and self.cfg.selection in ("guided", "llm")

    def _fit(self, memory: ArchitectureMemory) -> EvidenceModel:
        return fit_evidence(trial_rows(memory), asdict(memory.baseline) if memory.baseline else None, self.task,
                            self.noise_level, claims=[p.claim for p in memory.principles if p.claim],
                            prior_sd=self.cfg.prior_sd, interaction_prior_sd=self.cfg.interaction_prior_sd)

    def _show_history(self) -> bool:
        return not (self._v2() and self.cfg.design_view == "reflection_only")

    def _hypotheses(self, memory: ArchitectureMemory, best_arch, excluded: set,
                    model: Optional[EvidenceModel] = None) -> Optional[List[Dict[str, Any]]]:
        """Untested combinations from the latest reflection that are still untried and within
        HYPOTHESIS_BUDGET of the current best (canonicalized, deduplicated)."""
        if not (self._v2() and self.cfg.propose_hypotheses):
            return None
        out, seen = [], set()
        for h in (memory.latest_reflection() or {}).get("untested_combinations", []):
            try:
                arch = canonicalize(h["architecture"])
            except Exception:  # noqa: BLE001 - malformed hypothesis, skip it
                continue
            sig = architecture_signature(arch)
            diff = count_architecture_difference(arch, best_arch)
            if sig in excluded or sig in seen or not 1 <= diff <= max(self.cfg.mutation_budget, HYPOTHESIS_BUDGET):
                continue
            seen.add(sig)
            item = {"architecture": arch, "rationale": h.get("rationale", ""),
                    "principle_ids": list(h.get("principle_ids", []))}
            if model is not None:  # v3
                if isinstance(h.get("predicted_delta"), (int, float)):
                    item["predicted_delta"] = round(float(h["predicted_delta"]), 3)
                mu, sd = model.predict(arch)
                item["model_prediction"] = f"{mu:+.2f} +- {sd:.2f}"
                item["tests"] = [{"id": p.id, "confidence": round(p.confidence, 3)}
                                 for p in (memory.principle(i) for i in item["principle_ids"]) if p is not None]
            out.append(item)
        return out

    def _uses_principle_memory(self) -> bool:
        """Principles from memory are shown to the Design Agent (all principle modes)."""
        return self.cfg.reflection_style == "full" and self.cfg.use_principle_memory

    def _learns_principles(self) -> bool:
        """The Reflector's principle output is written back to memory (online and no_update only)."""
        return self._uses_principle_memory() and self.cfg.principle_mode in ("online", "no_update")

    def propose(self, memory: ArchitectureMemory, round_: int) -> Proposal:
        mode = self._search_mode(round_)
        failed = memory.failed_signatures() if self.cfg.use_failure_memory else set()
        best = best_info(memory)
        budget = self._mutation_budget(memory, round_)
        excluded = memory.tried_signatures() | failed
        model = self._fit(memory) if self._computed() else None
        if self._guided():
            return self._propose_guided(memory, round_, mode, failed, best, budget, excluded, model)
        hyps = self._hypotheses(memory, best["architecture"], excluded, model)
        cands = None
        if not self._show_history():  # positive list of what may still be proposed
            cands = sorted(architecture_signature(a) for a in local_mutations(best["architecture"], self.task, budget)
                           if architecture_signature(a) not in excluded)
        # the Reflector's own predicted delta is recorded for evaluation but not shown to the Design Agent
        design_hyps = [{k: v for k, v in h.items() if k != "predicted_delta"} for h in hyps] if hyps is not None else None
        ctx = DesignContext(
            task=self.task, metric=self.metric, baseline=baseline_info(memory),
            best=best, history=trial_rows(memory),
            tried=memory.tried_signatures(), failed=failed, search_mode=mode,
            mutation_budget=budget, noise_level=self.noise_level,
            reflection_context=self._reflection_context(memory), round=round_,
            candidate_costs=self._candidate_costs(memory, best["architecture"], budget, excluded),
            show_history=self._show_history(), hypotheses=design_hyps,
            evidence_table=model.table(self.task) if model is not None else None, candidates=cands)
        arch, note, cited_ids = self.design.propose(ctx, record=self.proposal_attempts)
        cited = []
        for pid in cited_ids:  # snapshot the confidence the Design Agent saw (calibration analysis)
            p = memory.principle(pid)
            if p is not None:
                cited.append({"id": p.id, "confidence": round(p.confidence, 4), "status": p.status})
        predictions: Dict[str, Any] = {}
        if model is not None:  # predictions made before training, checked against the result
            mu, sd = model.predict(arch)
            predictions = {"model": round(mu, 3), "model_sd": round(sd, 3)}
            sig = architecture_signature(arch)
            for h in hyps or []:
                if architecture_signature(h["architecture"]) == sig and "predicted_delta" in h:
                    predictions["reflector"] = h["predicted_delta"]
        return Proposal(arch, note=note, search_mode=mode, cited_principles=cited, predictions=predictions)

    def _propose_guided(self, memory: ArchitectureMemory, round_: int, mode: str, failed: set,
                        best: Dict[str, Any], budget: int, excluded: set, model: EvidenceModel) -> Proposal:
        base = memory.baseline
        scored_trials = sorted([t for t in memory.ok_trials()], key=lambda t: -t.score)
        parents = [{"signature": t.signature, "architecture": t.architecture, "score": t.score,
                    "delta": t.score - base.score} for t in scored_trials[: self.cfg.n_parent_options]]
        if base.signature not in {p["signature"] for p in parents}:
            parents.append({"signature": base.signature, "architecture": base.architecture, "score": base.score,
                            "delta": 0.0})
        cands = sorted(architecture_signature(a) for a in local_mutations(best["architecture"], self.task, budget)
                       if architecture_signature(a) not in excluded)
        ctx = DesignContext(
            task=self.task, metric=self.metric, baseline=baseline_info(memory), best=best, history=trial_rows(memory),
            tried=memory.tried_signatures(), failed=failed, search_mode=mode, mutation_budget=budget,
            noise_level=self.noise_level, reflection_context=self._reflection_context(memory), round=round_,
            candidate_costs=self._candidate_costs(memory, best["architecture"], budget, excluded),
            show_history=False, evidence_table=model.table(self.task), candidates=cands,
            parents=parents, n_candidates=self.cfg.n_candidates)
        options = self.design.propose_candidates(ctx, record=self.proposal_attempts)
        library = [{"id": p.id, "confidence": p.confidence, "claim": p.claim}
                   for p in memory.active_principles() if p.claim] if self._uses_principle_memory() else []
        deltas = {p["signature"]: p["delta"] for p in parents}
        deltas.update({t.signature: t.score - base.score for t in memory.ok_trials()})
        scored = score_candidates(options, model, library, deltas, best["score"] - base.score, self.noise_level,
                                  mode, llm_prior_sd=self.cfg.llm_prior_sd,
                                  weights=self.cfg.selection_weights or DEFAULT_WEIGHTS)
        idx = 0 if options[0]["note"].startswith("fallback") else choose(scored, self.cfg.selection)
        pick, s = options[idx], scored[idx]
        cited = []
        for pid in pick["cited_principles"]:  # snapshot the confidence the Design Agent saw
            p = memory.principle(pid)
            if p is not None:
                cited.append({"id": p.id, "confidence": round(p.confidence, 4), "status": p.status})
        predictions = {"model_vs_parent": s["model_mean"], "model_sd": s["model_sd"],
                       "combined_vs_parent": s["combined_mean"]}
        if pick["expected_improvement"] is not None:
            predictions["design_vs_parent"] = round(pick["expected_improvement"], 3)
        selection = {"mode": self.cfg.selection, "search_mode": mode, "chosen": idx, "candidates": scored}
        note = pick["note"] + (f" [selected {idx + 1}/{len(options)}]" if len(options) > 1 else "")
        return Proposal(pick["architecture"], note=note, search_mode=mode, cited_principles=cited,
                        predictions=predictions, parent=pick["parent"], selection=selection)

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
        keep = self._learns_principles()
        computed = self._computed() and self._uses_principle_memory()
        model = None
        if self._guided() and keep and memory.trials:  # parent -> child comparisons of the new trials
            done = self.state.get("pairs_recorded_through", 0)  # controller state survives resume
            for t in memory.trials:
                if t.round > done and t.parent:
                    memory.record_pair_evidence(t, from_signature(t.parent), self.noise_level)
            self.state["pairs_recorded_through"] = max(t.round for t in memory.trials)
        if self._computed():
            if keep and self.cfg.principle_mode == "online":
                memory.recompute_confidence(self._fit(memory), round_)  # the Reflector sees current values
            model = self._fit(memory)
        ctx = ReflectionContext(
            task=self.task, metric=self.metric, baseline=baseline_info(memory),
            best=best_info(memory), history=history[-1:] if immediate else history, new_trials=new_trials,
            failed=[] if immediate or not self.cfg.use_failure_memory else [asdict(f) for f in memory.failures],
            noise_level=self.noise_level,
            previous_principles=[asdict(p) for p in memory.active_principles()] if keep else [],
            propose_hypotheses=self._v2() and self.cfg.propose_hypotheses,
            tried=sorted(memory.tried_signatures()),
            computed_evidence=computed, evidence_table=model.table(self.task) if model is not None else None,
            model=model, library=self._guided())
        ref = self.reflector.reflect(ctx)
        if ref is None:
            log.warning("round %d: reflection failed, keeping previous knowledge", round_)
            return None
        if keep:
            ref["applied"] = self._apply_reflection(memory, ref, round_)
        memory.add_reflection(round_, ref)
        return ref

    def _apply_reflection(self, memory: ArchitectureMemory, ref: Dict[str, Any], round_: int) -> Dict[str, Any]:
        """Evidence updates (one per principle per reflection, unknown ids ignored), then new principles.
        no_update: the updates are recorded in the reflection but not applied, so confidence stays 0.5."""
        updated, ignored = [], []
        if self._computed():  # v3: add the conjectures, then let the data score every claim
            added = [memory.add_principle(f, self.task, round_, claim=f.get("claim"), restate_bonus=False).id
                     for f in ref["new_principles"]]
            if self.cfg.principle_mode == "online":
                updated = memory.recompute_confidence(self._fit(memory), round_)
            return {"updated": updated, "ignored": [], "added": added, "evidence": "computed"}
        if self.cfg.principle_mode == "no_update":
            added = [memory.add_principle(f, self.task, round_, already_updated={p.id for p in memory.principles}).id
                     for f in ref["new_principles"]]
            return {"updated": [], "ignored": [u["principle_id"] for u in ref["principle_updates"]], "added": added}
        for u in ref["principle_updates"]:
            pid = u["principle_id"]
            if pid in updated:
                ignored.append(pid)
                continue
            p = memory.update_principle(pid, u["evidence"], u["strength"], round_,
                                        reason=u.get("reason", ""), trials=u.get("trials", ""),
                                        eta=self.cfg.confidence_eta, floor=self.cfg.min_evidence_strength)
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
