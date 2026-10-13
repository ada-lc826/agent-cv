"""LAPT (Zhou et al., AAAI 2025): LLM-learned design principles shrink the search space; REA searches inside.

Original (NAS-Bench-201 setting): principles learned from 50 top architectures, at most 2 operators per
layer, REA (population 5, tournament 2), principle adaptation after each search iteration (r=5 top
architectures): if a better architecture was found, update the principles from it; otherwise ask the LLM
for different options (Algorithm 2).

Here (10-trial budget): the first `n_init` trials are random (no prior task to transfer from; with
--init_principles of another run, LAPT's transfer setting, see run_search.py), then principles are
learned from everything evaluated so far, at most k=2 options per dimension, REA with P=2, S=2 inside
the subspace, adaptation every `adapt_every` trials with the top r=2 architectures. If the subspace is
exhausted, the "no improvement" adaptation is triggered immediately. All of this lives in controller state.
"""
from __future__ import annotations

import itertools
import json
import logging
from typing import Any, Dict, List, Optional

from ..agent_client import AgentClient, AgentClientError, AgentClientFatal
from ..baseline_prompts import LAPT_LEARN, LAPT_SYSTEM, LAPT_UPDATE_IMPROVED, LAPT_UPDATE_STUCK, lapt_schema
from ..memory import ArchitectureMemory, Trial
from ..search_space import (DIMENSIONS, Architecture, architecture_signature, canonicalize,
                            count_architecture_difference, get_search_space)
from .base import BaseController, Proposal
from .llm_common import fmt_arch, fmt_result, fmt_space, ok_trials_with_baseline

log = logging.getLogger(__name__)


class LAPTController(BaseController):
    name = "lapt"

    def __init__(self, task: str, seed: int, client: AgentClient, metric: str, n_init: int = 3, k: int = 2,
                 adapt_every: int = 2, top_r: int = 2, population_size: int = 2, tournament_size: int = 2,
                 init_principles: Optional[List[str]] = None):
        super().__init__(task, seed)
        self.client, self.metric = client, metric
        self.n_init, self.k, self.adapt_every, self.top_r = n_init, k, adapt_every, top_r
        self.P, self.S = population_size, tournament_size
        self._init_principles = init_principles or []

    # ------------------------------------------------------------------ LLM calls
    def _call(self, purpose: str, prompt: str, memory: ArchitectureMemory) -> bool:
        space = get_search_space(self.task)
        try:
            out = self.client.complete_json(purpose, LAPT_SYSTEM, prompt, lapt_schema(space, DIMENSIONS),
                                            mock_fn=lambda: self._mock(memory))
        except AgentClientFatal:
            raise
        except AgentClientError as e:
            log.warning("lapt %s failed: %s", purpose, e)
            return False
        allowed = {}
        for d in DIMENSIONS:
            vals = [v for v in dict.fromkeys(out["allowed"].get(d, [])) if v in space[d]][: self.k]
            allowed[d] = vals or list(space[d])          # an empty list would make the space empty
        self.state.update(principles=list(out["principles"]), allowed=allowed)
        return True

    def _table(self, trials: List[Trial]) -> str:
        return "\n".join(f"- {fmt_arch(t.architecture)}: {fmt_result(t, self.metric)}" for t in trials)

    def _learn(self, memory: ArchitectureMemory) -> None:
        trials = sorted(ok_trials_with_baseline(memory), key=lambda t: t.score, reverse=True)
        prompt = LAPT_LEARN.format(space=fmt_space(self.task), metric=self.metric, table=self._table(trials), k=self.k)
        if self._init_principles:
            prompt += "\n\nPrinciples transferred from an earlier task (revise them with the evidence above):\n" \
                      + "\n".join(f"- {p}" for p in self._init_principles)
        if not self._call("lapt_learn", prompt, memory):
            self.state.update(principles=[], allowed={d: list(v) for d, v in get_search_space(self.task).items()})
        self.state["base"] = self._best_score(memory)
        self.state["last_adapt"] = memory.last_round()

    def _adapt(self, memory: ArchitectureMemory, force_stuck: bool = False) -> None:
        best = self._best_score(memory)
        principles = "\n".join(f"- {p}" for p in self.state.get("principles", [])) or "(none)"
        allowed = json.dumps(self.state["allowed"])
        if best > self.state.get("base", float("-inf")) and not force_stuck:
            inside = [t for t in memory.ok_trials() if self._in_subspace(t.architecture)]
            top = sorted(inside, key=lambda t: t.score, reverse=True)[: self.top_r]
            prompt = LAPT_UPDATE_IMPROVED.format(principles=principles, allowed=allowed, metric=self.metric,
                                                 table=self._table(top), k=self.k)
            self.state["base"] = best
            self._call("lapt_update", prompt, memory)
        else:
            self._call("lapt_expand", LAPT_UPDATE_STUCK.format(principles=principles, allowed=allowed, k=self.k),
                       memory)
        self.state["last_adapt"] = memory.last_round()

    # ------------------------------------------------------------------ search inside the subspace
    def _best_score(self, memory: ArchitectureMemory) -> float:
        return (memory.best_trial() or memory.baseline).score

    def _in_subspace(self, arch: Architecture) -> bool:
        return architecture_signature(arch) in {architecture_signature(c) for c in self._subspace()}

    def _subspace(self) -> List[Architecture]:
        a = self.state["allowed"]
        seen, out = set(), []
        for combo in itertools.product(*(a[d] for d in DIMENSIONS)):
            arch = canonicalize(dict(zip(DIMENSIONS, combo)))
            sig = architecture_signature(arch)
            if sig not in seen:
                seen.add(sig)
                out.append(arch)
        return out

    def _rea_step(self, memory: ArchitectureMemory, round_: int) -> Optional[Proposal]:
        tried = memory.tried_signatures()
        untried = [a for a in self._subspace() if architecture_signature(a) not in tried]
        if not untried:
            return None
        inside = [t for t in sorted(memory.trials, key=lambda t: t.round) if self._in_subspace(t.architecture)]
        population = [t for t in inside[-self.P:] if t.status == "ok"]
        if len(inside) < self.P or not population:
            return Proposal(self.rng(round_, "sub").choice(untried), note="lapt: random in learned subspace")
        rng = self.rng(round_, "tournament")
        parent = max((rng.choice(population) for _ in range(self.S)), key=lambda t: t.score)
        children = [a for a in untried if count_architecture_difference(a, parent.architecture) == 1]
        if children:
            return Proposal(rng.choice(children), note=f"lapt: REA mutation of r{parent.round:03d} in subspace")
        return Proposal(rng.choice(untried), note="lapt: random in learned subspace")

    def propose(self, memory: ArchitectureMemory, round_: int) -> Proposal:
        if round_ <= self.n_init and not self._init_principles:
            return Proposal(self.random_untried(memory, round_, "init"), note="lapt: random initial sample")
        if "allowed" not in self.state:
            self._learn(memory)
        elif memory.last_round() - self.state.get("last_adapt", 0) >= self.adapt_every:
            self._adapt(memory)
        prop = self._rea_step(memory, round_)
        if prop is None:                                   # subspace exhausted: ask for different options
            self._adapt(memory, force_stuck=True)
            prop = self._rea_step(memory, round_)
        if prop is None:
            return Proposal(self.random_untried(memory, round_), note="lapt-fallback: subspace exhausted, random")
        prop.note += f" | allowed={json.dumps(self.state['allowed'])}"
        return prop

    # ------------------------------------------------------------------
    def _mock(self, memory: ArchitectureMemory) -> Dict[str, Any]:
        """Dry runs only: keep the k values with the best mean score per dimension."""
        space = get_search_space(self.task)
        trials = ok_trials_with_baseline(memory)
        allowed = {}
        for d in DIMENSIONS:
            means = {}
            for v in space[d]:
                s = [t.score for t in trials if t.architecture[d] == v]
                means[v] = sum(s) / len(s) if s else float("-inf")
            allowed[d] = sorted(space[d], key=lambda v: means[v], reverse=True)[: self.k]
        return {"principles": ["mock: keep the options with the best mean score"], "allowed": allowed}
