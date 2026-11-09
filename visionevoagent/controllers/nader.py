"""NADER (Yang et al., CVPR 2025) without the Reader, re-implemented for a config search space.

Per trial (2 LLM calls):
  Proposer  sees the whole model-optimisation tree (every trial with its parent and the inspiration that
            produced it) and picks a (parent model, inspiration) pair. The official code hands the tree
            to the LLM; there is no explicit DFS/BFS (the paper's wording).
  Modifier  applies the inspiration to the parent, keeping unrelated choices, with the top-5 experiences
            retrieved for that inspiration.
Inspirations: without the Reader, the LLM proposes 10 of them once (the paper's "w/o Reader" ablation).
Research reflector: every `extract_every` trials, each new child that scored worse than its parent by
more than `drop_threshold` gets one LLM suggestion, stored with its inspiration text as the key; only
failures are stored and nothing is ever updated, as in the original (which builds the store offline
from a previous run). Threshold: the original uses "below 90% of the parent"; module insertion moves
top-1 by about a point, so that rule would never fire. The default here is the proxy noise level, the
same yardstick our own method uses (set `drop_threshold=None` for the original 0.9 rule).
Retrieval: the original embeds inspirations with an OpenAI embedding model; here a bag-of-words cosine,
to keep the comparison free of a second LLM provider.
The development reflector (repairing invalid graphs) has no counterpart: config candidates are always valid.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from ..agent_client import AgentClient, AgentClientError, AgentClientFatal
from ..baseline_prompts import (NADER_EXPERIENCE, NADER_INSPIRATIONS, NADER_MODIFY, NADER_PROPOSE,
                                NADER_PROPOSE_SCHEMA, NADER_REFLECT, NADER_SUGGESTION_SCHEMA, NADER_SYSTEM,
                                nader_inspirations_schema)
from ..memory import ArchitectureMemory, Trial
from ..search_space import MODULE_NOTES
from .base import BaseController, Proposal
from .llm_common import ask_architecture, fmt_arch, fmt_result, fmt_space, mock_neighbour

log = logging.getLogger(__name__)


def _bow(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def _cosine(a: Counter, b: Counter) -> float:
    num = sum(a[w] * b[w] for w in a)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


class NaderController(BaseController):
    name = "nader"

    def __init__(self, task: str, seed: int, client: AgentClient, metric: str, noise_level: float,
                 n_inspirations: int = 10, extract_every: int = 2, top_k: int = 5,
                 drop_threshold: Optional[float] = -1.0):
        super().__init__(task, seed)
        self.client, self.metric = client, metric
        self.n_inspirations, self.extract_every, self.top_k = n_inspirations, extract_every, top_k
        # -1.0 -> use the noise level; None -> original rule (score < 0.9 x parent score)
        self.drop_threshold = noise_level if drop_threshold == -1.0 else drop_threshold

    # ------------------------------------------------------------------ inspirations
    def _inspirations(self, memory: ArchitectureMemory) -> List[str]:
        if "inspirations" not in self.state:
            b = memory.baseline
            prompt = NADER_INSPIRATIONS.format(baseline=fmt_arch(b.architecture), space=fmt_space(self.task),
                                               n=self.n_inspirations, metric=self.metric)
            insp: List[str] = []
            try:
                out = self.client.complete_json("nader_inspirations", NADER_SYSTEM, prompt,
                                                nader_inspirations_schema(), mock_fn=self._mock_inspirations)
                insp = [s for s in out["inspirations"] if isinstance(s, str) and s.strip()][: self.n_inspirations]
            except AgentClientFatal:
                raise
            except AgentClientError as e:
                log.warning("nader inspirations failed: %s", e)
            self.state["inspirations"] = insp or self._mock_inspirations()["inspirations"]
        return self.state["inspirations"]

    @staticmethod
    def _mock_inspirations() -> Dict[str, Any]:
        return {"inspirations": [f"Use {k}: {v}" for k, v in MODULE_NOTES.items()][:10]}

    # ------------------------------------------------------------------ tree + experiences
    def _tree(self, memory: ArchitectureMemory) -> str:
        edges = self.state.get("edges", {})
        lines = [f"r000 (baseline) {fmt_arch(memory.baseline.architecture)}: {fmt_result(memory.baseline, self.metric)}"]
        for t in sorted(memory.trials, key=lambda t: t.round):
            e = edges.get(str(t.round))
            link = f"r{e['parent']:03d} --[{e['inspiration']}]--> " if e else "(no parent) "
            lines.append(f"{link}r{t.round:03d} {fmt_arch(t.architecture)}: {fmt_result(t, self.metric)}")
        return "\n".join(lines)

    def _retrieve(self, inspiration: str) -> List[str]:
        q = _bow(inspiration)
        exps = self.state.get("experiences", [])
        ranked = sorted(exps, key=lambda e: _cosine(q, _bow(e["key"])), reverse=True)
        return [e["suggestion"] for e in ranked[: self.top_k]]

    # ------------------------------------------------------------------
    def propose(self, memory: ArchitectureMemory, round_: int) -> Proposal:
        insp = self._inspirations(memory)
        nodes: Dict[int, Trial] = {0: memory.baseline, **{t.round: t for t in memory.ok_trials()}}
        rng = self.rng(round_, "mock")
        try:
            pick = self.client.complete_json(
                "nader_propose", NADER_SYSTEM,
                NADER_PROPOSE.format(metric=self.metric, tree=self._tree(memory),
                                     inspirations="\n".join(f"{i}: {s}" for i, s in enumerate(insp))),
                NADER_PROPOSE_SCHEMA,
                mock_fn=lambda: {"parent_round": max(nodes, key=lambda r: nodes[r].score),
                                 "inspiration_index": rng.randrange(len(insp)), "rationale": "mock"})
            parent_round, i = int(pick["parent_round"]), int(pick["inspiration_index"])
        except AgentClientFatal:
            raise
        except (AgentClientError, KeyError, ValueError, TypeError) as e:
            log.warning("nader proposer failed: %s", e)
            parent_round, i = max(nodes, key=lambda r: nodes[r].score), rng.randrange(len(insp))
        parent = nodes.get(parent_round) or nodes[max(nodes, key=lambda r: nodes[r].score)]
        i = i if 0 <= i < len(insp) else 0

        exps = self._retrieve(insp[i])
        experience = NADER_EXPERIENCE.format(items="\n".join(f"- {e}" for e in exps)) if exps else ""
        tried = memory.tried_signatures()
        arch, why, _ = ask_architecture(
            self.client, "nader_modify", NADER_SYSTEM,
            NADER_MODIFY.format(space=fmt_space(self.task), inspiration=insp[i],
                                arch=fmt_arch(parent.architecture), experience=experience),
            self.task, tried, record=self.proposal_attempts,
            mock_fn=lambda att: mock_neighbour(parent.architecture, self.task, tried, self.rng(round_, f"m{att}")))
        self.state.setdefault("edges", {})[str(round_)] = {"parent": parent.round, "inspiration": i}
        if arch is None:
            return Proposal(self.random_untried(memory, round_), note="nader-fallback: random untried")
        return Proposal(arch, note=f"nader: r{parent.round:03d} + inspiration {i} ({len(exps)} exp): {why}")

    # ------------------------------------------------------------------ research reflector
    def after_trial(self, memory: ArchitectureMemory, round_: int) -> Optional[dict]:
        if round_ % self.extract_every != 0:
            return None
        done = set(self.state.setdefault("reflected", []))
        edges = self.state.get("edges", {})
        by_round = {0: memory.baseline, **{t.round: t for t in memory.trials}}
        insp = self.state.get("inspirations", [])
        added = 0
        for t in memory.ok_trials():
            e = edges.get(str(t.round))
            if t.round in done or not e:
                continue
            done.add(t.round)
            parent = by_round.get(e["parent"])
            if parent is None or parent.score is None:
                continue
            dropped = (t.score < 0.9 * parent.score) if self.drop_threshold is None \
                else (t.score < parent.score - self.drop_threshold)
            if not dropped:
                continue
            text = insp[e["inspiration"]] if e["inspiration"] < len(insp) else ""
            prompt = NADER_REFLECT.format(metric=self.metric, raw=fmt_arch(parent.architecture), raw_score=parent.score,
                                          new=fmt_arch(t.architecture), new_score=t.score, inspiration=text)
            try:
                out = self.client.complete_json(
                    "nader_reflect", NADER_SYSTEM, prompt, NADER_SUGGESTION_SCHEMA,
                    mock_fn=lambda: {"analysis": "mock", "suggestion": f"Avoid {t.signature} style changes."})
            except AgentClientFatal:
                raise
            except AgentClientError as err:
                log.warning("nader reflection failed: %s", err)
                continue
            self.state.setdefault("experiences", []).append(
                {"key": text, "suggestion": out["suggestion"], "round": t.round})
            added += 1
        self.state["reflected"] = sorted(done)
        return {"design_principles": [f"nader: stored {added} new experience(s)"]} if added else None
