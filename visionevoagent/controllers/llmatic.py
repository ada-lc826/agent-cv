"""LLMatic (Nasir et al., GECCO 2024), scaled to the 10-trial budget.

Two quality-diversity archives (MAP-Elites style):
  - network archive: cell = (parameter-change bin, target_stage), elite = best score in the cell.
    Original: CVT grid of 100 niches over (width/depth ratio, FLOPs). 3x3 here, because 10 trials
    cannot fill 100 niches and our architectures differ in where/what is inserted, not in width/depth.
  - prompt archive: cell = (instruction id, sampling level), value = curiosity.
    Original second dimension: sampling temperature. When the client applies temperature (DeepSeek with
    thinking disabled) the levels are temperatures 0.3 / 0.6 / 0.9 (paper: initial 0.6, range 0-1);
    otherwise (DeepSeek thinking mode, or Claude, which rejects temperature) the effort level
    low / medium / high is the closest available knob, reported as a deviation.
Operators follow the paper (mutation 0.7 / crossover 0.3; the official code uses 0.85 / 0.15).
Curiosity: +1 if the child entered the network archive, -0.5 if it did not, -1 if it failed to train
(the paper's rule; the official code has no -1 case). Mutation picks a random archive network and,
with probability 0.5, the most curious prompt individual, otherwise a random one (exploration).
The first `random_init` trials mutate the baseline with random instructions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..agent_client import AgentClient
from ..baseline_prompts import LLMATIC_CROSSOVER, LLMATIC_INSTRUCTIONS, LLMATIC_MUTATE, LLMATIC_SYSTEM
from ..memory import ArchitectureMemory, Trial
from .base import BaseController, Proposal
from .llm_common import ask_architecture, fmt_arch, fmt_space, mock_neighbour

EFFORTS = ("low", "medium", "high")
TEMPERATURE_LEVELS = (0.3, 0.6, 0.9)


class LLMaticController(BaseController):
    name = "llmatic"

    def __init__(self, task: str, seed: int, client: AgentClient, p_mutation: float = 0.7,
                 random_init: int = 2, param_bins: Tuple[float, float] = (0.5, 2.0)):
        super().__init__(task, seed)
        self.client = client
        self.p_mutation = p_mutation
        self.random_init = random_init
        self.param_bins = param_bins          # |Δparams| in millions: small < 0.5 <= medium < 2.0 <= large
        self.knob = "temperature" if client.temperature_effective else "effort"
        self.levels = TEMPERATURE_LEVELS if self.knob == "temperature" else EFFORTS

    # ------------------------------------------------------------------ archives (in controller state)
    def _net_cell(self, t: Trial, base_params: float) -> str:
        dp = abs((t.params or base_params) - base_params)
        size = "small" if dp < self.param_bins[0] else ("medium" if dp < self.param_bins[1] else "large")
        return f"{size}|{t.architecture['target_stage']}"

    def _net_archive(self) -> Dict[str, Dict[str, Any]]:
        return self.state.setdefault("network_archive", {})

    def _prompt_archive(self) -> Dict[str, float]:
        return self.state.setdefault("prompt_curiosity", {})

    def _ensure_baseline(self, memory: ArchitectureMemory) -> None:
        arc = self._net_archive()
        b = memory.baseline
        cell = self._net_cell(b, b.params)
        if cell not in arc:
            arc[cell] = {"round": 0, "score": b.score}

    # ------------------------------------------------------------------
    def propose(self, memory: ArchitectureMemory, round_: int) -> Proposal:
        self._ensure_baseline(memory)
        rng = self.rng(round_, "op")
        tried = memory.tried_signatures()
        by_round = {0: memory.baseline, **{t.round: t for t in memory.trials}}
        arc = self._net_archive()
        elites = [by_round[v["round"]] for v in arc.values() if v["round"] in by_round]

        if round_ <= self.random_init:
            op, parents = "mutation", [memory.baseline]
            instr, level = rng.randrange(len(LLMATIC_INSTRUCTIONS)), rng.choice(self.levels)
        elif len(elites) >= 2 and rng.random() >= self.p_mutation:
            op, instr, level = "crossover", None, self.levels[1]
            parents = rng.sample(elites, 2)
        else:
            op = "mutation"
            parents = [rng.choice(elites)]
            instr, level = self._pick_prompt(rng)

        space = fmt_space(self.task)
        if op == "mutation":
            prompt = LLMATIC_MUTATE.format(space=space, arch=fmt_arch(parents[0].architecture),
                                           instruction=LLMATIC_INSTRUCTIONS[instr])
        else:
            prompt = LLMATIC_CROSSOVER.format(space=space, a=fmt_arch(parents[0].architecture),
                                              b=fmt_arch(parents[1].architecture))
        arch, why, _ = ask_architecture(
            self.client, f"llmatic_{op}", LLMATIC_SYSTEM, prompt, self.task, tried, **self._call_kwargs(level),
            record=self.proposal_attempts,
            mock_fn=lambda att: mock_neighbour(parents[0].architecture, self.task, tried,
                                               self.rng(round_, f"mock{att}")))
        self.state.setdefault("pending", {})[str(round_)] = {"op": op, "instruction": instr, "level": level,
                                                            "llm": arch is not None}
        names = "+".join(f"r{p.round:03d}" for p in parents)
        if arch is None:
            return Proposal(self.random_untried(memory, round_), note=f"llmatic-fallback ({op} of {names}): random")
        label = f"instr {instr}, {self.knob} {level}" if op == "mutation" else "crossover"
        return Proposal(arch, note=f"llmatic {op} of {names} ({label}): {why}")

    def _call_kwargs(self, level: Any) -> Dict[str, Any]:
        return {"temperature": float(level)} if self.knob == "temperature" else {"effort": level}

    def _pick_prompt(self, rng) -> Tuple[int, Any]:
        cur = self._prompt_archive()
        if cur and rng.random() < 0.5:
            key = max(sorted(cur), key=lambda k: cur[k])
            i, lv = key.split("|")
            return int(i), (float(lv) if self.knob == "temperature" else lv)
        return rng.randrange(len(LLMATIC_INSTRUCTIONS)), rng.choice(self.levels)

    # ------------------------------------------------------------------
    def after_trial(self, memory: ArchitectureMemory, round_: int) -> Optional[dict]:
        pend = self.state.get("pending", {}).pop(str(round_), None)
        t = memory.trials[-1]
        added = False
        if t.status == "ok":
            arc = self._net_archive()
            cell = self._net_cell(t, memory.baseline.params)
            if cell not in arc or t.score > arc[cell]["score"]:
                arc[cell] = {"round": t.round, "score": t.score}
                added = True
        if pend and pend["op"] == "mutation" and pend["llm"]:  # crossover and fallbacks don't score a prompt
            key = f"{pend['instruction']}|{pend['level']}"
            cur = self._prompt_archive()
            delta = -1.0 if t.status != "ok" else (1.0 if added else -0.5)
            cur[key] = cur.get(key, 0.0) + delta
        return None
