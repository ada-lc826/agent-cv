"""Search loop shared by every controller: baseline -> propose -> train/eval -> record -> (reflect).

All state lives in ArchitectureMemory under run_dir, saved after every round, so an interrupted
search resumes from the last finished round with the same seed.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .agent_client import AgentClientFatal
from .controllers import BaseController
from .memory import ArchitectureMemory, Failure, Trial
from .report import write_reports
from .runners import BaseRunner
from .search_space import BASELINE, architecture_signature, canonicalize, validate

log = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    task: str = "classification"
    mode: str = "agent_reflect"
    variant: str = ""                    # ablation / setting tag, part of the run identity (see run_search.py)
    seed: int = 0
    max_rounds: int = 10                 # trial budget, baseline excluded (research plan: 10; 30 for the sensitivity run)
    min_improvement: float = 0.2         # a new best must beat the old one by this much (>= proxy noise)
    noise_level: float = 0.2             # seed std of the proxy score; deltas below it are "no effect"
    early_stop_patience: int = 0         # 0 = off (must be off in method comparisons)
    target_score: Optional[float] = None
    max_consecutive_failures: int = 5
    run_dir: str = "runs/debug"


def _fingerprint(cfg: SearchConfig) -> Dict[str, Any]:
    return {k: getattr(cfg, k) for k in ("task", "mode", "seed", "variant")}


class SearchEngine:
    def __init__(self, cfg: SearchConfig, controller: BaseController, runner: BaseRunner,
                 memory: Optional[ArchitectureMemory] = None):
        self.cfg = cfg
        self.controller = controller
        self.runner = runner
        os.makedirs(cfg.run_dir, exist_ok=True)
        self.memory = memory or ArchitectureMemory.load(cfg.run_dir)
        self._check_resume()
        self._load_controller_state()

    # ------------------------------------------------------------------
    def _state_path(self) -> str:
        return os.path.join(self.cfg.run_dir, "controller_state.json")

    def _load_controller_state(self) -> None:
        path = self._state_path()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.controller.state = json.load(f)

    def _save_controller_state(self) -> None:
        tmp = self._state_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.controller.state, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._state_path())

    def _check_resume(self) -> None:
        path = os.path.join(self.cfg.run_dir, "run_config.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
            for k, v in _fingerprint(self.cfg).items():
                if old.get(k, "" if k == "variant" else None) != v:  # runs saved before `variant` existed
                    raise ValueError(f"run_dir {self.cfg.run_dir} belongs to a run with {k}={old.get(k)!r}, "
                                     f"not {v!r}; use a new --run_dir")
            if self.memory.trials:
                log.info("resuming from round %d", self.memory.last_round())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, indent=2)

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        mem, cfg = self.memory, self.cfg
        if mem.baseline is None:
            self._run_baseline()
        if mem.baseline.status != "ok":
            stop = "baseline_failed"
            write_reports(mem, cfg, stop)
            return {"stop_reason": stop}

        stop = "max_rounds"
        for r in range(mem.last_round() + 1, cfg.max_rounds + 1):
            self.controller.proposal_attempts = []
            try:
                prop = self.controller.propose(mem, r)
            except AgentClientFatal:
                raise
            except RuntimeError as e:  # search space exhausted
                log.warning("round %d: %s", r, e)
                stop = "space_exhausted"
                break
            arch = canonicalize(prop.architecture)
            bad = validate(arch, cfg.task)
            if bad or architecture_signature(arch) in mem.tried_signatures():
                raise RuntimeError(f"controller {self.controller.name} returned an illegal candidate: "
                                   f"{bad or 'already tried'}")  # a controller bug, not a trial failure
            trial = self._evaluate(r, arch, prop.note, prop.search_mode, prop.cited_principles,
                                   attempts=self.controller.proposal_attempts, predictions=prop.predictions)
            log.info("round %d [%s] %s -> %s%s", r, prop.search_mode or "-", trial.signature,
                     trial.score if trial.status == "ok" else f"FAILED({trial.failure_reason})",
                     "  * new best" if trial.is_best else "")
            try:
                ref = self.controller.after_trial(mem, r)
                if ref:
                    log.info("round %d reflection: %s", r, "; ".join(ref.get("design_principles", [])))
            except AgentClientFatal:
                raise
            except Exception:  # noqa: BLE001 -- reflection must never kill the search
                log.exception("round %d: after_trial hook failed", r)
            mem.save()
            self._save_controller_state()
            write_reports(mem, cfg, "running")

            reason = self._should_stop(r)
            if reason:
                stop = reason
                break
        summary = write_reports(mem, cfg, stop)
        log.info("search finished (%s): best %s", stop, summary["best"])
        return summary

    def _run_baseline(self) -> None:
        mem = self.memory
        arch = canonicalize(BASELINE)
        sig = architecture_signature(arch)
        res = self.runner.run(arch, f"r000_{sig}")
        mem.baseline = Trial(0, arch, sig, res.status, res.score, res.params, 0.0 if res.score is not None else None,
                             None, 1.0 if res.params else None, res.failure_reason, "", "baseline", False,
                             res.duration_s, self.cfg.task)
        mem.save()
        log.info("baseline %s -> %s (%s M params)", sig, res.score, res.params)

    def _current_best_score(self) -> float:
        b = self.memory.best_trial()
        return b.score if b else self.memory.baseline.score

    def _evaluate(self, r: int, arch, note: str, mode: str, cited: Optional[list] = None,
                  attempts: Optional[list] = None, predictions: Optional[dict] = None) -> Trial:
        mem, cfg = self.memory, self.cfg
        sig = architecture_signature(arch)
        best_before = self._current_best_score()
        res = self.runner.run(arch, f"r{r:03d}_{sig}")
        t = Trial(round=r, architecture=arch, signature=sig, status=res.status, score=res.score,
                  params=res.params, failure_reason=res.failure_reason, search_mode=mode,
                  proposer_note=note, duration_s=res.duration_s, task=cfg.task,
                  cited_principles=list(cited or []), proposal_attempts=list(attempts or []),
                  predictions=dict(predictions or {}))
        base = mem.baseline
        if res.status == "ok":
            t.delta_vs_baseline = round(res.score - base.score, 3)
            t.delta_vs_previous = round(res.score - best_before, 3)  # vs. best-so-far before this round
            if res.params and base.params:
                t.param_ratio = round(res.params / base.params, 4)
            t.is_best = res.score >= best_before + cfg.min_improvement
            if t.delta_vs_baseline < -cfg.noise_level:
                mem.add_failure(Failure(sig, "score_drop", t.delta_vs_baseline, r))
        else:
            mem.add_failure(Failure(sig, res.failure_reason or "crash", None, r))
        mem.add_trial(t)
        mem.save()
        return t

    def _should_stop(self, r: int) -> Optional[str]:
        mem, cfg = self.memory, self.cfg
        if cfg.target_score is not None and self._current_best_score() >= cfg.target_score:
            return "target_reached"
        if cfg.early_stop_patience > 0:
            b = mem.best_trial()
            if r - (b.round if b else 0) >= cfg.early_stop_patience:
                return "no_improvement"
        recent = mem.trials[-cfg.max_consecutive_failures:]
        if len(recent) == cfg.max_consecutive_failures and all(t.status != "ok" for t in recent):
            return "too_many_failures"
        return None
