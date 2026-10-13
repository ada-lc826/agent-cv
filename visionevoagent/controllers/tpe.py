"""TPE (Bergstra et al., NeurIPS 2011) through Optuna's TPESampler.

The 4 dimensions are categorical variables. n_startup_trials is set to 3: Optuna's default (10)
equals the whole budget, so TPE would never leave random sampling. gamma and the number of EI
candidates stay at Optuna's defaults.

Resume: the study is rebuilt every round from ArchitectureMemory (baseline included as an
observation), with a sampler seed derived from (seed, round), so no extra state is needed.
"""
from __future__ import annotations

import logging

from ..memory import ArchitectureMemory
from ..search_space import DIMENSIONS, architecture_signature, canonicalize, get_search_space
from .base import BaseController, Proposal

log = logging.getLogger(__name__)


class TPEController(BaseController):
    name = "tpe"

    def __init__(self, task: str, seed: int, n_startup_trials: int = 3, max_asks: int = 50):
        super().__init__(task, seed)
        try:
            import optuna  # noqa: F401
        except ImportError as e:
            raise ImportError("--mode tpe needs optuna (pip install optuna)") from e
        self.n_startup_trials = n_startup_trials
        self.max_asks = max_asks

    def propose(self, memory: ArchitectureMemory, round_: int) -> Proposal:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        space = get_search_space(self.task)
        dists = {d: optuna.distributions.CategoricalDistribution(space[d]) for d in DIMENSIONS}
        sampler = optuna.samplers.TPESampler(n_startup_trials=self.n_startup_trials,
                                             seed=self.seed * 100003 + round_)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        observed = ([memory.baseline] if memory.baseline else []) + list(memory.trials)
        for t in observed:
            state = optuna.trial.TrialState.COMPLETE if t.status == "ok" else optuna.trial.TrialState.FAIL
            study.add_trial(optuna.trial.create_trial(
                params=dict(t.architecture), distributions=dists, state=state,
                value=t.score if t.status == "ok" else None))

        tried = memory.tried_signatures()
        scores = {t.signature: t.score for t in observed}
        for _ in range(self.max_asks):
            trial = study.ask(dists)
            arch = canonicalize({d: trial.params[d] for d in DIMENSIONS})
            sig = architecture_signature(arch)
            if sig not in tried:
                return Proposal(arch, note="tpe: sampler suggestion")
            # Equivalent to an already-evaluated architecture: report its known result and ask again.
            if scores.get(sig) is not None:
                study.tell(trial, scores[sig])
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
        return Proposal(self.random_untried(memory, round_), note="tpe: sampler kept repeating, random")
