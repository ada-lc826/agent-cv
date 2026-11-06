"""One place that maps a task to its training runner (2026-11-06). Imports are lazy so that dry runs and
score-table searches never need the training frameworks."""
from __future__ import annotations

from typing import Any, Dict

from .base import BaseRunner


def make_runner(task: str, rc: Dict[str, Any], **overrides: Any) -> BaseRunner:
    """rc: the yaml `runner` section; overrides: seed, work_root, schedule, ... for this run."""
    cfg = {**rc, **{k: v for k, v in overrides.items() if v is not None}}
    if task == "segmentation":
        from .mmseg_runner import MMSegRunner, MMSegRunnerConfig
        return MMSegRunner(MMSegRunnerConfig(**cfg))
    if task == "classification":
        from .mmpretrain_runner import MMPretrainRunner, MMPretrainRunnerConfig
        return MMPretrainRunner(MMPretrainRunnerConfig(**cfg))
    raise ValueError(f"no training runner for task {task!r} (NAS-Bench-201 tasks read a score table)")
