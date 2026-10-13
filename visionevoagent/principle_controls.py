"""Principle sets for the control experiments (advisor feedback, 2026-10-13).

random_principles(): the "random principles" control and the "random initialisation" arm of the
transfer experiment. Each principle names a real search-space choice and a predicted direction and
size, all drawn at random, and is written in exactly the same falsifiable format as induced
principles, so the Design Agent cannot tell them apart by form. Nothing in the text says it is random.

The "fixed principles" control loads a hand-written file instead (configs/principles_fixed_cls.json).
"""
from __future__ import annotations

import random
from typing import Dict, List

from .memory import INITIAL_CONFIDENCE, Principle, render_principle
from .search_space import BASELINE, DIMENSIONS, MODULE_NOTES, get_search_space

_SCOPE = "{task} backbone at a fixed training recipe, all other dimensions held at their current values"
_BECAUSE = "it changes {what}, which the proxy training is sensitive to"
_DISPROVED = "a controlled change of {dim} to {val} moves the score in the opposite direction by more than the noise level"


def random_principles(task: str, n: int, seed: int) -> List[Principle]:
    rng = random.Random(f"random-principles-{task}-{seed}")
    space = get_search_space(task)
    pairs = [(d, v) for d in DIMENSIONS for v in space[d] if v != BASELINE[d]]
    rng.shuffle(pairs)
    out = []
    for i, (dim, val) in enumerate(pairs[:n], start=1):
        up = rng.random() < 0.5
        lo = round(rng.uniform(0.1, 0.6), 1)
        hi = round(lo + rng.uniform(0.2, 0.6), 1)
        what = MODULE_NOTES.get(val, f"the {dim} choice")
        fields: Dict[str, str] = {
            "if": f"{dim} is set to {val}",
            "scope": _SCOPE.format(task=task),
            "then": f"the score {'rises' if up else 'falls'} by about {lo} to {hi} points",
            "because": _BECAUSE.format(what=what),
            "disproved_if": _DISPROVED.format(dim=dim, val=val),
        }
        out.append(Principle(render_principle(fields), INITIAL_CONFIDENCE, 0, task, 0, 0, id=f"p{i}",
                             fields=fields))
    return out
