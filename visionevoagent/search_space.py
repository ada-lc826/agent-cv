"""Architecture search space, signatures and mutation helpers.

An architecture is a dict over the dimensions of SEARCH_SPACE, e.g.
{"attention": "eca", "conv": "dilated", "fusion": "none", "target_stage": "late"}.
"""
from __future__ import annotations

import itertools
import random
from typing import Dict, Iterable, List, Optional

Architecture = Dict[str, str]

# Same dimensions for every task; the runner decides where each choice is inserted.
SEARCH_SPACES: Dict[str, Dict[str, List[str]]] = {
    "classification": {
        "attention": ["none", "se", "eca", "cbam", "coordatt"],
        "conv": ["standard", "dilated", "depthwise"],
        "fusion": ["none", "weighted"],
        "target_stage": ["early", "late", "all"],
    },
}

DIMENSIONS = ["attention", "conv", "fusion", "target_stage"]

BASELINE: Architecture = {"attention": "none", "conv": "standard", "fusion": "none", "target_stage": "late"}

MODULE_NOTES: Dict[str, str] = {
    "se": "Squeeze-and-Excitation channel attention (FC bottleneck, ratio 16)",
    "eca": "Efficient Channel Attention (1D conv over channels, almost no parameters)",
    "cbam": "CBAM channel + spatial attention",
    "coordatt": "Coordinate Attention (direction-aware pooling)",
    "dilated": "3x3 conv with dilation 2 (larger receptive field, same params)",
    "depthwise": "depthwise-separable 3x3 conv (fewer params and FLOPs)",
    "weighted": "learnable weighted fusion of the last two stages' pooled features",
    "early": "modify stages 1-2",
    "late": "modify stages 3-4",
    "all": "modify stages 1-4",
}


def get_search_space(task: str) -> Dict[str, List[str]]:
    if task not in SEARCH_SPACES:
        raise KeyError(f"unknown task {task!r}; available: {list(SEARCH_SPACES)}")
    return SEARCH_SPACES[task]


def canonicalize(arch: Architecture) -> Architecture:
    """target_stage is meaningless when neither attention nor conv is changed."""
    a = {d: arch[d] for d in DIMENSIONS}
    if a["attention"] == "none" and a["conv"] == "standard":
        a["target_stage"] = BASELINE["target_stage"]
    return a


def architecture_signature(arch: Architecture) -> str:
    a = canonicalize(arch)
    return "__".join(a[d] for d in DIMENSIONS)


def from_signature(sig: str) -> Architecture:
    return dict(zip(DIMENSIONS, sig.split("__")))


def validate(arch: object, task: str) -> Optional[str]:
    """Return None if valid, otherwise a human-readable reason."""
    if not isinstance(arch, dict):
        return "architecture must be an object"
    space = get_search_space(task)
    for d in DIMENSIONS:
        if d not in arch:
            return f"missing dimension {d!r}"
        if arch[d] not in space[d]:
            return f"{d}={arch[d]!r} is not in the search space {space[d]}"
    extra = set(arch) - set(DIMENSIONS)
    if extra:
        return f"unknown dimensions {sorted(extra)}"
    return None


def count_architecture_difference(a: Architecture, b: Architecture) -> int:
    ca, cb = canonicalize(a), canonicalize(b)
    return sum(ca[d] != cb[d] for d in DIMENSIONS)


def all_architectures(task: str) -> List[Architecture]:
    """Every distinct (canonical) architecture, baseline included."""
    space = get_search_space(task)
    seen, out = set(), []
    for combo in itertools.product(*(space[d] for d in DIMENSIONS)):
        arch = canonicalize(dict(zip(DIMENSIONS, combo)))
        sig = architecture_signature(arch)
        if sig not in seen:
            seen.add(sig)
            out.append(arch)
    return out


def local_mutations(arch: Architecture, task: str, budget: int = 1) -> List[Architecture]:
    """All canonical architectures within `budget` dimension changes of `arch` (excluding itself)."""
    base = architecture_signature(arch)
    out = []
    for cand in all_architectures(task):
        if architecture_signature(cand) != base and count_architecture_difference(arch, cand) <= budget:
            out.append(cand)
    return out


def random_local_mutation(arch: Architecture, task: str, rng: random.Random,
                          exclude: Iterable[str] = (), budget: int = 1) -> Optional[Architecture]:
    excluded = set(exclude)
    for b in range(1, budget + 1):  # prefer the smallest change that is still untried
        cands = [c for c in local_mutations(arch, task, b) if architecture_signature(c) not in excluded]
        if cands:
            return rng.choice(cands)
    return None
