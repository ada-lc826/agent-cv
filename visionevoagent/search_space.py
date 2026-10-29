"""Architecture search space, signatures and mutation helpers.

An architecture is a dict over the dimensions of SEARCH_SPACE, e.g.
{"attention": "eca", "conv": "dilated", "fusion": "none", "target_stage": "late"}.

Two spaces (profiles) exist. "modules" is the module-insertion space of the classification experiments;
"nb201" is the NAS-Bench-201 cell space (2026-10-30, second set of experiments; tasks nb201_cifar10,
nb201_cifar100, nb201_imagenet16). A process works in one space: run_search.py calls activate(task)
once, which rewrites DIMENSIONS, BASELINE and MODULE_NOTES in place so that every module that
imported them sees the active space. The modules space is active by default.
"""
from __future__ import annotations

import itertools
import random
from typing import Dict, Iterable, List, Optional

Architecture = Dict[str, str]

NB201_OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
# edge_i_j: the operation from node i to node j of the 4-node cell (NAS-Bench-201 string order)
NB201_EDGES = ["edge_0_1", "edge_0_2", "edge_1_2", "edge_0_3", "edge_1_3", "edge_2_3"]
_NB201_SPACE = {e: list(NB201_OPS) for e in NB201_EDGES}

# Same dimensions for every task of a profile; the runner decides where each choice is inserted.
SEARCH_SPACES: Dict[str, Dict[str, List[str]]] = {
    "classification": {
        "attention": ["none", "se", "eca", "cbam", "coordatt"],
        "conv": ["standard", "dilated", "depthwise"],
        "fusion": ["none", "weighted"],
        "target_stage": ["early", "late", "all"],
    },
    "nb201_cifar10": _NB201_SPACE,
    "nb201_cifar100": _NB201_SPACE,
    "nb201_imagenet16": _NB201_SPACE,
}

PROFILES: Dict[str, Dict] = {
    "modules": {
        "dimensions": ["attention", "conv", "fusion", "target_stage"],
        "baseline": {"attention": "none", "conv": "standard", "fusion": "none", "target_stage": "late"},
        "notes": {
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
        },
        "space_note": "",
    },
    "nb201": {
        "dimensions": list(NB201_EDGES),
        # the ResNet-style cell of NAS-Bench-201: two 3x3 convs with a skip connection around them
        "baseline": {"edge_0_1": "nor_conv_3x3", "edge_0_2": "none", "edge_1_2": "nor_conv_3x3",
                     "edge_0_3": "skip_connect", "edge_1_3": "none", "edge_2_3": "skip_connect"},
        "notes": {
            "none": "no connection (the edge outputs zero)",
            "skip_connect": "identity",
            "nor_conv_1x1": "ReLU-Conv1x1-BN",
            "nor_conv_3x3": "ReLU-Conv3x3-BN",
            "avg_pool_3x3": "3x3 average pooling",
        },
        "space_note": ("NAS-Bench-201 cell: nodes 0-3, node 0 is the cell input, node j = sum over i < j of "
                       "op(edge_i_j)(node i), node 3 is the cell output. The network stacks this cell "
                       "(3 stages x 5 cells with residual reduction blocks); edge_i_j chooses one operation."),
    },
}

ACTIVE = "modules"
DIMENSIONS: List[str] = list(PROFILES["modules"]["dimensions"])
BASELINE: Architecture = dict(PROFILES["modules"]["baseline"])
MODULE_NOTES: Dict[str, str] = dict(PROFILES["modules"]["notes"])


def profile_of(task: str) -> str:
    return "nb201" if task.startswith("nb201") else "modules"


def activate(task: str) -> str:
    """Make the space of `task` the active one (in place, so imported names follow). Returns the profile."""
    global ACTIVE
    get_search_space(task)
    name = profile_of(task)
    prof = PROFILES[name]
    DIMENSIONS[:] = prof["dimensions"]
    BASELINE.clear()
    BASELINE.update(prof["baseline"])
    MODULE_NOTES.clear()
    MODULE_NOTES.update(prof["notes"])
    ACTIVE = name
    return name


def space_note() -> str:
    return PROFILES[ACTIVE]["space_note"]


def get_search_space(task: str) -> Dict[str, List[str]]:
    if task not in SEARCH_SPACES:
        raise KeyError(f"unknown task {task!r}; available: {list(SEARCH_SPACES)}")
    return SEARCH_SPACES[task]


def canonicalize(arch: Architecture) -> Architecture:
    """Modules space: target_stage is meaningless when neither attention nor conv is changed.
    NAS-Bench-201: every one of the 15,625 cells is its own entry of the benchmark (no merging)."""
    a = {d: arch[d] for d in DIMENSIONS}
    if ACTIVE == "modules" and a["attention"] == "none" and a["conv"] == "standard":
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
