"""JSON schemas for LLM outputs (sent as structured-output format or, for DeepSeek, written into the
prompt) and provider-independent validators."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .search_space import DIMENSIONS, get_search_space

# Same for every LLM method, ours included (decision 2026-10-13): one ask plus at most 2 re-asks with
# the rejection reason as feedback; after that the round falls back to a random untried architecture.
MAX_PROPOSAL_ATTEMPTS = 3


def attempt_outcome(reason: Optional[str]) -> str:
    """Category of one proposed architecture, for the duplicate-rate statistics: ok | duplicate | invalid."""
    if reason is None:
        return "ok"
    if "already evaluated" in reason or "known failed configuration" in reason:
        return "duplicate"
    return "invalid"

EVIDENCE_TYPES = ["supports", "contradicts", "neutral"]
PRINCIPLE_FIELDS = ["if", "scope", "then", "because", "disproved_if"]
MAX_NEW_PRINCIPLES = 3


def design_schema(task: str, with_citations: bool = False) -> Dict[str, Any]:
    space = get_search_space(task)
    props: Dict[str, Any] = {
        "architecture": {
            "type": "object",
            "properties": {d: {"type": "string", "enum": space[d]} for d in DIMENSIONS},
            "required": list(DIMENSIONS),
            "additionalProperties": False,
        },
        "rationale": {"type": "string"},
    }
    required = ["architecture", "rationale"]
    if with_citations:
        # ids of the principles in REFLECTION_CONTEXT this proposal relies on (may be empty)
        props["cited_principles"] = {"type": "array", "items": {"type": "string"}}
        required.append("cited_principles")
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


_PATTERN = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string"},
        "evidence": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["pattern", "evidence", "confidence"],
    "additionalProperties": False,
}

_PRINCIPLE_UPDATE = {
    "type": "object",
    "properties": {
        "principle_id": {"type": "string"},
        "evidence": {"type": "string", "enum": EVIDENCE_TYPES},
        "strength": {"type": "number"},
        "trials": {"type": "string"},          # which new rounds this evidence comes from, e.g. "r5, r6"
        "reason": {"type": "string"},
    },
    "required": ["principle_id", "evidence", "strength", "trials", "reason"],
    "additionalProperties": False,
}

_NEW_PRINCIPLE = {
    "type": "object",
    "properties": {f: {"type": "string"} for f in PRINCIPLE_FIELDS},
    "required": list(PRINCIPLE_FIELDS),
    "additionalProperties": False,
}

REFLECTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "effective_patterns": {"type": "array", "items": _PATTERN},
        "ineffective_patterns": {"type": "array", "items": _PATTERN},
        "synergies": {"type": "array", "items": {"type": "string"}},
        "principle_updates": {"type": "array", "items": _PRINCIPLE_UPDATE},
        "new_principles": {"type": "array", "items": _NEW_PRINCIPLE},
        "next_search_direction": {"type": "array", "items": {"type": "string"}},
        "data_sufficiency": {"type": "string", "enum": ["insufficient", "limited", "adequate"]},
    },
    "required": ["effective_patterns", "ineffective_patterns", "synergies", "principle_updates",
                 "new_principles", "next_search_direction", "data_sufficiency"],
    "additionalProperties": False,
}


def _err(msg: str) -> Optional[str]:
    return msg


def validate_design(obj: Any, task: str) -> Optional[str]:
    """Return None if valid else a reason. Claude's structured output already constrains the shape;
    DeepSeek's JSON mode does not, so this check is what actually enforces it there."""
    if not isinstance(obj, dict):
        return _err("response is not an object")
    if not isinstance(obj.get("rationale"), str):
        return _err("missing rationale")
    arch = obj.get("architecture")
    if not isinstance(arch, dict):
        return _err("missing architecture")
    space = get_search_space(task)
    for d in DIMENSIONS:
        if arch.get(d) not in space[d]:
            return _err(f"{d}={arch.get(d)!r} not in search space")
    cited = obj.get("cited_principles", [])
    if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
        return _err("cited_principles must be a list of principle ids")
    return None


def validate_reflection(obj: Any) -> Optional[str]:
    if not isinstance(obj, dict):
        return _err("response is not an object")
    for key in ("effective_patterns", "ineffective_patterns"):
        items = obj.get(key)
        if not isinstance(items, list):
            return _err(f"{key} must be a list")
        for it in items:
            if not isinstance(it, dict) or not isinstance(it.get("pattern"), str):
                return _err(f"bad item in {key}")
            c = it.get("confidence")
            if not isinstance(c, (int, float)) or not 0.0 <= float(c) <= 1.0:
                return _err(f"confidence out of [0,1] in {key}")
    for key in ("synergies", "next_search_direction"):
        v = obj.get(key)
        if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
            return _err(f"{key} must be a list of strings")
    ups = obj.get("principle_updates")
    if not isinstance(ups, list):
        return _err("principle_updates must be a list")
    for u in ups:
        if not isinstance(u, dict) or not isinstance(u.get("principle_id"), str):
            return _err("bad item in principle_updates")
        if u.get("evidence") not in EVIDENCE_TYPES:
            return _err(f"evidence must be one of {EVIDENCE_TYPES}")
        s = u.get("strength")
        if not isinstance(s, (int, float)) or not 0.0 <= float(s) <= 1.0:
            return _err("strength out of [0,1] in principle_updates")
    new = obj.get("new_principles")
    if not isinstance(new, list) or len(new) > MAX_NEW_PRINCIPLES:
        return _err(f"new_principles must be a list of at most {MAX_NEW_PRINCIPLES} items")
    for p in new:
        if not isinstance(p, dict) or any(not isinstance(p.get(f), str) or not p[f].strip()
                                          for f in PRINCIPLE_FIELDS):
            return _err(f"every new principle needs non-empty fields {PRINCIPLE_FIELDS}")
    if obj.get("data_sufficiency") not in ("insufficient", "limited", "adequate"):
        return _err("bad data_sufficiency")
    return None


def clamp_list(items: List[Any], n: int) -> List[Any]:
    return list(items)[:n]
