"""JSON schemas for LLM outputs (sent as structured-output format) and validators."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .search_space import DIMENSIONS, get_search_space


def design_schema(task: str) -> Dict[str, Any]:
    space = get_search_space(task)
    return {
        "type": "object",
        "properties": {
            "architecture": {
                "type": "object",
                "properties": {d: {"type": "string", "enum": space[d]} for d in DIMENSIONS},
                "required": list(DIMENSIONS),
                "additionalProperties": False,
            },
            "rationale": {"type": "string"},
        },
        "required": ["architecture", "rationale"],
        "additionalProperties": False,
    }


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

REFLECTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "effective_patterns": {"type": "array", "items": _PATTERN},
        "ineffective_patterns": {"type": "array", "items": _PATTERN},
        "synergies": {"type": "array", "items": {"type": "string"}},
        "design_principles": {"type": "array", "items": {"type": "string"}},
        "next_search_direction": {"type": "array", "items": {"type": "string"}},
        "data_sufficiency": {"type": "string", "enum": ["insufficient", "limited", "adequate"]},
    },
    "required": ["effective_patterns", "ineffective_patterns", "synergies", "design_principles",
                 "next_search_direction", "data_sufficiency"],
    "additionalProperties": False,
}


def _err(msg: str) -> Optional[str]:
    return msg


def validate_design(obj: Any, task: str) -> Optional[str]:
    """Return None if valid else a reason. The structured-output format already constrains
    the shape; this is a second, provider-independent check (also used for the mock client)."""
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
    for key in ("synergies", "design_principles", "next_search_direction"):
        v = obj.get(key)
        if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
            return _err(f"{key} must be a list of strings")
    if not 1 <= len(obj["design_principles"]) <= 5:
        return _err("design_principles must contain 1-5 items")
    if obj.get("data_sufficiency") not in ("insufficient", "limited", "adequate"):
        return _err("bad data_sufficiency")
    return None


def clamp_list(items: List[Any], n: int) -> List[Any]:
    return list(items)[:n]
