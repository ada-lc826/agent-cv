"""All LLM prompts. Kept in one place so the paper's appendix can quote them verbatim."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .search_space import DIMENSIONS, MODULE_NOTES

DESIGN_SYSTEM = """You are the Design Agent of VisionEvoAgent, a system that improves an established vision \
model by inserting modules from a fixed search space.
Each round you propose exactly ONE candidate architecture, which is then trained and evaluated.
Hard rules:
- Use only values that exist in SEARCH_SPACE.
- Never propose an architecture that was already evaluated or that is listed as failed.
- Respect MUTATION_BUDGET: the candidate may differ from the CURRENT BEST in at most that many dimensions.
- SEARCH_MODE "exploit": refine around the current best. SEARCH_MODE "explore": prefer a module or \
combination that has not been tried yet, still within the mutation budget.
- Score differences smaller than NOISE_LEVEL are not meaningful; do not chase them.
Return JSON only."""

DESIGN_REFLECTION_ADDENDUM = """
You also receive REFLECTION_CONTEXT: patterns, design principles and search directions distilled from \
earlier rounds. Prefer designs consistent with well-supported principles (high confidence, high \
support_count), avoid patterns marked ineffective, and say in the rationale which principle you follow."""

REFLECTOR_SYSTEM = """You are the Neural Architecture Reflection Agent of VisionEvoAgent.
From the experiment history of a module-insertion search on a vision model, induce reusable knowledge:
1. Modules that are repeatedly beneficial.
2. Modules that are repeatedly harmful or not worth their parameter cost.
3. Synergies or conflicts between modules (does a module help only in combination with another?).
4. Whether the gain matches the parameter growth.
5. 2-5 design principles that generalise beyond single trials.
6. Concrete directions for the next rounds.
Rules:
- Induce, do not restate. "ECA scored highest" is a restatement; "lightweight channel attention is \
consistently beneficial and outperforms heavier attention at this parameter budget" is an induction.
- Only mention modules that exist in SEARCH_SPACE; never invent new architectures.
- Treat differences smaller than NOISE_LEVEL as no difference.
- Base every claim on evidence in the history and cite it briefly (which rounds / scores).
- With few or noisy trials, lower the confidence values and set data_sufficiency accordingly.
Return JSON only."""


def _fmt_space(space: Dict[str, List[str]]) -> str:
    lines = []
    for d in DIMENSIONS:
        opts = ", ".join(f"{o} ({MODULE_NOTES[o]})" if o in MODULE_NOTES else o for o in space[d])
        lines.append(f"- {d}: {opts}")
    return "\n".join(lines)


def _fmt_history(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(no trials yet)"
    head = "round | attention | conv | fusion | target_stage | status | score | delta_vs_baseline | params(M) | param_ratio | note"
    out = [head]
    for r in rows:
        a = r["architecture"]
        score = "-" if r.get("score") is None else f"{r['score']:.2f}"
        delta = "-" if r.get("delta_vs_baseline") is None else f"{r['delta_vs_baseline']:+.2f}"
        params = "-" if r.get("params") is None else f"{r['params']:.2f}"
        ratio = "-" if r.get("param_ratio") is None else f"{r['param_ratio']:.3f}"
        note = r.get("failure_reason") or ""
        out.append(f"{r['round']} | {a['attention']} | {a['conv']} | {a['fusion']} | {a['target_stage']} | "
                   f"{r['status']} | {score} | {delta} | {params} | {ratio} | {note}")
    return "\n".join(out)


def _fmt_costs(costs: List[Dict[str, Any]], baseline_params: float, limit: int = 40) -> str:
    rows = sorted(costs, key=lambda c: c["params"])[:limit]
    out = []
    for c in rows:
        a = c["architecture"]
        out.append(f"- {a['attention']} | {a['conv']} | {a['fusion']} | {a['target_stage']}: "
                   f"{c['params'] - baseline_params:+.2f}M (x{c['params'] / baseline_params:.3f})")
    if len(costs) > limit:
        out.append(f"(... {len(costs) - limit} more, all larger)")
    return "\n".join(out)


def build_design_prompt(*, task: str, metric: str, space: Dict[str, List[str]], baseline: Dict[str, Any],
                        history: List[Dict[str, Any]], best: Dict[str, Any], failed: List[str],
                        search_mode: str, mutation_budget: int, noise_level: float,
                        reflection_context: Optional[Dict[str, Any]],
                        candidate_costs: Optional[List[Dict[str, Any]]] = None,
                        feedback: str = "") -> str:
    parts = [
        f"TASK: {task} (primary metric: {metric}, higher is better)",
        f"SEARCH_SPACE:\n{_fmt_space(space)}",
        f"BASELINE: {json.dumps(baseline['architecture'])} score={baseline['score']:.2f} params={baseline['params']:.2f}M",
        f"CURRENT BEST: {json.dumps(best['architecture'])} score={best['score']:.2f}",
        f"HISTORY:\n{_fmt_history(history)}",
        f"FAILED ARCHITECTURES (do not repeat): {failed if failed else 'none'}",
        f"SEARCH_MODE: {search_mode}",
        f"MUTATION_BUDGET: {mutation_budget}",
        f"NOISE_LEVEL: {noise_level:.2f} {metric} points",
    ]
    if candidate_costs:
        parts.append("CANDIDATE COSTS (every untried architecture within MUTATION_BUDGET of the current best, "
                     "as attention | conv | fusion | target_stage: params vs baseline). Weigh the expected gain "
                     "against this cost; a gain within NOISE_LEVEL does not justify extra parameters.\n"
                     + _fmt_costs(candidate_costs, baseline["params"]))
    if reflection_context is not None:
        parts.append(f"REFLECTION_CONTEXT:\n{json.dumps(reflection_context, indent=2, ensure_ascii=False)}")
    if feedback:
        parts.append(f"YOUR PREVIOUS ANSWER WAS REJECTED: {feedback}\nPropose a different candidate.")
    parts.append("Propose the next candidate architecture and explain why in one or two sentences.")
    return "\n\n".join(parts)


def build_reflection_prompt(*, task: str, metric: str, space: Dict[str, List[str]], baseline: Dict[str, Any],
                            history: List[Dict[str, Any]], recent_n: int, best: Dict[str, Any],
                            failed: List[Dict[str, Any]], noise_level: float,
                            previous_principles: List[Dict[str, Any]]) -> str:
    recent = history[-recent_n:] if recent_n > 0 else history
    parts = [
        f"TASK: {task} (primary metric: {metric}, higher is better)",
        f"SEARCH_SPACE:\n{_fmt_space(space)}",
        f"BASELINE: {json.dumps(baseline['architecture'])} score={baseline['score']:.2f} params={baseline['params']:.2f}M",
        f"CURRENT BEST: {json.dumps(best['architecture'])} score={best['score']:.2f} params={best.get('params', 0):.2f}M",
        f"NOISE_LEVEL: {noise_level:.2f} {metric} points (single-run seed noise of the proxy training)",
        f"FULL HISTORY:\n{_fmt_history(history)}",
        f"MOST RECENT {len(recent)} TRIALS:\n{_fmt_history(recent)}",
        f"FAILED CONFIGURATIONS: {json.dumps(failed) if failed else 'none'}",
        "PRINCIPLES FROM EARLIER REFLECTIONS (revise, confirm or drop them):\n"
        + (json.dumps(previous_principles, indent=2, ensure_ascii=False) if previous_principles else "none"),
        "Produce the reflection.",
    ]
    return "\n\n".join(parts)
