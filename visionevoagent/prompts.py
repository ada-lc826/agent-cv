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
You also receive REFLECTION_CONTEXT: patterns, search directions and DESIGN PRINCIPLES distilled from \
earlier rounds. Each principle has an id, a falsifiable statement, a confidence in [0, 1] that is updated \
by experimental evidence, and a status (confirmed > 0.75, uncertain, refuted < 0.25; refuted ones are \
not shown). Prefer designs consistent with confirmed principles; an uncertain principle is worth testing \
when the round is an exploration round. Avoid patterns marked ineffective. List in cited_principles the \
ids of the principles your proposal relies on (empty if none) and name them in the rationale."""

DESIGN_REFLECTION_ONLY_ADDENDUM = """
You do NOT see the raw experiment history. REFLECTION_CONTEXT, written by the Reflector after every \
round, is your only account of what has been learned; TRIED ARCHITECTURES only tells you what not to \
repeat. HYPOTHESES_TO_TEST are untested architectures the Reflector proposed to test its principles or \
to combine changes that looked promising separately. In an exploit round, propose one of them unless \
it is clearly worse than the alternatives; in an explore round you may test one or try something new. \
A hypothesis may differ from the current best in up to 2 dimensions even when MUTATION_BUDGET is 1."""

DESIGN_COMPUTED_ADDENDUM = """
Principle confidence is NOT an opinion: the system computes it from all trials with a statistical model \
that knows NOISE_LEVEL (posterior probability that the principle's claim holds). EVIDENCE_TABLE gives the \
model's effect estimates (+- sd). Each hypothesis lists the Reflector's predicted delta vs the baseline and \
the model's prediction (+- sd). In an exploit round prefer a hypothesis with a high predicted gain; in an \
explore round prefer one that tests a principle whose confidence is near 0.5 (most informative)."""

REFLECTOR_HYPOTHESES_ADDENDUM = """
Additional duties (required):
6. UNTESTED COMBINATIONS: propose 1-3 architectures that do NOT appear in TRIED ARCHITECTURES and that \
would test your principles most sharply or combine changes that each looked promising but were never \
tried together (e.g. two single-dimension changes that each helped). Each may differ from the CURRENT \
BEST in at most 2 dimensions. Give the ids of the principles each one tests and a one-sentence rationale.
Evidence (stricter about not wasting information):
- A new trial counts as evidence for every principle whose IF-condition it touches, even when the \
comparison is not perfectly controlled: compare it with the closest earlier trial and lower the \
strength for confounding instead of calling it neutral. Use neutral only when the new trials do not \
touch the principle or the difference is within NOISE_LEVEL."""

REFLECTOR_SYSTEM_V3 = """You are the Neural Architecture Reflection Agent of VisionEvoAgent.
From the experiment history of a module-insertion search on a vision model, you conjecture falsifiable \
design knowledge; the experiments decide whether it holds.
1. Effective and ineffective patterns (modules that help, hurt, or are not worth their parameter cost).
2. Synergies or conflicts between modules.
3. At most 3 NEW principles, only when the data suggest something the existing ones do not cover.
4. Concrete directions for the next rounds.
You do NOT judge evidence. The system computes every principle's confidence from all trials with a \
Bayesian linear model (main effect per module choice + the interactions your claims name, known seed \
noise NOISE_LEVEL): the posterior probability that the claim holds. You see the result in \
EVIDENCE_TABLE and in each existing principle's confidence and effect estimate. Use it: do not restate \
a principle the data already decide; propose principles about what is still uncertain or untested, \
in particular interactions (two choices that may help or hurt together more than separately).
Every new principle must pass all of:
- Mechanistic: says WHY (a causal mechanism), not just which configuration scored well.
- Scoped: states where it applies (stages, module family, model / task).
- Predictive: states the expected direction and rough size of the effect on the metric or parameters.
- Falsifiable: states what result would disprove it.
- Novel: not a restatement of an existing principle.
- Testable: can be checked by changing one or two dimensions of SEARCH_SPACE.
- Transferable: no round numbers or one-off scores in the statement itself.
Write it as fields: if (the architectural choice), scope, then (predicted effect), because (mechanism), \
disproved_if (falsification criterion), and a machine-checkable claim:
  condition: 1-2 {dim, value} terms (the choice; 2 terms = a joint choice / interaction),
  reference: what it is compared with, {dim, value} terms on the same dims ([] = the baseline values \
of those dims, e.g. attention=none),
  direction: increase | decrease (of the primary metric),
  min_effect: the size of the claimed effect in metric points (use NOISE_LEVEL unless you claim more).
Example: "eca together with weighted fusion" -> claim {condition: [{dim: attention, value: eca}, \
{dim: fusion, value: weighted}], reference: [], direction: increase, min_effect: 0.5}.
5. UNTESTED COMBINATIONS: propose 1-3 architectures that do NOT appear in TRIED ARCHITECTURES and that \
would test the most uncertain or most promising principles, or combine changes that each looked \
promising but were never tried together. Each may differ from the CURRENT BEST in at most 2 \
dimensions. Give the ids of the principles each one tests, a one-sentence rationale and \
predicted_delta: your prediction of its score minus the BASELINE score (it will be checked).
General rules:
- Only mention modules that exist in SEARCH_SPACE; never invent new architectures.
- With few or noisy trials, set data_sufficiency accordingly.
Return JSON only."""

REFLECTOR_SYSTEM = """You are the Neural Architecture Reflection Agent of VisionEvoAgent.
From the experiment history of a module-insertion search on a vision model, maintain a body of \
falsifiable design knowledge:
1. Effective and ineffective patterns (modules that help, hurt, or are not worth their parameter cost).
2. Synergies or conflicts between modules.
3. EVIDENCE for the existing principles: for every existing principle that the NEW TRIALS bear on, \
report supports / contradicts / neutral with a strength in [0, 1]. The system turns this into the \
principle's confidence (supporting evidence raises it, contradicting evidence lowers it).
4. At most 3 NEW principles, only when the evidence shows something the existing ones do not cover.
5. Concrete directions for the next rounds.
Evidence rules:
- Use only the NEW TRIALS SINCE THE LAST REFLECTION as evidence; earlier trials were already counted \
(each principle lists the rounds it was updated with). Give at most one update per principle.
- A difference within NOISE_LEVEL is neutral. Strength 0.3-0.6 for a difference between 1x and 2x \
NOISE_LEVEL or a single trial; 0.7-1.0 for more than 2x NOISE_LEVEL or consistent evidence from several trials.
- Name the rounds in "trials" and explain in one sentence in "reason".
Every new principle must pass all of:
- Mechanistic: says WHY (a causal mechanism), not just which configuration scored well.
- Scoped: states where it applies (stages, module family, model / task).
- Predictive: states the expected direction and rough size of the effect on the metric or parameters.
- Falsifiable: states what result would disprove it.
- Novel: not a restatement of an existing principle (give evidence for that one instead).
- Testable: can be checked by changing one or two dimensions of SEARCH_SPACE.
- Transferable: no round numbers or one-off scores in the statement itself.
Write it as fields: if (the architectural choice), scope, then (predicted effect), because (mechanism), \
disproved_if (falsification criterion).
General rules:
- Induce, do not restate. "ECA scored highest" is a restatement; "lightweight channel attention is \
consistently beneficial and outperforms heavier attention at this parameter budget" is an induction.
- Only mention modules that exist in SEARCH_SPACE; never invent new architectures.
- With few or noisy trials, prefer weak strengths and set data_sufficiency accordingly.
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
                        feedback: str = "", show_history: bool = True,
                        tried: Optional[List[str]] = None,
                        hypotheses: Optional[List[Dict[str, Any]]] = None,
                        evidence_table: Optional[List[Dict[str, Any]]] = None) -> str:
    if show_history:
        hist = f"HISTORY:\n{_fmt_history(history)}"
    else:  # v2: the Reflector's summary replaces the raw history; only the "do not repeat" list remains
        hist = ("TRIED ARCHITECTURES (attention__conv__fusion__target_stage; do not repeat): "
                + (", ".join(sorted(tried or [])) or "none"))
    parts = [
        f"TASK: {task} (primary metric: {metric}, higher is better)",
        f"SEARCH_SPACE:\n{_fmt_space(space)}",
        f"BASELINE: {json.dumps(baseline['architecture'])} score={baseline['score']:.2f} params={baseline['params']:.2f}M",
        f"CURRENT BEST: {json.dumps(best['architecture'])} score={best['score']:.2f}",
        hist,
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
    if evidence_table:
        parts.append("EVIDENCE_TABLE (computed from all trials; estimate +- sd in metric points):\n"
                     + _fmt_evidence(evidence_table))
    if hypotheses:
        parts.append("HYPOTHESES_TO_TEST (untested; proposed by the Reflector):\n"
                     + json.dumps(hypotheses, indent=2, ensure_ascii=False))
    if feedback:
        parts.append(f"YOUR PREVIOUS ANSWER WAS REJECTED: {feedback}\nPropose a different candidate.")
    parts.append("Propose the next candidate architecture and explain why in one or two sentences.")
    return "\n\n".join(parts)


def _fmt_evidence(rows: List[Dict[str, Any]]) -> str:
    return "\n".join(f"- {r['term']} ({r['kind']}): {r['estimate']:+.2f} +- {r['sd']:.2f}, "
                     f"in {r['n_trials_with_it']} scored trials" for r in rows)


def principle_view(p: Dict[str, Any]) -> Dict[str, Any]:
    """Compact form of a stored principle for prompts."""
    view = {"id": p.get("id", ""), "statement": p.get("principle", ""), "confidence": round(p.get("confidence", 0.5), 3),
            "status": p.get("status", "uncertain"), "supports": p.get("support_count", 0),
            "contradicts": p.get("contradict_count", 0),
            "updated_in_rounds": sorted({e["round"] for e in p.get("evidence", [])})}
    if p.get("transferred_from"):  # learned in another task; its earlier rounds belong to that task
        view["transferred_from"] = p["transferred_from"]
    if p.get("claim"):  # v3: confidence computed from the trials
        from .evidence import claim_text
        view["claim"] = claim_text(p["claim"])
        st = p.get("stats") or {}
        if st:
            view["effect_estimate"] = f"{st['effect_mean']:+.2f} +- {st['effect_sd']:.2f}"
            view["direct_trials"] = st.get("n_direct", 0)
        for k in ("supports", "contradicts"):
            view.pop(k, None)
    return view


def build_reflection_prompt(*, task: str, metric: str, space: Dict[str, List[str]], baseline: Dict[str, Any],
                            history: List[Dict[str, Any]], new_trials: List[Dict[str, Any]], best: Dict[str, Any],
                            failed: List[Dict[str, Any]], noise_level: float,
                            previous_principles: List[Dict[str, Any]],
                            tried: Optional[List[str]] = None,
                            evidence_table: Optional[List[Dict[str, Any]]] = None) -> str:
    computed = evidence_table is not None  # v3: the system, not the Reflector, scores the principles
    new_hdr = ("NEW TRIALS SINCE THE LAST REFLECTION" if computed
               else "NEW TRIALS SINCE THE LAST REFLECTION (the only admissible new evidence)")
    prin_hdr = ("EXISTING PRINCIPLES (confidence computed from the trials; do not restate them as new)" if computed
                else "EXISTING PRINCIPLES (report evidence for them by id; do not restate them as new)")
    parts = [
        f"TASK: {task} (primary metric: {metric}, higher is better)",
        f"SEARCH_SPACE:\n{_fmt_space(space)}",
        f"BASELINE: {json.dumps(baseline['architecture'])} score={baseline['score']:.2f} params={baseline['params']:.2f}M",
        f"CURRENT BEST: {json.dumps(best['architecture'])} score={best['score']:.2f} params={best.get('params', 0):.2f}M",
        f"NOISE_LEVEL: {noise_level:.2f} {metric} points (single-run seed noise of the proxy training)",
        f"FULL HISTORY (context):\n{_fmt_history(history)}",
        f"{new_hdr}:\n{_fmt_history(new_trials)}",
        f"FAILED CONFIGURATIONS: {json.dumps(failed) if failed else 'none'}",
        f"{prin_hdr}:\n"
        + (json.dumps([principle_view(p) for p in previous_principles], indent=2, ensure_ascii=False)
           if previous_principles else "none"),
    ]
    if evidence_table is not None:
        parts.append("EVIDENCE_TABLE (Bayesian linear model over all scored trials, baseline included; "
                     "estimate +- sd in metric points):\n" + (_fmt_evidence(evidence_table) or "none"))
    if tried is not None:
        parts.append("TRIED ARCHITECTURES (attention__conv__fusion__target_stage; untested combinations must "
                     "not be among these): " + (", ".join(sorted(tried)) or "none"))
    parts.append("Produce the reflection.")
    return "\n\n".join(parts)
