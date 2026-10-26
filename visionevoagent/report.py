"""Run outputs: search_history.csv, search_summary.json, evolution.json (best-so-far curve)."""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List

from .memory import ArchitectureMemory
from .search_space import DIMENSIONS

if TYPE_CHECKING:
    from .search import SearchConfig

CSV_FIELDS = ["round", "signature"] + DIMENSIONS + [
    "status", "score", "params", "delta_vs_baseline", "delta_vs_previous", "param_ratio", "is_best",
    "search_mode", "failure_reason", "duration_s", "proposer_note"]


def _llm_usage(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "llm_calls.jsonl")
    """Cost of the method's LLM use, reported next to GPU hours (equal trial counts alone are not fairness)."""
    out = {"calls": 0, "failed_calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
           "cache_hit_tokens": 0, "latency_s": 0.0, "calls_by_purpose": {}, "models_served": []}
    if not os.path.exists(path):
        return out
    purposes, models = Counter(), set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out["calls"] += 1
            out["failed_calls"] += 0 if rec.get("ok") else 1
            for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cache_hit_tokens"):
                out[k] += rec.get(k) or 0
            out["latency_s"] = round(out["latency_s"] + (rec.get("latency_s") or 0.0), 2)
            purposes[rec.get("purpose", "?")] += 1
            models.add(str(rec.get("served_model") or rec.get("model") or "?"))
    out["calls_by_purpose"] = dict(purposes)
    out["models_served"] = sorted(models)
    return out


def _proposal_stats(mem: ArchitectureMemory) -> Dict[str, Any]:
    """Duplicate rate and fallback share (decision 2026-10-13): every method gets one ask plus at most two
    re-asks; an answer naming an already-evaluated architecture counts as a duplicate, and a round whose
    attempts were all rejected falls back to a random untried architecture."""
    answers = [o for t in mem.trials for o in t.proposal_attempts if o != "call_error"]
    n = len(mem.trials)
    fallback = sum(1 for t in mem.trials if "fallback" in t.proposer_note.split(":")[0])
    dup = answers.count("duplicate")
    return {
        "n_llm_answers": len(answers),
        "n_duplicate_answers": dup,
        "duplicate_rate": round(dup / len(answers), 3) if answers else None,
        "n_rounds_reasked": sum(1 for t in mem.trials if len(t.proposal_attempts) > 1),
        "fallback_rate": round(fallback / n, 3) if n else None,
        # reflection v2: rounds that tested one of the Reflector's untested combinations
        "n_hypothesis_rounds": sum(1 for t in mem.trials if t.proposer_note.startswith("llm[hypothesis]")),
    }


def _corr(xs: List[float], ys: List[float]) -> Any:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 3) if sx and sy else None


def _ranks(v: List[float]) -> List[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):  # average ranks for ties
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def _prediction_stats(mem: ArchitectureMemory) -> Dict[str, Any]:
    """Reflection v3: were the predictions made before training right? For the Reflector's predicted
    delta (hypothesis rounds) and the evidence model's prediction (every round): n, MAE, Pearson and
    Spearman correlation with the measured delta vs the baseline, and the share with the right sign."""
    out: Dict[str, Any] = {}
    # v3 keys predict the delta vs the baseline; v4 keys (*_vs_parent) the parent -> child difference
    for key in ("reflector", "model", "design_vs_parent", "model_vs_parent", "combined_vs_parent"):
        target = "delta_vs_parent" if key.endswith("_vs_parent") else "delta_vs_baseline"
        pairs = [(float(t.predictions[key]), getattr(t, target)) for t in mem.trials
                 if t.status == "ok" and getattr(t, target) is not None and key in (t.predictions or {})]
        if not pairs:
            continue
        p, y = [a for a, _ in pairs], [b for _, b in pairs]
        out[key] = {"n": len(pairs), "mae": round(sum(abs(a - b) for a, b in pairs) / len(pairs), 3),
                    "pearson": _corr(p, y), "spearman": _corr(_ranks(p), _ranks(y)),
                    "sign_agreement": round(sum((a > 0) == (b > 0) for a, b in pairs) / len(pairs), 3)}
    return out


def _selection_stats(mem: ArchitectureMemory) -> Dict[str, Any]:
    """Hypothesis-guided selection: how often it overrode the Design Agent's first candidate, and the
    mean terms of the chosen candidates; plus the library's parent -> child records."""
    sel = [t.selection for t in mem.trials if t.selection and t.selection.get("candidates")]
    if not sel:
        return {}
    chosen = [s["candidates"][s["chosen"]] for s in sel]
    mean = lambda k: round(sum(c[k] for c in chosen) / len(chosen), 3)  # noqa: E731
    pairs = Counter(e.get("type") for p in mem.principles for e in p.evidence
                    if e.get("type") in ("support", "counterexample", "inconclusive"))
    return {"rounds": len(sel), "overrode_first_candidate": sum(s["chosen"] != 0 for s in sel),
            "mean_candidates": round(sum(len(s["candidates"]) for s in sel) / len(sel), 2),
            "chosen_mean_gain": mean("gain"), "chosen_mean_uncertainty": mean("uncertainty"),
            "chosen_mean_verification": mean("verification"), "pair_evidence": dict(pairs)}


def evolution_curve(mem: ArchitectureMemory) -> List[Dict[str, Any]]:
    best = mem.baseline.score if mem.baseline else None
    sig = mem.baseline.signature if mem.baseline else None
    curve = [{"round": 0, "best_score": best, "best_signature": sig}]
    for t in sorted(mem.trials, key=lambda t: t.round):
        if t.is_best:
            best, sig = t.score, t.signature
        curve.append({"round": t.round, "score": t.score, "status": t.status,
                      "best_score": best, "best_signature": sig})
    return curve


def write_reports(mem: ArchitectureMemory, cfg: "SearchConfig", stop_reason: str) -> Dict[str, Any]:
    root = cfg.run_dir
    rows = ([mem.baseline] if mem.baseline else []) + sorted(mem.trials, key=lambda t: t.round)
    with open(os.path.join(root, "search_history.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for t in rows:
            d = asdict(t)
            w.writerow({**{k: d.get(k) for k in CSV_FIELDS if k not in DIMENSIONS},
                        **{k: t.architecture[k] for k in DIMENSIONS}})

    curve = evolution_curve(mem)
    with open(os.path.join(root, "evolution.json"), "w", encoding="utf-8") as f:
        json.dump(curve, f, indent=2)

    best = mem.best_trial() or mem.baseline
    ok = mem.ok_trials()
    summary = {
        "task": cfg.task, "mode": cfg.mode, "seed": cfg.seed, "stop_reason": stop_reason,
        "rounds_done": mem.last_round(), "max_rounds": cfg.max_rounds,
        "variant": getattr(cfg, "variant", ""),
        "n_trials": len(mem.trials), "n_ok": len(ok), "n_failed": len(mem.trials) - len(ok),
        # share of trials that produced no score (crash / OOM / NaN / timeout); compare with LLM-GE's 40-52%
        "invalid_trial_rate": round((len(mem.trials) - len(ok)) / len(mem.trials), 3) if mem.trials else None,
        "failure_reasons": dict(Counter(t.failure_reason or "unknown" for t in mem.trials if t.status != "ok")),
        # trials proposed by a fallback (LLM answer unusable) rather than by the method itself
        "n_fallback_proposals": sum(1 for t in mem.trials if "fallback" in t.proposer_note.split(":")[0]),
        **_proposal_stats(mem),
        "baseline": asdict(mem.baseline) if mem.baseline else None,
        "best": asdict(best) if best else None,
        "best_improvement": round(best.score - mem.baseline.score, 3) if best and mem.baseline
        and best.score is not None and mem.baseline.score is not None else None,
        "best_delta_params": round(best.params - mem.baseline.params, 3) if best and mem.baseline
        and best.params is not None and mem.baseline.params is not None else None,
        "rounds_to_best": best.round if best else None,
        "gpu_hours": round(sum(t.duration_s for t in rows) / 3600, 3),
        "llm": _llm_usage(root),
        "n_reflections": len(mem.reflections),
        "principle_status": dict(Counter(p.status for p in mem.principles)),
        "n_trials_citing_principles": sum(1 for t in mem.trials if t.cited_principles),
        "predictions": _prediction_stats(mem),
        "selection": _selection_stats(mem),
        "max_observed_improvement": round(max((t.delta_vs_baseline for t in mem.trials
                                               if t.delta_vs_baseline is not None), default=0.0), 3),
        "principles": [asdict(p) for p in sorted(mem.principles, key=lambda p: -p.confidence)],
        "failures": [asdict(x) for x in mem.failures],
    }
    with open(os.path.join(root, "search_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
