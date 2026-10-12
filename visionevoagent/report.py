"""Run outputs: search_history.csv, search_summary.json, evolution.json (best-so-far curve)."""
from __future__ import annotations

import csv
import json
import os
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
    out = {"calls": 0, "failed_calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out["calls"] += 1
            out["failed_calls"] += 0 if rec.get("ok") else 1
            out["input_tokens"] += rec.get("input_tokens") or 0
            out["output_tokens"] += rec.get("output_tokens") or 0
            out["latency_s"] = round(out["latency_s"] + (rec.get("latency_s") or 0.0), 2)
    return out


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
        "n_trials": len(mem.trials), "n_ok": len(ok), "n_failed": len(mem.trials) - len(ok),
        "baseline": asdict(mem.baseline) if mem.baseline else None,
        "best": asdict(best) if best else None,
        "best_improvement": round(best.score - mem.baseline.score, 3) if best and mem.baseline
        and best.score is not None and mem.baseline.score is not None else None,
        "rounds_to_best": best.round if best else None,
        "gpu_hours": round(sum(t.duration_s for t in rows) / 3600, 3),
        "llm": _llm_usage(root),
        "n_reflections": len(mem.reflections),
        "principles": [asdict(p) for p in sorted(mem.principles, key=lambda p: (-p.support_count, -p.confidence))],
        "failures": [asdict(x) for x in mem.failures],
    }
    with open(os.path.join(root, "search_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
