"""Do the principles a search decided on the proxy also hold under the full training recipe?

Inputs: a run's principle_memory.json (claims decided on the 60-epoch proxy) and full-training results
(200 epochs, official recipe) of architectures that were trained fully, e.g. the 16 of the proxy check.
The same Bayesian linear model as in the search (visionevoagent/evidence.py) is fitted on the
full-training results, with the full-recipe seed noise, and every claim the search confirmed or refuted
is re-scored on that independent data:
  agree         confirmed on the proxy and > 0.75 on full training, or refuted and < 0.25
  contradict    confirmed and < 0.25, or refuted and > 0.75
  undetermined  in between (the 16 full results do not decide it)
Also reports how well the search's own evidence model (fitted on its proxy trials) ranks the fully
trained architectures, separately for architectures the search never trained.

    python tools/check_principles_full.py --run runs_b/classification_agent_reflect_t30_s0 \\
        --full configs/full_results_cls.json --out runs_b/principles_vs_full.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionevoagent.evidence import claim_text, fit  # noqa: E402
from visionevoagent.search_space import from_signature  # noqa: E402


def spearman(x: List[float], y: List[float]) -> float:
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", help="run directory (principle_memory.json, trial_memory.json, baseline.json)")
    p.add_argument("--principles", help="principle_memory.json (instead of --run)")
    p.add_argument("--proxy", help="JSON {signature: proxy score} for the search's evidence model (instead of --run)")
    p.add_argument("--full", required=True, help="JSON {baseline_signature, noise_sd, scores: {signature: score}}")
    p.add_argument("--task", default="classification")
    p.add_argument("--proxy_noise", type=float, default=0.25)
    p.add_argument("--out", default="")
    a = p.parse_args()

    with open(a.full, encoding="utf-8") as f:
        full = json.load(f)
    base_sig = full["baseline_signature"]
    scores: Dict[str, float] = full["scores"]
    principles = json.load(open(a.principles or os.path.join(a.run, "principle_memory.json"), encoding="utf-8"))
    decided = [x for x in principles if x.get("claim") and x.get("status") in ("confirmed", "refuted")]
    claims = [x["claim"] for x in decided]

    rows = [{"architecture": from_signature(s), "status": "ok", "score": v} for s, v in scores.items() if s != base_sig]
    model = fit(rows, {"architecture": from_signature(base_sig), "status": "ok", "score": scores[base_sig]},
                a.task, full["noise_sd"], claims=claims)

    out: List[str] = ["| proxy status | proxy conf | claim | full-training effect | full conf | verdict |",
                      "|---|---|---|---|---|---|"]
    tally = {"agree": 0, "contradict": 0, "undetermined": 0}
    seen = set()
    for x in sorted(decided, key=lambda x: (x["status"], -x["confidence"])):
        key = claim_text(x["claim"])
        if key in seen:  # the same claim stated twice (different ids) is counted once
            continue
        seen.add(key)
        st = model.confidence(x["claim"])
        c = st["confidence"]
        if (x["status"] == "confirmed" and c > 0.75) or (x["status"] == "refuted" and c < 0.25):
            v = "agree"
        elif (x["status"] == "confirmed" and c < 0.25) or (x["status"] == "refuted" and c > 0.75):
            v = "contradict"
        else:
            v = "undetermined"
        tally[v] += 1
        out.append(f"| {x['status']} | {x['confidence']:.2f} | {key} | {st['effect_mean']:+.2f} ± {st['effect_sd']:.2f} "
                   f"({st['n_direct']} direct) | {c:.2f} | {v} |")
    decided_n = tally["agree"] + tally["contradict"]
    summary = [f"distinct decided claims: {len(seen)}; agree {tally['agree']}, contradict {tally['contradict']}, "
               f"undetermined {tally['undetermined']}"
               + (f"; agreement among those full training decides: {tally['agree'] / decided_n:.0%}" if decided_n else "")]

    # the search's evidence model (proxy trials) predicting the full-training ranking
    if a.run or a.proxy:
        if a.proxy:
            proxy = json.load(open(a.proxy, encoding="utf-8"))
            pbase = proxy.pop("__baseline__")
        else:
            trials = json.load(open(os.path.join(a.run, "trial_memory.json"), encoding="utf-8"))
            proxy = {t["signature"]: t["score"] for t in trials if t.get("status") == "ok"}
            pbase = json.load(open(os.path.join(a.run, "baseline.json"), encoding="utf-8"))["score"]
        pm = fit([{"architecture": from_signature(s), "status": "ok", "score": v} for s, v in proxy.items()],
                 {"architecture": from_signature(base_sig), "status": "ok", "score": pbase}, a.task, a.proxy_noise,
                 claims=[x["claim"] for x in principles if x.get("claim")])
        sigs = [s for s in scores if s != base_sig]
        pred = {s: pm.predict(from_signature(s))[0] for s in sigs}
        unseen = [s for s in sigs if s not in proxy]
        rho_all = spearman([pred[s] for s in sigs], [scores[s] for s in sigs])
        line = f"search evidence model -> full-training ranking: Spearman {rho_all:.2f} over {len(sigs)} architectures"
        if len(unseen) >= 4:
            rho_u = spearman([pred[s] for s in unseen], [scores[s] for s in unseen])
            line += f"; {rho_u:.2f} over the {len(unseen)} the search never trained"
        summary.append(line)
    text = "\n".join(summary) + "\n\n" + "\n".join(out)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
