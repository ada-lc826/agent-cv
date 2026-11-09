"""Paper figures as vector SVG, without dependencies (2026-11-03).

Inputs (written from the analysis outputs, see their _source fields):
  curves.json   tools/export_curves.py: per-round mean / sd of the best improvement, per method and task
  results.json  calibration bins, principle audit, NAS-Bench-201 CIFAR-100 ablation AUCs

    python tools/plot_figures.py --data <dir with curves.json, results.json> --out <dir>

Writes fig1_search_curves, fig2_transfer, fig3_calibration, fig4_audit, fig5_ablation (.svg). Colours are
the categorical slots of a colour-blind-checked palette, assigned in a fixed order (same method = same
colour in every figure); methods beyond seven share one grey. For PDF: inkscape --export-type=pdf *.svg
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

INK, INK2, GRID, SURF, GREY = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff", "#b5b4ae"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
FONT = "font-family='Arial, Helvetica, sans-serif'"

METHODS = [  # key in curves.json, label, colour; drawn in reverse so ours is on top
    ("agent_reflect", "Ours (Agent + principle library)", SLOTS[0]),
    ("agent", "Agent only (no library)", SLOTS[1]),
    ("genius", "GENIUS", SLOTS[2]),
    ("nader", "NADER", SLOTS[3]),
    ("tpe", "TPE", SLOTS[4]),
    ("reg_evo", "Reg-Evo", SLOTS[5]),
    ("random", "Random", SLOTS[6]),
]
OTHERS = ["llmatic", "lapt", "evoprompting"]
TASKS = [("classification", "t30", "(a) Module space, ResNet-50 / CIFAR-100"),
         ("nb201_cifar10", "t50", "(b) NAS-Bench-201, CIFAR-10"),
         ("nb201_cifar100", "t50", "(c) NAS-Bench-201, CIFAR-100"),
         ("nb201_imagenet16", "t50", "(d) NAS-Bench-201, ImageNet16-120")]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=11, color=INK2, anchor="start", weight="normal", rotate=None) -> str:
    rot = f" transform='rotate({rotate} {x:.1f} {y:.1f})'" if rotate is not None else ""
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{color}' text-anchor='{anchor}' "
            f"font-weight='{weight}'{rot}>{esc(s)}</text>")


def nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    t, out = (lo // step) * step, []
    while t <= hi + 1e-9:
        if t >= lo - 1e-9:
            out.append(round(t, 6))
        t += step
    return out


class Panel:
    """One plot area inside an SVG: data -> pixel transforms plus axis drawing."""

    def __init__(self, x0, y0, w, h, xlim, ylim):
        self.x0, self.y0, self.w, self.h, self.xlim, self.ylim = x0, y0, w, h, xlim, ylim

    def x(self, v):
        return self.x0 + (v - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w

    def y(self, v):
        return self.y0 + self.h - (v - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h

    def axes(self, title, xlabel, ylabel, xticks, yticks, yfmt="{:.1f}", xfmt="{:g}") -> List[str]:
        s = [text(self.x0, self.y0 - 10, title, 12, INK, weight="bold")]
        for t in yticks:
            s.append(f"<line x1='{self.x0}' x2='{self.x0 + self.w}' y1='{self.y(t):.1f}' y2='{self.y(t):.1f}' "
                     f"stroke='{GRID}' stroke-width='1'/>")
            s.append(text(self.x0 - 6, self.y(t) + 4, yfmt.format(t), 10, anchor="end"))
        s.append(f"<line x1='{self.x0}' x2='{self.x0 + self.w}' y1='{self.y0 + self.h}' y2='{self.y0 + self.h}' "
                 f"stroke='{INK2}' stroke-width='1'/>")
        for t in xticks:
            s.append(text(self.x(t), self.y0 + self.h + 15, xfmt.format(t), 10, anchor="middle"))
        if xlabel:
            s.append(text(self.x0 + self.w / 2, self.y0 + self.h + 32, xlabel, 11, anchor="middle"))
        if ylabel:
            s.append(text(self.x0 - 38, self.y0 + self.h / 2, ylabel, 11, anchor="middle", rotate=-90))
        return s

    def step(self, ys: Sequence[float], color, width=2.0, opacity=1.0) -> str:
        pts = []
        for i, v in enumerate(ys):
            if i:
                pts.append(f"{self.x(i + 1):.1f},{self.y(ys[i - 1]):.1f}")
            pts.append(f"{self.x(i + 1):.1f},{self.y(v):.1f}")
        return (f"<polyline points='{' '.join(pts)}' fill='none' stroke='{color}' stroke-width='{width}' "
                f"stroke-linejoin='round' opacity='{opacity}'/>")

    def band(self, lo: Sequence[float], hi: Sequence[float], color, opacity=0.14) -> str:
        top, bot = [], []
        lo = [max(v, self.ylim[0]) for v in lo]  # keep the band inside the plot area
        hi = [min(v, self.ylim[1]) for v in hi]
        for i in range(len(lo)):
            for k in ((i, i + 1) if i + 1 < len(lo) else (i,)):
                top.append(f"{self.x(k + 1):.1f},{self.y(hi[i]):.1f}")
                bot.append(f"{self.x(k + 1):.1f},{self.y(lo[i]):.1f}")
        return (f"<polygon points='{' '.join(top + bot[::-1])}' fill='{color}' fill-opacity='{opacity}' "
                f"stroke='none'/>")


def svg(w, h, body: List[str]) -> str:
    return "\n".join([f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='100%' {FONT}>",
                      f"<rect width='{w}' height='{h}' fill='{SURF}'/>"] + body + ["</svg>"])


def legend(x, y, items: List[Tuple[str, str]], cols: int, colw: float) -> List[str]:
    s = []
    for i, (label, color) in enumerate(items):
        cx, cy = x + (i % cols) * colw, y + (i // cols) * 18
        s.append(f"<line x1='{cx}' x2='{cx + 18}' y1='{cy}' y2='{cy}' stroke='{color}' stroke-width='3'/>")
        s.append(text(cx + 24, cy + 4, label, 11, INK))
    return s


def series(curves, task, key, variant):
    g = curves[task].get(f"{key}:{variant}")
    if not g:
        return None
    mean = [float(v) for v in g[1].split()]
    sd = [float(v) for v in g[2].split()]
    return mean, sd


def fig_search(curves) -> str:
    pw, ph, gx, gy, left, top = 330, 210, 90, 92, 70, 92
    body = legend(left, 24, [(lab, c) for _, lab, c in METHODS] + [("LLMatic / LAPT / EvoPrompting", GREY)], 4, 200)
    for i, (task, var, title) in enumerate(TASKS):
        rounds = 30 if task == "classification" else 50
        allv = [m + s for k in [x[0] for x in METHODS] + OTHERS if (ms := series(curves, task, k, var))
                for m, s in zip(*ms)]
        ymax = max(allv) * 1.05
        p = Panel(left + (i % 2) * (pw + gx), top + (i // 2) * (ph + gy), pw, ph, (1, rounds), (0, ymax))
        body += p.axes(title, "Search round" if i >= 2 else "",
                       "Best improvement (points)" if i % 2 == 0 else "",
                       [1] + list(range(10, rounds + 1, 10)), nice_ticks(0, ymax, 4), "{:g}")
        for k in OTHERS:
            if (ms := series(curves, task, k, var)):
                body.append(p.step(ms[0], GREY, 1.4))
        for k, _, c in reversed(METHODS):
            if (ms := series(curves, task, k, var)):
                m, s = ms
                if k in ("agent_reflect", "agent"):
                    body.append(p.band([a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)], c))
                body.append(p.step(m, c, 2.6 if k == "agent_reflect" else 1.8))
    body.append(text(left, top + 2 * ph + gy + 46, "Mean over seeds (3 in the module space, 5 on NAS-Bench-201); "
                     "shaded: +-1 sd, for Ours and Agent only.", 10))
    return svg(left + 2 * pw + gx + 30, top + 2 * ph + gy + 60, body)


def fig_transfer(curves) -> str:
    items = [("agent_reflect", "transfer-t50", "Ours + principles transferred from CIFAR-10", SLOTS[0]),
             ("agent_reflect", "t50", "Ours, from scratch", SLOTS[1]),
             ("agent_reflect", "randinit-t50", "Ours + random initial principles", SLOTS[2]),
             ("agent", "t50", "Agent only (no library)", SLOTS[3])]
    pw, ph, gx, left, top, rounds = 330, 220, 90, 70, 84, 30
    body = legend(left, 22, [(lab, c) for *_, lab, c in items], 2, 330)
    for i, (task, title) in enumerate([("nb201_cifar100", "(a) NAS-Bench-201, CIFAR-100"),
                                       ("nb201_imagenet16", "(b) NAS-Bench-201, ImageNet16-120")]):
        p = Panel(left + i * (pw + gx), top, pw, ph, (1, rounds), (0, 3.2))
        body += p.axes(title, "Search round", "Best improvement (points)" if i == 0 else "",
                       [1, 5, 10, 15, 20, 25, 30], nice_ticks(0, 3.2, 4))
        for key, var, _, c in reversed(items):
            m, s = series(curves, task, key, var)
            m, s = m[:rounds], s[:rounds]
            if key == "agent_reflect":
                body.append(p.band([a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)], c, 0.12))
            body.append(p.step(m, c, 2.6 if var.startswith("transfer") else 1.8))
    return svg(left + 2 * pw + gx + 30, top + ph + 60, body)


def fig_calibration(res) -> str:
    cal = {k: v for k, v in res["calibration"].items() if not k.startswith("_")}
    left, top, w, h = 70, 70, 340, 300
    p = Panel(left, top, w, h, (0.25, 1.0), (0.3, 1.0))
    body = p.axes("Calibration of principle confidence", "Confidence of the cited principle (bin)",
                  "Direction predicted correctly", [0.25, 0.5, 0.75, 1.0], [0.4, 0.6, 0.8, 1.0], "{:.0%}", "{:.2f}")
    body.append(f"<line x1='{p.x(0.3):.1f}' y1='{p.y(0.3):.1f}' x2='{p.x(1):.1f}' y2='{p.y(1):.1f}' "
                f"stroke='{INK2}' stroke-dasharray='4 4'/>")
    body.append(text(p.x(0.40), p.y(0.37), "perfect calibration", 10, rotate=-41))
    centers = [0.375, 0.625, 0.875]
    for i, (name, bins) in enumerate(cal.items()):
        c = SLOTS[i]
        pts = " ".join(f"{p.x(cx):.1f},{p.y(a):.1f}" for cx, (_, a) in zip(centers, bins))
        body.append(f"<polyline points='{pts}' fill='none' stroke='{c}' stroke-width='2'/>")
        for cx, (n, a) in zip(centers, bins):
            r = 3 + 2.2 * (n ** 0.5) / 4
            body.append(f"<circle cx='{p.x(cx):.1f}' cy='{p.y(a):.1f}' r='{r:.1f}' fill='{c}' stroke='{SURF}' "
                        f"stroke-width='2'/>")
    body += legend(left + w + 30, top + 10, [(k, SLOTS[i]) for i, k in enumerate(cal)], 1, 0)
    body.append(text(left + w + 30, top + 80, "Marker area ~ number of checkable", 10))
    body.append(text(left + w + 30, top + 94, "citations (15-152 per bin).", 10))
    return svg(left + w + 280, top + h + 50, body)


def fig_audit(res) -> str:
    aud = {k: v for k, v in res["audit"].items() if isinstance(v, dict)}
    names = list(aud)
    short = ["Module", "NB201\nC-10", "NB201\nC-100", "NB201\nIN16"]
    pw, ph, gx, left, top = 230, 220, 70, 70, 76
    body = legend(left, 24, [("Where the search was (local)", SLOTS[0]), ("Whole space", SLOTS[1])], 2, 240)
    specs = [("(a) False confirmation", "fc"), ("(b) False refutation", "fr")]
    for i, (title, key) in enumerate(specs):
        p = Panel(left + i * (pw + gx), top, pw, ph, (0, len(names)), (0, 0.4))
        body += p.axes(title, "", "Error rate" if i == 0 else "", [], [0, 0.1, 0.2, 0.3, 0.4], "{:.0%}")
        for j, nm in enumerate(names):
            for b, (sfx, c) in enumerate((("local", SLOTS[0]), ("global", SLOTS[1]))):
                e, n = aud[nm][f"{key}_{sfx}"]
                v = e / n if n else 0
                bx = p.x(j + 0.18 + b * 0.32)
                body.append(f"<rect x='{bx:.1f}' y='{p.y(v):.1f}' width='{p.w / len(names) * 0.3:.1f}' "
                            f"height='{p.y(0) - p.y(v):.1f}' fill='{c}' rx='2'/>")
                body.append(text(bx + p.w / len(names) * 0.15, p.y(v) - 4, f"{v:.0%}", 9, anchor="middle"))
            for k, line in enumerate(short[j].split("\n")):
                body.append(text(p.x(j + 0.5), p.y(0) + 15 + 12 * k, line, 10, anchor="middle"))
    p = Panel(left + 2 * (pw + gx), top, pw, ph, (0, len(names)), (0, 0.3))
    body += p.axes("(c) Brier score (local truth)", "", "", [], [0, 0.1, 0.2, 0.3], "{:.2f}")
    for j, nm in enumerate(names):
        v = aud[nm]["brier_local"]
        bx = p.x(j + 0.3)
        body.append(f"<rect x='{bx:.1f}' y='{p.y(v):.1f}' width='{p.w / len(names) * 0.4:.1f}' "
                    f"height='{p.y(0) - p.y(v):.1f}' fill='{SLOTS[0]}' rx='2'/>")
        body.append(text(bx + p.w / len(names) * 0.2, p.y(v) - 4, f"{v:.2f}", 9, anchor="middle"))
        for k, line in enumerate(short[j].split("\n")):
            body.append(text(p.x(j + 0.5), p.y(0) + 15 + 12 * k, line, 10, anchor="middle"))
    ref = res["audit"]["no_update_or_random_brier"]
    body.append(f"<line x1='{p.x0}' x2='{p.x0 + p.w}' y1='{p.y(ref):.1f}' y2='{p.y(ref):.1f}' stroke='{INK2}' "
                f"stroke-dasharray='4 3'/>")
    body.append(text(p.x0 + p.w, p.y(ref) - 5, "confidence frozen / random principles", 9, anchor="end"))
    body.append(text(left, top + ph + 52, "Verdicts of the full method checked against the complete benchmark "
                     "(matched pairs). Local: pairs anchored at the architectures the search evaluated.", 10))
    return svg(left + 3 * pw + 2 * gx + 30, top + ph + 66, body)


def fig_ablation(res) -> str:
    rows = [(k, v) for k, v in res["ablation_cifar100_auc"].items() if not k.startswith("_")]
    rows.sort(key=lambda kv: kv[1][0])
    left, top, w, rh = 290, 50, 360, 26
    h = rh * len(rows)
    p = Panel(left, top, w, h, (0, 3.0), (0, len(rows)))
    body = [text(left - 280, 24, "NAS-Bench-201 CIFAR-100: area under the best-so-far curve (50 rounds, 5 seeds)",
                 12, INK, weight="bold")]
    for t in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        body.append(f"<line x1='{p.x(t):.1f}' x2='{p.x(t):.1f}' y1='{top}' y2='{top + h}' stroke='{GRID}'/>")
        body.append(text(p.x(t), top + h + 15, f"{t:g}", 10, anchor="middle"))
    body.append(text(left + w / 2, top + h + 32, "AUC (mean best improvement, points); bars: mean, whiskers: +-1 sd",
                     11, anchor="middle"))
    for i, (name, (m, sd)) in enumerate(rows):
        yc = top + h - (i + 0.5) * rh
        ours = name in ("Full method (v4)", "Full method + transferred principles")
        c = SLOTS[0] if ours else GREY
        body.append(f"<rect x='{left}' y='{yc - 8:.1f}' width='{p.x(m) - left:.1f}' height='16' fill='{c}' rx='2'/>")
        body.append(f"<line x1='{p.x(m - sd):.1f}' x2='{p.x(m + sd):.1f}' y1='{yc:.1f}' y2='{yc:.1f}' "
                    f"stroke='{INK}' stroke-width='1.2'/>")
        body.append(text(left - 8, yc + 4, name, 11, INK if ours else INK2, anchor="end"))
        body.append(text(p.x(m + sd) + 6, yc + 4, f"{m:.2f}", 10, anchor="start"))
    return svg(left + w + 60, top + h + 46, body)


KIND = {  # box kind -> (stroke, fill); the legend explains the encoding
    "llm": (SLOTS[0], "#eaf2fc"), "stats": (SLOTS[2], "#e7f6f0"),
    "memory": (SLOTS[3], "#fdf4de"), "env": (INK2, "#f3f3f1")}


def box(x, y, w, h, kind, title, lines, dashed=False) -> List[str]:
    stroke, fill = KIND[kind]
    dash = " stroke-dasharray='5 4'" if dashed else ""
    s = [f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{fill}' stroke='{stroke}' "
         f"stroke-width='1.6'{dash}/>", text(x + w / 2, y + 22, title, 13, INK, "middle", "bold")]
    s += [text(x + w / 2, y + 42 + 16 * i, ln, 11, INK2, "middle") for i, ln in enumerate(lines)]
    return s


def arrow(points: List[Tuple[float, float]], label="", lx=0.0, ly=0.0, anchor="middle", dashed=False,
          rotate=None) -> List[str]:
    pts = " ".join(f"{x},{y}" for x, y in points)
    dash = " stroke-dasharray='5 4'" if dashed else ""
    s = [f"<polyline points='{pts}' fill='none' stroke='{INK2}' stroke-width='1.5'{dash} marker-end='url(#arr)'/>"]
    if label:
        for i, ln in enumerate(label.split("\n")):
            s.append(text(lx, ly + 13 * i, ln, 10.5, INK, anchor, rotate=rotate))
    return s


def fig_framework() -> str:
    """One search round of the hypothesis-guided framework: LLMs propose, statistics decide."""
    W, H = 1010, 600
    body = ["<defs><marker id='arr' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
            f"orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{INK2}'/></marker></defs>",
            text(30, 28, "One search round: the LLMs propose, the statistics decide", 14, INK, weight="bold")]
    # boxes (grid: rows at y = 70 and 390, columns at x = 40 / 320 / 600 / 840)
    body += box(40, 190, 210, 150, "memory", "Principle library",
                ["if-then claim + scope", "confidence (computed)", "support / counterexamples",
                 "confirmed > 0.75, refuted < 0.25"])
    body += box(320, 70, 210, 112, "llm", "Design Agent (LLM)",
                ["proposes 4 candidates:", "parent + one or two edits,", "expected gain, cited principles"])
    body += box(600, 70, 210, 112, "stats", "Hypothesis-guided selection",
                ["expected gain (EI, LLM prior)", "+ principle uncertainty 4c(1-c)", "+ verification value"])
    body += box(860, 70, 120, 112, "env", "Evaluation", ["train the child", "(or read the", "score table)"])
    body += box(600, 390, 210, 112, "stats", "Evidence model",
                ["Bayesian linear model:", "main effects + claimed", "interactions, known noise"])
    body += box(320, 390, 210, 112, "llm", "Reflector (LLM)",
                ["conjectures conditional,", "falsifiable principles;", "never judges evidence"])
    body += box(40, 440, 150, 70, "env", "Audit (offline)", ["complete benchmark"], dashed=True)
    # data flow around the loop
    body += arrow([(250, 230), (285, 230), (285, 126), (320, 126)], "principles,\nevidence table", 290, 196, "start")
    body += arrow([(530, 126), (600, 126)], "4 candidates", 565, 116)
    body += arrow([(810, 126), (860, 126)], "1 child", 835, 116)
    body += arrow([(920, 182), (920, 446), (810, 446)], "score,\nchange vs parent", 928, 300, "start")
    body += arrow([(705, 390), (705, 182)], "posterior:\ngain, verification", 713, 280, "start")
    body += arrow([(600, 446), (530, 446)], "effects,\ncounter-\nexamples", 565, 464)
    body += arrow([(320, 446), (285, 446), (285, 310), (250, 310)], "new or\nnarrowed\nclaims", 290, 360, "start")
    body += arrow([(705, 502), (705, 560), (215, 560), (215, 340)], "confidence of every claim = P(claim holds | all trials)",
                  460, 552)
    # outside the loop (dashed): transfer in, audit out
    body += arrow([(145, 140), (145, 190)], dashed=True)
    body.append(text(145, 130, "principles learned on another dataset", 10.5, INK, "middle"))
    body.append(text(145, 116, "transfer (optional):", 10.5, INK2, "middle"))
    body += arrow([(115, 340), (115, 440)], "verdicts", 121, 395, "start", dashed=True)
    # legend
    lx, ly = 330, 560 + 26
    for i, (kind, lab) in enumerate((("llm", "LLM: proposes"), ("stats", "statistics: decides"),
                                     ("memory", "shared state"), ("env", "training / benchmark"))):
        stroke, fill = KIND[kind]
        x = lx + i * 165
        body.append(f"<rect x='{x}' y='{ly - 10}' width='16' height='12' rx='3' fill='{fill}' stroke='{stroke}' "
                    f"stroke-width='1.4'/>")
        body.append(text(x + 22, ly, lab, 11, INK2))
    return svg(W, H + 20, body)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    curves: Dict = json.load(open(os.path.join(a.data, "curves.json"), encoding="utf-8"))
    res: Dict = json.load(open(os.path.join(a.data, "results.json"), encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    figs = {"fig0_framework": fig_framework(), "fig1_search_curves": fig_search(curves), "fig2_transfer": fig_transfer(curves),
            "fig3_calibration": fig_calibration(res), "fig4_audit": fig_audit(res), "fig5_ablation": fig_ablation(res)}
    for name, s in figs.items():
        with open(os.path.join(a.out, name + ".svg"), "w", encoding="utf-8") as f:
            f.write(s)
    print("wrote " + ", ".join(f"{n}.svg" for n in figs))


if __name__ == "__main__":
    main()
