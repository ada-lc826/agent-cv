"""Command-line entry point.

Dry run (no GPU, no API key):
    python run_search.py --mode agent_reflect --dry_run
Real run:
    python run_search.py --config configs/cls_resnet50_cifar100.yaml --mode agent_reflect --seed 0
Comparison methods (same budget, runner and seeds):
    --mode random | reg_evo | tpe | genius | evoprompting | llmatic | lapt | nader
Ablations of VisionEvoAgent (each gets its own run directory automatically):
    --no_failure_memory  --no_principle_memory  --reflection_style rule|immediate  --no_param_costs
    --mutation_budget 4  --explore_every 0  --reflect_every_n 1|2|4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from visionevoagent.agent_client import AgentClient, AgentClientConfig
from visionevoagent.agents import DesignAgent, ReflectorAgent
from visionevoagent.controllers import (REFLECTION_STYLES, AgentSearchConfig, AgentSearchController,
                                        EvoPromptingController, GeniusController, LAPTController,
                                        LLMaticController, NaderController, RandomSearchController,
                                        RegularizedEvolutionController, TPEController)
from visionevoagent.memory import ArchitectureMemory
from visionevoagent.runners import MockRunner
from visionevoagent.search import SearchConfig, SearchEngine

AGENT_MODES = ("agent", "agent_reflect")
LLM_BASELINES = ("genius", "evoprompting", "llmatic", "lapt", "nader")
MODES = ("random", "reg_evo", "tpe") + LLM_BASELINES + AGENT_MODES


def load_yaml(path: str) -> dict:
    if not path:
        return {}
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="", help="YAML with search/agent/baselines/runner sections")
    p.add_argument("--mode", choices=MODES, default="agent_reflect")
    p.add_argument("--task", default="classification")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_rounds", type=int)
    p.add_argument("--run_dir", default="")
    p.add_argument("--dry_run", action="store_true", help="mock runner + mock LLM; numbers are fake")
    p.add_argument("--provider", choices=("deepseek", "anthropic", "mock"))
    p.add_argument("--thinking", choices=("on", "off"),
                   help="DeepSeek thinking mode (default on); off lets baseline temperatures take effect")
    p.add_argument("--model")
    p.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"))
    p.add_argument("--no_fallbacks", action="store_true", help="disable server-side refusal fallbacks")
    p.add_argument("--init_principles", default="",
                   help="principle_memory.json of another run (transfer; agent_reflect and lapt)")
    # ---- VisionEvoAgent ablations
    p.add_argument("--no_failure_memory", action="store_true", help="ablation D-1")
    p.add_argument("--no_principle_memory", action="store_true", help="ablation D-2")
    p.add_argument("--mutation_budget", type=int, help="ablation D-3 (e.g. 4 = unconstrained)")
    p.add_argument("--explore_every", type=int, help="ablation D-4 (0 = no explore rounds)")
    p.add_argument("--reflect_every_n", type=int, help="ablation D-5")
    p.add_argument("--reflection_style", choices=REFLECTION_STYLES,
                   help="ablations D-6 (rule) and D-7 (immediate); default full")
    p.add_argument("--no_param_costs", action="store_true", help="Design Agent without the cost table")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args(argv)


def variant_tag(a: argparse.Namespace) -> str:
    """Settings that change the method get their own run directory and run identity."""
    parts = []
    if a.no_failure_memory:
        parts.append("nofail")
    if a.no_principle_memory:
        parts.append("noprin")
    if a.mutation_budget is not None:
        parts.append(f"mb{a.mutation_budget}")
    if a.explore_every is not None:
        parts.append(f"ex{a.explore_every}")
    if a.reflect_every_n is not None:
        parts.append(f"rn{a.reflect_every_n}")
    if a.reflection_style and a.reflection_style != "full":
        parts.append(a.reflection_style)
    if a.no_param_costs:
        parts.append("nocost")
    if a.init_principles:
        parts.append("transfer")
    if a.max_rounds is not None:
        parts.append(f"t{a.max_rounds}")
    if a.thinking == "off":
        parts.append("nothink")
    return "-".join(parts)


def build_client(a: argparse.Namespace, ya: dict, run_dir: str) -> AgentClient:
    ccfg = AgentClientConfig(**ya.get("client", {}))
    if a.dry_run:
        ccfg.provider = "mock"
    if a.provider:
        ccfg.provider = a.provider
    if a.model:
        ccfg.model = a.model
    if a.effort:
        ccfg.effort = a.effort
    if a.no_fallbacks:
        ccfg.use_fallbacks = False
    if a.thinking:
        ccfg.thinking = a.thinking == "on"
    return AgentClient(ccfg, log_path=os.path.join(run_dir, "llm_calls.jsonl"))


def main(argv=None) -> int:
    a = parse_args(argv)
    y = load_yaml(a.config)
    ys, ya, yb, yr = y.get("search", {}), y.get("agent", {}), y.get("baselines", {}), y.get("runner", {})
    variant = variant_tag(a)

    run_dir = a.run_dir or os.path.join(
        "runs", f"{a.task}_{a.mode}{'_' + variant if variant else ''}_s{a.seed}{'_dry' if a.dry_run else ''}")
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(run_dir, "search.log"), encoding="utf-8")])

    scfg = SearchConfig(task=a.task, mode=a.mode, variant=variant, seed=a.seed, run_dir=run_dir,
                        **{k: v for k, v in ys.items() if k not in ("task", "mode", "seed", "run_dir", "variant")})
    if a.max_rounds is not None:
        scfg.max_rounds = a.max_rounds

    memory = ArchitectureMemory.load(run_dir)
    transferred = []
    if a.init_principles:
        with open(a.init_principles, encoding="utf-8") as f:
            transferred = [x["principle"] for x in json.load(f)]
        if a.mode == "agent_reflect" and not memory.principles:
            n = memory.load_principles_from(a.init_principles, a.task)
            logging.info("loaded %d transferred principles from %s", n, a.init_principles)

    # ---- runner ----
    if a.dry_run:
        runner = MockRunner(seed=a.seed, noise_std=scfg.noise_level * 0.75)
    else:
        from visionevoagent.runners.mmpretrain_runner import MMPretrainRunner, MMPretrainRunnerConfig
        rc = MMPretrainRunnerConfig(**{**yr, "seed": a.seed,
                                       "work_root": yr.get("work_root") or os.path.join(run_dir, "trials")})
        runner = MMPretrainRunner(rc)

    # ---- controller ----
    m = a.mode
    if m == "random":
        controller = RandomSearchController(a.task, a.seed)
    elif m == "reg_evo":
        controller = RegularizedEvolutionController(a.task, a.seed, **yb.get("reg_evo", {}))
    elif m == "tpe":
        controller = TPEController(a.task, a.seed, **yb.get("tpe", {}))
    elif m in LLM_BASELINES:
        client = build_client(a, ya, run_dir)
        kw = yb.get(m, {})
        if m == "genius":
            controller = GeniusController(a.task, a.seed, client, runner.metric, **kw)
        elif m == "evoprompting":
            controller = EvoPromptingController(a.task, a.seed, client, **kw)
        elif m == "llmatic":
            controller = LLMaticController(a.task, a.seed, client, **kw)
        elif m == "lapt":
            controller = LAPTController(a.task, a.seed, client, runner.metric, init_principles=transferred, **kw)
        else:
            controller = NaderController(a.task, a.seed, client, runner.metric, scfg.noise_level, **kw)
    else:
        client = build_client(a, ya, run_dir)
        acfg = AgentSearchConfig(**{**ya.get("search", {}), "use_reflection": m == "agent_reflect"})
        if a.no_failure_memory:
            acfg.use_failure_memory = False
        if a.no_principle_memory:
            acfg.use_principle_memory = False
        for k in ("mutation_budget", "explore_every", "reflect_every_n", "reflection_style"):
            if getattr(a, k) is not None:
                setattr(acfg, k, getattr(a, k))
        if a.mutation_budget is not None:
            acfg.stagnation_budget = max(acfg.stagnation_budget, a.mutation_budget)
        if a.no_param_costs:
            acfg.show_param_costs = False
        reflector = ReflectorAgent(client) if acfg.use_reflection and acfg.reflection_style != "rule" else None
        controller = AgentSearchController(a.task, a.seed, DesignAgent(client, seed=a.seed), reflector, acfg,
                                           metric=runner.metric, noise_level=scfg.noise_level,
                                           param_table_fn=lambda: runner.param_table(a.task))

    summary = SearchEngine(scfg, controller, runner, memory).run()
    print(f"\nstop: {summary['stop_reason']}  outputs: {os.path.abspath(run_dir)}")
    return 0 if summary.get("stop_reason") != "baseline_failed" else 1


if __name__ == "__main__":
    sys.exit(main())
