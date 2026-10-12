"""Command-line entry point.

Dry run (no GPU, no API key):
    python run_search.py --mode agent_reflect --dry_run --run_dir runs/dry_reflect
Real run:
    python run_search.py --config configs/cls_resnet50_cifar100.yaml --mode agent_reflect --seed 0
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from visionevoagent.agent_client import AgentClient, AgentClientConfig
from visionevoagent.agents import DesignAgent, ReflectorAgent
from visionevoagent.controllers import AgentSearchConfig, AgentSearchController, RandomSearchController
from visionevoagent.memory import ArchitectureMemory
from visionevoagent.runners import MockRunner
from visionevoagent.search import SearchConfig, SearchEngine

MODES = ("random", "agent", "agent_reflect")


def load_yaml(path: str) -> dict:
    if not path:
        return {}
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="", help="YAML with search/agent/runner sections")
    p.add_argument("--mode", choices=MODES, default="agent_reflect")
    p.add_argument("--task", default="classification")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_rounds", type=int)
    p.add_argument("--run_dir", default="")
    p.add_argument("--dry_run", action="store_true", help="mock runner + mock LLM; numbers are fake")
    p.add_argument("--provider", choices=("anthropic", "mock"))
    p.add_argument("--model")
    p.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"))
    p.add_argument("--no_fallbacks", action="store_true", help="disable server-side refusal fallbacks")
    p.add_argument("--init_principles", default="", help="principle_memory.json of another run (transfer)")
    p.add_argument("--no_failure_memory", action="store_true", help="ablation")
    p.add_argument("--no_principle_memory", action="store_true", help="ablation")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    y = load_yaml(a.config)
    ys, ya, yr = y.get("search", {}), y.get("agent", {}), y.get("runner", {})

    run_dir = a.run_dir or os.path.join("runs", f"{a.task}_{a.mode}_s{a.seed}{'_dry' if a.dry_run else ''}")
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(run_dir, "search.log"), encoding="utf-8")])

    scfg = SearchConfig(task=a.task, mode=a.mode, seed=a.seed, run_dir=run_dir,
                        **{k: v for k, v in ys.items() if k not in ("task", "mode", "seed", "run_dir")})
    if a.max_rounds is not None:
        scfg.max_rounds = a.max_rounds

    memory = ArchitectureMemory.load(run_dir)
    if a.init_principles and not memory.principles:
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
    if a.mode == "random":
        controller = RandomSearchController(a.task, a.seed)
    else:
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
        client = AgentClient(ccfg, log_path=os.path.join(run_dir, "llm_calls.jsonl"))
        acfg = AgentSearchConfig(**{**ya.get("search", {}), "use_reflection": a.mode == "agent_reflect"})
        if a.no_failure_memory:
            acfg.use_failure_memory = False
        if a.no_principle_memory:
            acfg.use_principle_memory = False
        reflector = ReflectorAgent(client) if acfg.use_reflection else None
        controller = AgentSearchController(a.task, a.seed, DesignAgent(client, seed=a.seed), reflector, acfg,
                                           metric=runner.metric, noise_level=scfg.noise_level)

    summary = SearchEngine(scfg, controller, runner, memory).run()
    print(f"\nstop: {summary['stop_reason']}  outputs: {os.path.abspath(run_dir)}")
    return 0 if summary.get("stop_reason") != "baseline_failed" else 1


if __name__ == "__main__":
    sys.exit(main())
