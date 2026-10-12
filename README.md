# VisionEvoAgent

**A Lightweight Agent-Driven Framework for Autonomous Vision Architecture Exploration**

VisionEvoAgent improves an established vision model, such as the official MMPretrain ResNet-50 on CIFAR-100, by inserting plug-in modules chosen from a fixed search space. Each round, an LLM agent proposes one candidate. The candidate is trained under a proxy schedule, and the result is written back to the agent's memory.

| Innovation | Component | Switch |
|---|---|---|
| 1. Agent-Guided Module Exploration | `DesignAgent`: reads the full history table, the current best and the failed configurations, then proposes the next candidate. Includes an explore/exploit schedule, a mutation budget that widens when the search stagnates, rule checks with retries and a legal fallback. | `--mode agent` |
| 2. Reflective Architecture Evolution | `ReflectorAgent`: every N rounds it distils effective/ineffective patterns, synergies and design principles into `ArchitectureMemory` (trial/failure/principle memory). These feed the next design. Principles can be transferred across tasks. | `--mode agent_reflect` |
| Baseline | `RandomSearchController`: sampling without replacement. | `--mode random` |

## Search space (classification)

| Dimension | Options |
|---|---|
| attention | none, SE, ECA, CBAM, CoordAtt (inside each Bottleneck, after BN3, before the residual addition) |
| conv | standard, dilated (d=2), depthwise-separable (replaces the 3×3 conv2 of each Bottleneck) |
| fusion | none, weighted (learnable softmax weights over the GAP features of stage 3 and stage 4) |
| target_stage | early (stage 1-2), late (stage 3-4), all |

There are 86 distinct architectures after deduplicating equivalent ones. All 86 have been checked: they build in MMPretrain and run a forward pass. The baseline has 23.705M parameters, which matches the official ResNet-50 CIFAR model.

## Code layout

```
visionevoagent/
  search_space.py        search space, signatures, mutation, legality checks
  memory.py              ArchitectureMemory: trial / failure / principle / reflection, atomic JSON writes
  schemas.py             JSON schemas for the Design and Reflector outputs, plus validation
  prompts.py             all prompts (can be quoted verbatim in the paper appendix)
  agent_client.py        Claude API (official SDK, structured JSON output) or mock; every call logged
  agents/                DesignAgent (innovation 1), ReflectorAgent (innovation 2)
  controllers/           RandomSearch, AgentSearch (the other comparison methods go here)
  runners/               MMPretrainRunner (real training), MockRunner (dry runs only)
  plugins/               module implementations + MMPretrain registration (EvoResNet_CIFAR, fusion neck)
  search.py              main loop: baseline → propose → train → record → reflect, with resume
  report.py              search_history.csv / search_summary.json / evolution.json
run_search.py            command-line entry point
tools/measure_noise.py   run the baseline with several seeds to measure the proxy noise
configs/                 cls_resnet50_cifar100.yaml (real runs), smoke_cpu.yaml (plumbing test)
```

## Usage

```bash
pip install -r requirements.txt     # plus the OpenMMLab environment used for the reproduction
export PYTHONPATH=$PWD
```

1. Dry run, with no GPU and no API key. The mock LLM and mock scores are for testing the pipeline only; never put them in the paper:
   ```bash
   python run_search.py --mode agent_reflect --dry_run --max_rounds 12
   ```
2. Measure the proxy noise, then write the std into `noise_level` / `min_improvement` in the yaml:
   ```bash
   python tools/measure_noise.py --config configs/cls_resnet50_cifar100.yaml --seeds 0 1 2
   ```
3. Formal search. Use the same `max_rounds` for all three modes, and seeds 0/1/2:
   ```bash
   export ANTHROPIC_API_KEY=...        # set it yourself; never commit it to the repository
   python run_search.py --config configs/cls_resnet50_cifar100.yaml --mode random        --seed 0
   python run_search.py --config configs/cls_resnet50_cifar100.yaml --mode agent         --seed 0
   python run_search.py --config configs/cls_resnet50_cifar100.yaml --mode agent_reflect --seed 0
   ```
   If a run is interrupted, run the same command again: it resumes from the last finished round. If a trial finished but was not yet recorded, its `result.json` is read directly and the trial is not retrained.
4. Ablations and transfer:
   ```bash
   python run_search.py ... --mode agent_reflect --no_principle_memory
   python run_search.py ... --mode agent_reflect --no_failure_memory
   python run_search.py ... --mode agent_reflect --init_principles runs/<other run>/principle_memory.json
   ```

### Outputs (`runs/<task>_<mode>_s<seed>/`)

| File | Contents |
|---|---|
| `search_history.csv` | One row per round: architecture, score, params, Δ vs baseline / Δ vs previous best, param ratio, is_best, explore/exploit, failure reason, duration, agent rationale |
| `search_summary.json` | Best result, rounds to best, GPU hours, LLM calls and tokens, final principles, failure list, stop reason |
| `evolution.json` | Best-so-far curve, for the search-efficiency plot |
| `trial_memory.json` / `failure_memory.json` / `principle_memory.json` / `reflections.json` | Agent memory (used for resume and transfer) |
| `llm_calls.jsonl` | Every LLM call: tokens, latency, served_model, request_id, errors |
| `trials/r###_<signature>/` | Each candidate's MMPretrain config, training log and result.json (checkpoints are deleted by default) |

## Design decisions

- **Fairness.** All methods share the same search space, the same runner, the same trial budget and the same seeds. Early stopping is off by default (`early_stop_patience: 0`).
- **Noise.** A new best must beat the previous one by at least `min_improvement`, which should be ≥ the proxy seed std. The noise level is also passed to both agents, so they don't chase noise.
- **Failure protection.**
  - OOM, NaN, timeouts, crashes and score drops below −noise go into the failure memory, and those configurations are never proposed again.
  - A malformed or illegal LLM output is retried with feedback. If the retry also fails, the agent falls back to a legal local mutation.
  - A failed reflection does not stop the search.
  - A wrong API key or invalid request (`AgentClientFatal`) stops the search immediately, so GPU time is not spent on the fallback.
  - Five consecutive failures stop the search.
- **LLM.** Defaults to `claude-opus-5-5` with structured JSON output. Server-side refusal fallback is on by default: if the main model declines a request, the server retries it with a fallback model. The model that actually answered each call is logged as `served_model` in `llm_calls.jsonl`. For strict reproducibility, switch it off with `--no_fallbacks`.

## Status

- [x] Innovations 1 and 2, Random baseline, classification runner, resume, reporting
- [x] Tested: dry runs of all three modes, resume giving identical results, failure and space-exhaustion paths, all 86 architectures building in MMPretrain, a real end-to-end CPU training smoke test
- [ ] The other 7 comparison methods (RegEvo, TPE, GENIUS, EvoPrompting, LLMatic, LAPT, NADER) as `controllers/*`
- [ ] Detection (MMDetection) and segmentation (MMSegmentation) runners
- [ ] Visualisation scripts (evolution curves, Grad-CAM heatmaps)
