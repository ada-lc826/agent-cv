# VisionEvoAgent

**A Lightweight Agent-Driven Framework for Autonomous Vision Architecture Exploration**

VisionEvoAgent improves an established vision model, such as the official MMPretrain ResNet-50 on CIFAR-100, by inserting plug-in modules chosen from a fixed search space. Each round, an LLM agent proposes one candidate. The candidate is trained under a proxy schedule, and the result is written back to the agent's memory.

| Innovation | Component | Switch |
|---|---|---|
| 1. Agent-Guided Module Exploration | `DesignAgent`: reads the full history table, the current best and the failed configurations, plus a **cost table** with the parameter change of every legal untried move from the current best, then proposes the next candidate. Includes an explore/exploit schedule, a mutation budget that widens when the search stagnates, rule checks with retries and a legal fallback. | `--mode agent` |
| 2. Reflective Architecture Evolution | `ReflectorAgent`: every N rounds it reads the trials since its last reflection and reports, for each existing design principle, supporting / contradicting / neutral evidence with a strength; the principle's confidence moves up or down accordingly (confirmed > 0.75, refuted < 0.25, refuted ones are no longer shown to the Design Agent). New principles must be falsifiable (IF … IN … THEN … BECAUSE … DISPROVED IF …). The Design Agent cites the principles it relies on, which makes the confidences checkable (`tools/principle_calibration.py`). Principles can be transferred across tasks. | `--mode agent_reflect` |
| Baselines | Random search plus the 7 comparison methods of the research plan, re-implemented under our search space (see "Comparison methods"). | `--mode random\|reg_evo\|tpe\|genius\|evoprompting\|llmatic\|lapt\|nader` |

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
  prompts.py             VisionEvoAgent prompts (can be quoted verbatim in the paper appendix)
  baseline_prompts.py    prompts of the re-implemented LLM comparison methods
  agent_client.py        Claude API (official SDK, structured JSON output) or mock; every call logged
  agents/                DesignAgent (innovation 1), ReflectorAgent (innovation 2)
  controllers/           AgentSearch (ours) + random, reg_evo, tpe, genius, evoprompting, llmatic, lapt, nader
  runners/               MMPretrainRunner (real training), MockRunner (dry runs only)
  plugins/               module implementations + MMPretrain registration (EvoResNet_CIFAR, fusion neck)
  search.py              main loop: baseline → propose → train → record → reflect, with resume
  report.py              search_history.csv / search_summary.json / evolution.json
run_search.py            command-line entry point
tools/measure_noise.py   run the baseline with several seeds to measure the proxy noise
tools/compare_runs.py    method-comparison table over finished runs (mean ± std over seeds)
tools/principle_calibration.py   hit rate of cited principles per confidence bin (are confidences meaningful?)
configs/                 cls_resnet50_cifar100.yaml (real runs), smoke_cpu.yaml (plumbing test)
```

## Usage

```bash
pip install -r requirements.txt     # plus the OpenMMLab environment used for the reproduction
export PYTHONPATH=$PWD
```

1. Dry run, with no GPU and no API key. The mock LLM and mock scores are for testing the pipeline only; never put them in the paper:
   ```bash
   python run_search.py --mode agent_reflect --dry_run
   ```
2. Preconditions, before any search:
   - Measure the proxy noise, then write the std into `noise_level` / `min_improvement` in the yaml:
     ```bash
     python tools/measure_noise.py --config configs/cls_resnet50_cifar100.yaml --seeds 0 1 2
     ```
   - Check that the proxy ranks like full training: train 16 architectures (baseline included) with both recipes, then analyse. Tau with a bootstrap CI, and whether the proxy's top picks keep their lead; lengthen the proxy if it fails:
     ```bash
     python tools/proxy_reliability.py sample --n 16 --seed 0
     python tools/proxy_reliability.py analyze --csv runs/proxy_check/scores.csv --top_k 3
     ```
   - Check that signature de-duplication matches the real compute graph (CPU only; done 2026-10-13: 90 combinations → 86 distinct networks, dedup exact, 85 candidates besides the baseline):
     ```bash
     python tools/check_equivalence.py --config configs/cls_resnet50_cifar100.yaml
     ```
3. Search. Pilot with the yaml's `max_rounds: 10`; the formal classification comparison adds `--max_rounds 30`. Seeds 0/1/2:
   ```bash
   export DEEPSEEK_API_KEY=...         # set it yourself in the shell (e.g. ~/.bashrc); never commit it
   # (with --provider anthropic: export ANTHROPIC_API_KEY=... instead)
   for m in random reg_evo tpe genius evoprompting llmatic lapt nader agent agent_reflect; do
     python run_search.py --config configs/cls_resnet50_cifar100.yaml --mode $m --seed 0
   done
   ```
   If a run is interrupted, run the same command again: it resumes from the last finished round (controller state such as LLMatic's archives or NADER's experience store is restored too). If a trial finished but was not yet recorded, its `result.json` is read directly and the trial is not retrained.
4. Ablations (each gets its own run directory, e.g. `runs/classification_agent_reflect_rule_s0`):

   | Research plan | Flag |
   |---|---|
   | D-1 no failure memory | `--no_failure_memory` |
   | D-2 no principle memory | `--no_principle_memory` |
   | D-3 no mutation budget | `--mutation_budget 4` |
   | D-4 no explore/exploit schedule | `--explore_every 0` |
   | D-5 reflection frequency | `--reflect_every_n 1` / `2` / `4` |
   | D-6 rule-based "reflection" (LLMENAS trigger: no improvement → change direction, improvement → refine) | `--reflection_style rule` |
   | D-7 immediate reflection without memory (RZ-NAS style) | `--reflection_style immediate` |
   | Design Agent without the cost table | `--no_param_costs` |
   | D-8 fixed principles: generic ones from the module papers (`configs/principles_fixed_cls.json`), never updated | `--principle_mode fixed` |
   | D-9 random principles in the same format, never updated | `--principle_mode random` |
   | D-10 principles induced as usual, confidence never updated | `--principle_mode no_update` |
   | Principle transfer | `--init_principles runs/<other run>/principle_memory.json` (also works for `--mode lapt`) |
   | Transfer control: randomly initialised principles, updated as usual | `--init_principles random` |
   | Formal budget | `--max_rounds 30` |
   | Thinking-mode control (main runs have thinking off) | `--thinking on` |
   | NADER with its original 90%-of-parent trigger | `--mode nader --nader_threshold orig` |
5. Comparison table:
   ```bash
   python tools/compare_runs.py --runs runs --task classification --out runs/comparison
   ```
6. Methods for segmentation, by the rule fixed on 2026-10-13 (top 2 of the five LLM comparison methods by mean best score over seeds 0/1/2 in the 30-trial classification comparison; refuses to select until all 15 runs are finished; Random, Reg-Evo and ours always run):
   ```bash
   python tools/select_seg_methods.py --runs runs --out runs/seg_method_selection.json
   ```

### Outputs (`runs/<task>_<mode>[_<variant>]_s<seed>/`)

| File | Contents |
|---|---|
| `search_history.csv` | One row per round: architecture, score, params, Δ vs baseline / Δ vs previous best, param ratio, is_best, explore/exploit, failure reason, duration, proposer note |
| `search_summary.json` | Best result and its Δparams, rounds to best, invalid-trial rate and failure reasons, fallback proposals, GPU hours, LLM calls and tokens, final principles, stop reason |
| `evolution.json` | Best-so-far curve, for the search-efficiency plot |
| `trial_memory.json` / `failure_memory.json` / `principle_memory.json` / `reflections.json` | Agent memory (used for resume and transfer). Each trial stores the principles it cited with their confidence at that moment; each principle stores its full evidence log (round, type, strength, before → after) |
| `controller_state.json` | State of the comparison method (archives, learned subspace, experiences, ...) |
| `llm_calls.jsonl` | Every LLM call: purpose, effort, tokens, latency, served_model, request_id, errors |
| `trials/r###_<signature>/` | Each candidate's MMPretrain config, training log and result.json (checkpoints are deleted by default) |

`tools/compare_runs.py` adds, per method, the sample-efficiency metric of Li & Talwalkar (UAI 2019): how many random-search trials are needed to reach the score the method reached, and that number divided by the method's rounds to best (">" when some random runs never got there).

## Comparison methods

All are re-implemented under our search space (the official code of every LLM method is tied to its own search space), share the runner, budget and seeds, and return the same JSON as the Design Agent. **Re-ask rule, identical for every LLM method including ours:** an unusable answer (illegal value, or an architecture that was already evaluated) is re-asked with the reason as feedback, at most twice (three answers in total); after that the round takes a random untried architecture (for our Design Agent: a random untried legal local mutation). Every answer's outcome is stored with the trial (`proposal_attempts`), and `search_summary.json` reports `duplicate_rate` (answers naming an evaluated architecture / all answers) and `fallback_rate` (rounds decided by the fallback / all rounds). EvoPrompting, LLMatic and NADER do not list every evaluated architecture in their original prompts, so they repeat themselves more often; this is reported, not corrected. Settings live under `baselines:` in the yaml. Deviations to report in the paper:

| Method | Original | Here | Deviation |
|---|---|---|---|
| Random (Li & Talwalkar, UAI 2019) | uniform sampling | without replacement | none |
| Reg-Evo (Real et al., AAAI 2019) | P=100, S=25, 20k models | P=4, S=2: 4 random, then tournament + 1-dimension mutation, oldest removed | population scaled to the budget |
| TPE (Bergstra et al., NeurIPS 2011) | 200 trials, first 30 random | Optuna TPESampler, categorical, `n_startup_trials=3` | Optuna's default of 10 startup trials would be the whole budget |
| GENIUS (Zheng et al., 2023) | one GPT-4 conversation, temperature 0, 10 iterations | one conversation, structured JSON, rebuilt from memory each round; temperature 0 | temperature has no effect in the `--thinking on` control; generic expert persona; no human transcription step |
| EvoPrompting (Chen et al., NeurIPS 2023) | 10 rounds × 160 children, top-10 parents, 2-shot, prompt-tuning, mixed temperature | 2 generations × 5 children, top-4, 2-shot, target metrics (90% params / 102% accuracy), fitness −params × error | no soft prompt-tuning (API model); temperature mixing has no effect in the `--thinking on` control |
| LLMatic (Nasir et al., GECCO 2024) | network archive 100 CVT niches (width/depth, FLOPs); prompt archive (16 prompts × temperature); 2000 evaluations | network archive 3×3 (Δparams bin × target_stage); prompt archive (13 instructions × temperature 0.3/0.6/0.9 when temperature applies, else × effort low/medium/high); mutation 0.7 / crossover 0.3; curiosity +1 / −0.5 / −1 | archives scaled down; effort replaces temperature in thinking mode; instructions re-worded for our space; code-vs-paper differences resolved in favour of the paper |
| LAPT (Zhou et al., AAAI 2025) | principles from 50 top architectures of a previous task, ≤2 options per layer, REA, adaptation per iteration | 3 random trials (or `--init_principles` for the transfer setting), principles → ≤2 options per dimension, REA (P=2, S=2) inside, adaptation every 2 trials with the top-2 | fewer samples to learn from |
| NADER (Yang et al., CVPR 2025) | Reader + Proposer + Modifier + Reflector on computation graphs; experience store built offline, OpenAI embeddings | no Reader (LLM proposes 10 inspirations); Proposer picks (parent, inspiration) from the tree; Modifier applies it with top-5 retrieved experiences; failures reflected every 2 trials | config space instead of graphs (no development reflector); drop threshold = noise level instead of 90% of the parent (`--nader_threshold orig` runs the original rule; report both); bag-of-words retrieval |

All of these are adaptations to our search space, not faithful reproductions; the paper must say so and list every change.

## Design decisions

- **Fairness.** All methods share the same search space, the same runner, the same trial budget and the same seeds. Early stopping is off by default (`early_stop_patience: 0`).
- **Budget.** `max_rounds: 10` for the pilot only. The formal classification comparison uses `--max_rounds 30`, because population-based methods (Reg-Evo, LLMatic) cannot work properly with 10. Equal trial counts are not the whole story: report GPU hours and LLM calls / tokens (`search_summary.json`) and every change made to the original methods.
- **Noise.** A new best must beat the previous one by at least `min_improvement`, which should be ≥ the proxy seed std. The noise level is also passed to both agents, so they don't chase noise.
- **Cost awareness.** The Design Agent sees the parameter change of every candidate move (computed once by building all 86 models on CPU, cached in `param_table.json`). Results should always be reported as accuracy *and* Δparams.
- **Failure protection.**
  - OOM, NaN, timeouts, crashes and score drops below −noise go into the failure memory, and those configurations are never proposed again.
  - A malformed, illegal or repeated LLM output is re-asked with feedback, at most twice. If all three answers fail, the agent falls back to a random untried legal local mutation.
  - A failed reflection does not stop the search.
  - A wrong API key or invalid request (`AgentClientFatal`) stops the search immediately, so GPU time is not spent on the fallback.
  - Five consecutive failures stop the search.
- **LLM.** DeepSeek V4.1 Flash (`deepseek-flash`, OpenAI-compatible API, key in `DEEPSEEK_API_KEY`), **thinking off for the main comparison** (yaml `thinking: false`), so the baselines' temperatures (GENIUS 0, EvoPrompting mixed, LLMatic levels) take effect as in the papers; `--thinking on` is a separate control (run directories tagged `think`). DeepSeek's JSON mode guarantees valid JSON but not the schema, so the schema is written into the system prompt and every answer is validated (invalid → one retry with feedback → legal fallback, counted in `n_fallback_proposals`). Responses are requested as gzip/deflate only (old `brotlipy` in conda base breaks the HTTP client's brotli decoder). All methods in one comparison must use the same model version and LLM setting. `--provider anthropic` switches to `claude-opus-5-5` (schema enforced server-side, refusal fallback on, no temperature). Every call is logged with tokens (incl. reasoning tokens), latency and the model that answered.
- **Principle confidence.** Update rule after HypoExplore (Koo et al. 2026): supports c ← c + 0.2·s·(1−c), contradicts c ← c − 0.2·s·c, new principles start at 0.5, confirmed > 0.75, refuted < 0.25. η stays at 0.2 (decision 2026-10-13); if confidences still barely move after the 30-trial runs, η gets a sensitivity experiment, but it is not tuned to spread the confidences. Only trials since the previous reflection count as new evidence, and each principle gets at most one update per reflection, so no trial is counted twice.

## Status

- [x] Innovations 1 and 2, classification runner, resume, reporting
- [x] Cost table for the Design Agent; ablations D-1 … D-10
- [x] All 8 comparison methods (Random, Reg-Evo, TPE, GENIUS, EvoPrompting, LLMatic, LAPT, NADER)
- [x] `tools/compare_runs.py` (mean ± std, invalid-trial rate, multiple of random search)
- [x] Tested: dry runs of all 10 modes and every ablation flag; an interrupted-and-resumed run equals an uninterrupted one for every mode; all 86 architectures build in MMPretrain; a real end-to-end CPU training smoke test (earlier commit)
- [x] DeepSeek V4.1 Flash provider (tested offline against a fake API: request shape, temperature handling, error mapping, key never logged); evidence-based principle confidence, falsifiable principles, citation logging, calibration tool
- [x] Real DeepSeek API with mock scores (AutoDL, 2026-10-13): agent_reflect (15 calls, all valid first time, 0 fallbacks, ~$0.08 with thinking on), agent, GENIUS, EvoPrompting, LLMatic, LAPT run end to end
- [x] Controls for innovation 2 (D-8 fixed / D-9 random / D-10 no-update principles, random-initialised transfer), NADER original-threshold run, proxy-reliability and equivalence tools; equivalence checked on the real MMPretrain build
- [x] Uniform re-ask rule with duplicate / fallback statistics (`tests/test_reask.py`); segmentation method selection tool
- [ ] Not yet tested: real training in a search; Claude provider against the real API
- [ ] Detection (MMDetection) and segmentation (MMSegmentation) runners
- [ ] Visualisation scripts (evolution curves, Grad-CAM heatmaps)
