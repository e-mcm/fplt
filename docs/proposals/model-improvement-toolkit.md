# Model improvement toolkit

**Status:** proposal (2026-11-29). **Tracker:** #412 (issues #401–#411). **Builds on:** `fplt-experiments.md` (stages, patches, component runs, decisions), `weekly-lines-and-post-gw.md` (post-GW analysis, `forecast_scores`), `fplt-data-model.md` (runs, builds, fingerprints).

## 0. Why

The weekly lines and post-GW analysis tell us **how well** the models do and **where** they go wrong. Improving them needs a fast loop in between: **diagnose → hypothesise → change → test offline → decide**. The pieces we have are slow or scattered:
- `promote-eval` refits and replans whole models across seasons for every A/B;
- `component_eval`, `minutes_measurement` and `other_points_measurement` are separate tools with separate outputs;
- shadow lines take a season to confirm anything.

This toolkit makes the common case fast. Most ideas change **one component** (usually minutes), so they should be scored in minutes at that component, and pay for a full decision-level test only if they survive.

Out of scope: the agentic idea loop, the experiments dashboard and the approve flow. Those sit on top of this toolkit and are specified separately.

## 1. Where to look: diagnostics

### 1.1 Error attribution by component (oracle gain)

For each scored player-fixture, swap one component's prediction for its **realised value** and re-score. This is the oracle arm of `fplt-experiments.md` §4, applied as a diagnostic over the live ledger and backtests.

- Components: `minutes`, `scoreline` (team goals / clean sheet), `involvement` (goal/assist shares), `other` (bonus, cards, defcon, saves).
- **Oracle gain** of component c = score(model) − score(model with c set to its realised value), for RPS, log score and MSE of the mean.
- Because the components interact, report both the **one-at-a-time** gains and a **Shapley** split over the 4 components (2⁴ = 16 re-simulations per deadline; cached outputs make this cheap). The Shapley shares sum to the gap to the all-oracle score.
- Output: `error_attribution` table by season, GW, position and price band. Headline: "share of squared error / RPS attributable to each component, season to date, with block-bootstrap-by-GW CIs".
- Requires the component output cache (§2.2) and component outputs stored in predict runs (already true for official runs).

Use: it tells you which component to work on. A component with a small oracle gain can't pay back much effort, however wrong it looks.

### 1.2 Signal screen: residuals vs available covariates

Find information the model ignores:
- **Residuals:** per player-fixture `y − μ̂` (points), and per component (minutes residual, goal-involvement residual) from `forecast_scores` and the component tables.
- **Candidate covariates:** PIT-safe fields with `available_at ≤` the run's deadline (T−90m for official runs), drawn from two sources: (a) staged columns not consumed by any feature the model uses (`required_tables` vs feature `inputs`); (b) a small declared list of derived candidates (rest days, predicted-lineup confidence, odds-implied team goals, set-piece duty change, manager rotation index, minutes in the last 3 matches).
- **Statistics:** per covariate, the partial correlation with the residual controlling for μ̂ and position; and a binned mean residual with CIs. Standard errors are cluster-robust by fixture and GW. **Benjamini–Hochberg FDR** at 10% across covariates.
- Output: a ranked `signal_screen` table (covariate, component, effect size, CI, q-value, n). Shown in the UI, and the main input to idea generation.
- PIT guard: a covariate with any `available_at` after the cut-off for a row is excluded for that row. Tested.

## 2. Fast experiments

These implement `fplt-experiments.md` §9 steps 2–5, prioritised for component work.

### 2.1 Patch engine (experiments step 2)
Arms are config diffs of a base `PipelineSpec`: dotted paths (`model.components.minutes.params.logistic_blend: 0.8`), component substitution (`model.components.minutes: empirical_minutes_v3`), grids, and auto-labels. Validity check: the resolved diff equals the patch. Arm identity = `pipeline_hash` of the resolved spec.

### 2.2 Component output cache + component runs (experiments step 4)
- `run_type=component` runs. Cache key = (component spec hash, upstream component hashes, code_sha, lockfile_hash, build_id, deadline).
- **Composite hook:** the layered model accepts precomputed component outputs for a deadline instead of fitting them. A minutes arm reuses the control's cached scoreline, involvement and other outputs, refits only minutes, then re-simulates.
- Explicit dependency graph (involvement normalises within team; lineup multipliers read minutes), so cache keys include upstream hashes where needed.

### 2.3 Common measure interface (experiments step 3)
One `Measure` protocol: `(stage, runs, outcomes, slices) → metrics table` with standard keys. Adapters wrap the existing scorers, with no rewrites:
- `component_eval`, `minutes_measurement`, `other_points_measurement`;
- `forecast_scores` (post-GW #395);
- the decisions board and `simulated_season`.

Every experiment, screen and dashboard reads the same table shape.

### 2.4 Component experiment command
`fplt exp component --base <pipeline> --component minutes --arm <patch|spec> [--arm …] --window <season[:gw-gw]>`:
- reuses the cached upstream outputs;
- scores **at the component stage** (Brier / log-loss / RPS on minutes outcomes, calibration);
- then downstream at the player-fixture stage (RPS and log score of the points PMF).

Target: a minutes A/B over one season in **≤ 10 minutes** on a GHA runner (measure and record it).

### 2.5 Decision engine (experiments step 5)
- Superiority, non-inferiority and equivalence tests on **paired per-GW differences**, with HAC / DM and block-bootstrap intervals (reusing existing code). Gates are frozen at submit.
- **Short-circuit:** a clear component-stage failure skips the downstream stages.
- **Holdout:** the latest complete season, re-tested only for the selected arm.

### 2.6 Two-stage screening
- **Stage 1 (cheap, wide):** many arms scored on component and forecast measures (§2.3) over the screening window. Arms pass when they beat control by a pre-declared margin with q < 0.1 (FDR across arms).
- **Stage 2 (expensive, narrow):** survivors only, measured on decisions (squad points, attainable share) via the decisions suite and `simulated_season`.
- Budget: a per-batch cap on stage-2 arms (config). A results row records which stage each arm stopped at, and why.

### 2.7 Sweeps
Grid expansion over patch paths; response-curve and heatmap tables; the selected point re-tested on the holdout (`fplt-experiments.md` §8 decisions).

## 3. Optimiser-side levers

Some losses aren't forecast errors. The **optimiser's curse** (picked players' μ̂ biased high; measured by the post-GW selection-conditional bias, #395), captaincy risk and chip timing need their own levers:
- **Shrinkage:** an empirical-Bayes shrink of μ̂ towards a position × price prior, in proportion to forecast uncertainty (σ̂ from the PMF), applied only inside the optimiser objective. It's a `planner` patch, so it's testable via §2.1.
- **Risk term:** a mean − λ·sd objective option (`planner.objective`), λ swept via §2.7.
- **Test:** stage 2 only (decisions suite / simulated season). Forecast scores are unchanged by construction.

## 4. Speed

- **Fit reuse across arms:** arms that share an upstream component share its fit (`fit_fp`, #346), so a minutes arm never refits Dixon–Coles.
- **Screening fidelity:** stage 1 may run with reduced `n_sims` (config, e.g. 2,000 instead of 10,000) and records it on the manifest. Stage 2 always uses production `n_sims`.
- **Timing:** per-phase timing records for every experiment run (`fplt.utils.timing`), and a bench entry for the §2.4 target.

## 5. PR sequence (each under ~200 lines of non-test code)

1. **Error attribution** (§1.1): one-at-a-time oracle gain from stored component outputs, plus the `error_attribution` table. Shapley follows once the §2.2 cache exists.
2. **Signal screen** (§1.2): covariate registry, PIT guard, partial correlations, BH-FDR, `signal_screen` table.
3. **Patch engine** (§2.1).
4. **Measure interface + adapters** (§2.3).
5. **Component output cache + composite hook** (§2.2).
6. **Component experiment command** (§2.4) with the ≤ 10 min target benchmarked.
7. **Decision engine** (§2.5).
8. **Two-stage screening** (§2.6).
9. **Sweeps** (§2.7).
10. **Optimiser levers:** shrinkage and risk term as planner patches (§3).
11. **Speed:** screening fidelity and the bench entry (§4).

Tests throughout:
- the oracle gain of a component that is already perfect is 0;
- Shapley shares sum to the all-oracle gap;
- the PIT guard on covariates;
- BH-FDR on synthetic nulls keeps false discoveries ≤ 10%;
- an arm reusing the cache is bit-identical to a full refit for unchanged components;
- short-circuit skips downstream stages;
- holdout re-test only for the selected arm;
- the shrinkage objective reduces to the current one at shrink = 0.

Never assert model quality in pytest (AGENTS.md §3): quality lives in experiment results.
