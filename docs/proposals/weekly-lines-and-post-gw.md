# Weekly model lines and post-gameweek analysis

Status: proposal, owner decisions recorded 2026-11-29. Tracker: #398 (issues #389–#397).

Two questions:
1. Which models run every gameweek, and how are they compared?
2. What analysis runs once a gameweek is final?

## 1. Weekly lines: three tiers

Compute is cheap: a fit is about 30 s, a predict 15–35 s and a plan 15–40 s on a GHA runner, and Actions minutes are free. The constraints are noise and attention. A full virtual squad for every model would produce a noisy table nobody can read: separating two decent models on realised squad points takes most of a season, while forecast skill per player-fixture separates them in weeks.

| Tier | `role` | Runs | Question | Scored on |
|---|---|---|---|---|
| Primary | `primary` | predict + plan, real squad (`v4_h3` today) | What we play | Real points |
| Shadow decision lines | `candidate` (virtual squad) | predict + plan on a virtual squad, **at most 6 lines**: v4 h5/h7/h9, baseline `ep_next_v1` h3, one challenger slot | Would this decision process score more? | Decision ledger: realised points of the virtual squad |
| Prediction-only | `shadow_predict` (new) | predict only, no plan | Is this forecast better? | Prediction ledger: RMSE, calibration, log-loss per player-fixture |

### 1.1 Shadow decision lines (at most 6)

- Horizon lines: `v4_h5`, `v4_h7`, `v4_h9` (existing).
- **Baseline: `ep_next_v1` at h3 (decided 2026-11-29).** FPL's own free projection run through *our* optimiser. If v4 cannot beat it, the modelling adds no value. It's a sharper test than `rolling_mean_v1`.
- **Challenger:** the current experiment winner. Approving an experiment adds or replaces this line, so it is the bridge to the agentic experiment loop (promotion to primary only after N live weeks).

Fairness rules:
- **Start:** a line created mid-season starts from the real squad at its first GW (existing `prev_run_id` chain).
- **Chips: decided 2026-11-29: both.** Each line plays chips independently under its own policy, and every comparison is reported twice:
  - **with chips:** the line's realised points as played;
  - **ex-chips:** realised points as if no chip had been active that GW, defined below.
- **Scoring:** FPL rules via `fplt.optimise.realised.score_gameweek` (autosubs, vice on blank, hits).
- **Comparison:** each line's **weekly paired difference** from the primary, plus the cumulative season-to-date difference with a confidence interval (paired bootstrap over GWs). Raw totals are shown but never ranked alone.

**Ex-chips definition** (per line, per GW when a chip was active):

| Chip | Ex-chips points |
|---|---|
| Bench Boost | Score without bench points (normal 11 + autosubs). |
| Triple Captain | Captain counted ×2, not ×3. |
| Free Hit | Score the squad the line held **before** the Free Hit (carried-over squad), with the XI and captain re-picked by the line's own lineup rule (`lineup.py`, by that line's predicted xPts). |
| Wildcard | Equal to with-chips (a wildcard changes the squad, not the scoring). Report the wildcard GW, and set `wildcard_active=true` in the row. |

Ex-chips rows carry the same keys plus `chip_adjusted=true`, so cumulative sums never mix definitions.

### 1.2 Prediction-only lines

**Roster (decided 2026-11-29):**

| Model | Why |
|---|---|
| `ep_next_v1`, `openfpl_v1` | External benchmarks (OpenFPL season guard below) |
| `dc_layered_v3`, `dc_layered_v2` | Ancestors: do the backtested improvements hold up live? |
| `rolling_mean_v1` | Naive floor |
| `lgbm_probs_v1` | Different model family (machine learning vs Dixon–Coles layering) |
| `forecast_combine_probs_v1` | Ensemble check (dropped offline, cheap to watch live) |

- **Components:** the primary's own minutes / scoreline / involvement outputs are already stored in each official predict run, and are scored by §2.3 at no extra cost. Alternative component models are compared in component experiments, not weekly lines, because they need the composite rebuilt around them.
- **Timing: non-blocking.** Prediction-only lines must never delay the T−90m official slate. They run in a separate step or job after the official slate has been written (or at T−6h). A failure there alerts but doesn't fail the slate. At about 1 min per model (fit + predict), the roster costs about 8 min.
- Benchmarks: archived FPL `ep_next` and OpenFPL projections (`fplt.data.benchmark_projections`) are scored as pseudo-models in the same ledger. Per AGENTS.md §6, OpenFPL is never benchmarked on 2020-21 to 2023-24.
- Run at T−90m together with the official slate, as `kind=official`, `run_type=predict`, with no plan, no PMF and no pointer. Fingerprint skip-if-done applies (#346, #380).

## 2. Post-gameweek analysis

### 2.1 Trigger and output

- **Trigger:** ops-tick sees `events[].data_checked` flip to true for a GW (results final).
- **Output:** one write-once **analysis run** per GW (`runs/run=…`, `run_type=eval`, `purpose=post_gw`, `tags.gw=<season>-<gw>`), holding the tables below plus `summary.md`. Done = that run exists (fingerprinted on the outcomes table sha + the GW's official pointers), so re-runs are skipped.
- **Notify:** one ntfy line, for example "GW13: 61 pts (exp 57, 64th pct), rank ↑; best line v4_h7 +6; biggest miss: Salah (−9, benched)".
- **UI:** a "Last week" page. It **leads with season-to-date numbers with intervals**; the single week is shown as detail. One GW is mostly noise.

### 2.2 Contents

**A. Scorecard**
- Real squad: realised vs expected, and the **percentile of realised points in the official points PMF** (`points_pmf_gw`). Over a season, the percentile histogram tests whether the distributions are calibrated.
- Context: FPL average and highest scores (bootstrap `events[]`), the manager's rank change (public entry history).
- Lines table (§1.1): weekly and cumulative paired differences vs primary, **with chips and ex-chips**.

**B. Decision attribution** (real squad and each line; reuses ops-08 `fplt.optimise.decision_scoring`)
- Transfers: points of players in minus players out over the window, minus hits.
- Captaincy: captain vs best XI alternative, and vs the model's second choice.
- Bench and autosubs: bench points; gain or loss from autosubs.
- Chips: value vs the ex-chips counterfactual.
- Luck vs model: XI realised minus XI expected.
- Hindsight regret: constrained hindsight optimum (existing `regret-board` logic).
- Interventions: manual overrides (intervention records) and whether they helped.

**C. Model diagnostics** (primary, prediction-only lines and benchmarks)
- Season-to-date calibration of xPts and each component, sliced by position, price band and team: minutes (`p_played`, `p_60` Brier), goals/assists (log-loss vs `p_goal` / `p_assist`), clean sheets (`p_cs`), bonus, cards.
- **Biggest misses** (top 15 by |actual − xpts|), each tagged with a reason: `benched`, `subbed_early`, `red_card`, `injury`, `penalty`, `dgw`, `blank`, `other`. Plus a **`predictable_at_T90` flag**: availability news or predicted-lineup data available before T−90m pointed the other way. Flagged misses are the actionable errors, and the input to idea generation.
- Versus benchmarks (`ep_next`, OpenFPL) on the same fixtures.
- Drift vs backtest expectations (ops-09 `fplt.automation.drift`).

**D. Ops review**
- Which checkpoint's plan was applied (`applied.json`), the data freshness at each checkpoint, run timings and any failures or alarms for the GW.

### 2.3 Forecast evaluation (statistics)

Each player-fixture has a predictive PMF p̂(k), k = −5…30+ (official `points_pmf`), and one outcome y. Principle: **maximise sharpness subject to calibration** (Gneiting et al.).

**Proper scores.** These are stored per player-fixture and aggregated season to date with cluster-robust intervals.
- **RPS** (discrete CRPS): Σₖ (F̂(k) − 𝟙{y ≤ k})². The headline, reported as a skill score against references: climatology (the empirical PMF by position × price band × minutes-likely bucket, fitted on prior seasons only) and `ep_next` / OpenFPL on the mean.
- **Log score:** −log p̂(y), with a documented probability floor.
- **Brier** for derived events: P(played), P(≥60), P(goal), P(assist), P(CS), P(haul ≥ 10), P(blank ≤ 2).
- Reliability / resolution / uncertainty decomposition of RPS and Brier.

**Calibration.**
- **Randomised PIT** u = F̂(y−1) + V·p̂(y), V ~ U(0,1) (Czado, Gneiting & Held 2009). Use a fixed per-row seed via `fplt.utils.rng.seed_for` so it's reproducible. Show the PIT histogram, plus central 50/80/90% interval coverage.
- Reliability diagrams for each event probability.
- Mincer–Zarnowitz regression of y on μ̂ (slope, intercept with CIs), overall and by position, price band, home/away and minutes certainty.

**Discrimination.** Within-position Spearman correlation of μ̂ vs y per GW; top-k hit rate (k = 5, 10 per position); MSE of the mean.

**FPL-specific.**
- **Selection-conditional bias (optimiser's curse):** realised − expected for players the plan picked vs all players. Reported for the real squad and every shadow line.
- **Minutes vs points given minutes:** P(played) / P(60) scored separately from the PMF conditional on playing (when the run stores component outputs).
- **Dependence:** intervals use a block bootstrap by GW (and cluster by fixture for per-GW statistics). Model comparisons use paired per-GW score differences with HAC / DM tests (existing code).
- **Squad total:** PIT of the realised squad total under the joint squad PMF, which needs **joint** draws, not a convolution of marginals. It's pooled across lines and seasons because of its low power.

**Storage.** The analysis run stores a `forecast_scores` table (one row per player-fixture: keys, RPS, log score, PIT u, event outcomes and probabilities, selected flags per line). Everything above is an aggregation of that table, so new slices need no re-scoring.

Tests: RPS and log score against hand-computed examples; PIT of a calibrated synthetic forecast is uniform (KS on 10k draws with a fixed seed); a deliberately overconfident synthetic forecast gives a U-shaped PIT; the selection-bias statistic is positive on a synthetic optimiser-picks-top-noise scenario; reproducible PIT under a fixed seed.

## 3. PR sequence (each under ~200 lines of non-test code)

Lines:
1. `role: shadow_predict` in `ProductionOps` / `production.yaml`: predict-only official runs at T−90m, fingerprinted, without plan, PMF or pointer.
2. Benchmark projections scored as pseudo-models in the prediction ledger (`ep_next`, OpenFPL with the season guard).
3. Baseline and challenger virtual lines (config), capped at 6 shadow decision lines (validator).
4. Line comparison tables: weekly paired diff + cumulative paired-bootstrap CI, with chips and ex-chips (`chip_adjusted`), per §1.1.

Post-GW:
5. `post_gw` step in ops-tick: detect `data_checked`, write the analysis run (scorecard + attribution from `decision_scoring`), `summary.md`, ntfy. Done = run exists.
6. Biggest misses with reason tags and the `predictable_at_T90` flag.
7. Forecast evaluation (§2.3): `forecast_scores` table (RPS, log score, randomised PIT, event Brier), climatology reference, selection-conditional bias; aggregations (PIT, coverage, reliability, Mincer–Zarnowitz, top-k) with block-bootstrap CIs.
8. "Last week" UI page (season-to-date first, week detail second).
9. Benchmarks, rank context and the ops review section.

Tests throughout:
- ex-chips for each chip, including a Free Hit that reverts and a Bench Boost in a DGW;
- paired CI on synthetic lines with a known difference;
- `predictable_at_T90` uses only data with `available_at` ≤ T−90m (PIT);
- the post-GW run is written exactly once per GW and is skipped on a re-tick;
- blank and double gameweeks in every table.
