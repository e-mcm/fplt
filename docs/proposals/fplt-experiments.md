# fplt experiments: one general definition

**Status:** proposal · **Builds on:** `experiment-framework-design.md` (arms, levers, CRN, statistics), `fplt-data-model.md` §3.8 (storage)

## 1. The pipeline as a graph of stages

Every experiment changes something at one stage and measures at the same stage or a later one:

```
data ─► features ─► components ─────────────────► simulator ─► planner ─► policy ─► season
(sources,  (versions)   score · minutes ·          (player-fixture  (plan over  (apply rules, (realised
 builds,                involvement · other         points PMF)      horizon H)  chips, hits)   points, rank)
 timing)
                                    environment (code, libraries, mc backend, hardware) cuts across all stages
```

- **Intervention point:** where the arms differ.
- **Measurement point:** the stage whose output is scored. It must be at or after the intervention point. Measuring before it would show no difference by construction.
- Measuring **at** the intervention point (e.g. minutes Brier for a minutes change) is cheap and diagnostic. Measuring **downstream** (season points for a minutes change) is what you actually care about, but noisier and costlier. A good experiment usually does both.

## 2. The definition

An experiment is **E = (Question, Base, Arms, Context, Measures, Decision)**.

| Part | Meaning |
|---|---|
| **Question** | One sentence, required. What would change your mind? |
| **Base** | A fully resolved `PipelineSpec`: data config, feature versions, `ModelSpec` (components + simulator params), `PlannerSpec`, `PolicySpec`, environment. Usually `production` (the primary line) or a named spec. |
| **Arms** | Each arm is a **patch** on the base: dotted-path overrides, a component substitution (including `oracle`), or an environment change. The control is the empty patch. |
| **Context** | Mode (`backtest`, `live_forward`, `snapshot` for "as of now"), deadlines / windows (primary, confirm, holdout), population filter, horizon, seeds (CRN), and the pinned data build. |
| **Measures** | One or more `(stage, metrics, slices)` tuples, evaluated in order. Cheap stages first. **Short-circuit is on by default:** if a gated earlier stage fails clearly (the CI lies entirely on the wrong side of zero, or of the margin), later stages are skipped and recorded with the reason. `--no-short-circuit` forces every stage. |
| **Decision** | `none` (look only), `superiority`, `non_inferiority` (with margin) or `equivalence` (with margin), plus a primary metric, gates and α. The code computes a verdict; you make the decision in `registry.yaml`. |

**Validity rules enforced at submit:**
1. Each arm's resolved config differs from the base by **exactly its patch**. The harness computes the diff and refuses hidden changes.
2. Every measurement stage is at or downstream of every arm's intervention point.
3. All arms share the context: same build, deadlines and seeds.
4. **Holdout:** the standing holdout is the most recent complete season (currently 2025-26). It is required whenever an experiment has more than 2 treatment arms or is a sweep. Arms are selected on the primary and confirm windows only; the selected arm alone is then re-tested on the holdout, and that re-test is the reported verdict. This is the multiple-comparison guard: no Holm or MCS correction by default. The holdout rolls forward when a season completes, and runs already submitted keep the holdout they were frozen with.
5. Oracle arms are evaluation-only: never in production lines, and never the "winner" of a superiority test.

## 3. Why this needs a `PipelineSpec`

Today `ModelSpec` covers the model, but planner and policy settings (horizon, decay, hit rules, chip policy, pool size, solver limits) live in CLI flags, `production.yaml` and code. Experiments on them can't be hashed or diffed the same way. So:

```yaml
# a resolved PipelineSpec (hashed like a ModelSpec, §3.7 of the data model)
data:     {sources: {odds_capture: false, fplcache: false, api_football_injuries: true}, capture_checkpoint: T-90m}
model:    dc_layered_v4_native            # a registered ModelSpec name, or inline
planner:  {horizon: 3, decay: 0.85, pool: {per_position: 15}, mip_gap: 0.001, time_limit_s: 600, objective: mean}
policy:   {max_hits: 1, hit_min_gain: 4.0, chips: {scheduler: season_v1}, captain: max_xpts}
env:      {mc_backend: native}
```

Official lines (data model §4.1) become `(line name, PipelineSpec)`. "v4 at horizon 3" vs "v4 at horizon 9" is then just a planner patch.

Dotted override paths address any node: `model.components.minutes.params.logistic_blend`, `model.params.n_sims`, `planner.horizon`, `policy.hit_min_gain`, `data.sources.odds_capture`, `env.mc_backend`.

## 4. Component experiments (e.g. minutes)

An arm can change or swap one component:

```yaml
arms:
  control: {}
  blend_08: {model.components.minutes.params.logistic_blend: 0.8}
  gated:    {model.components.minutes: empirical_minutes_v3}   # swap the whole component
  oracle:   {model.components.minutes: oracle}                 # perfect minutes, value-of-information bound
```

What makes these cheap and clean:

- **Component runs.** A run can be a single component's output at each deadline (`run_type: component`), stored at `predictions/components/minutes.parquet`, keyed like everything else.
- **Component output cache.** A component's output depends only on `(component spec hash, code_sha, build_id, deadline)`. A downstream arm that changes minutes **reuses the control's cached scoreline, involvement and other outputs** and only refits minutes, then re-simulates. Dixon–Coles isn't refitted for a minutes experiment. This needs one hook: the composite model accepts precomputed component outputs for a deadline instead of fitting them.
- **Interaction caveat.** Some components read others: involvement normalises within the team, lineup multipliers (model-03) read minutes. The cache key must include upstream component hashes where there is a dependency, so the graph is explicit, not assumed.
- **Measurement:**
  - at the component stage: the channel scorecard from eval-01 (Brier, RPS, calibration), both on the model's own minutes and on the minutes actually played;
  - downstream: player-fixture RMSE and haul Brier, then season points and attainable share on the decision board.

  Short-circuit rule: if the component-level gate fails clearly, skip the season stage.

## 5. The catalogue: experiments this covers

| # | Kind | Intervene at | Measure at | Decision | Example |
|---|---|---|---|---|---|
| 1 | Parameter change / sweep | component or model | component → season | superiority on holdout | `xg_blend ∈ {0.3…0.9}`; Optuna trials are arms |
| 2 | Component swap | component | component → season | superiority | `empirical_minutes_v2` vs a gated v3 |
| 3 | Feature ablation / add | features | component → season | superiority | drop congestion features; feature v2 vs v1 |
| 4 | Data source value | data | component → season | superiority | add deadline odds; add fplcache history; injuries overlay |
| 5 | Data timing / freshness | data (checkpoint) | prediction → plan | superiority | predictions from T−24h vs T−90m data |
| 6 | Data degradation | data | season | none (sensitivity) | run without Understat: how much do we lose if it disappears? |
| 7 | Value of information (oracle) | component | season | none (upper bound) | perfect minutes / penalty taker / scoreline (eval-02) |
| 8 | Training configuration | model | component → season | superiority | window length, time-decay ξ, refit cadence, pooled seasons, prior strength |
| 9 | Post-hoc calibration | component / prediction | prediction | superiority | isotonic P(start); shrink xPts toward the position mean |
| 10 | Ensemble / combination | prediction | prediction → season | superiority | `forecast_combine` weights; model averaging |
| 11 | Simulator fidelity | simulator | prediction → plan | superiority or equivalence | `n_sims` 5k vs 10k; hard XI; correlated bonus; parameter uncertainty |
| 12 | Planner | planner | plan → season | superiority | horizon {3,5,7,9}; decay; pool size; robust vs mean objective; MIP gap |
| 13 | Policy | policy | season | superiority | hit threshold; captain rule; chip scheduler; Free Hit timing |
| 14 | Forecast × optimiser interaction | prediction × planner | season | none (diagnostic) | 2×2 swap: better forecasts vs better planning (exists: `forecast_optimiser_swap.py`) |
| 15 | Equivalence / refactor | env | prediction (bitwise or tolerance) → season | equivalence | native vs Python MC; library upgrade; refactor |
| 16 | Robustness / noise floor | seeds or windows | any | none | same spec, 10 seeds: how big is MC and solver noise? Run before reading any small effect |
| 17 | Segment analysis | none (slicing) | any | none | does the gain come from premiums, GKs, double gameweeks, early season? |
| 18 | Benchmark | arms are external sources | prediction | none | vs FPL `ep_next`, OpenFPL, bookmaker-implied |
| 19 | Snapshot comparison | model or planner | prediction / plan, no outcomes | none | "H=10 xPts, A vs B, as of now" |
| 20 | Live forward test | any | prediction → season, accrued weekly | superiority over time | candidate official lines with virtual squads |
| 21 | Sensitivity to a specific input | data (perturbation) | prediction / plan | none | set one player's flag to doubtful: how much does the plan change? |

Kinds 16 and 17 run on top of others: every reported effect should carry its noise floor (16) and its segment breakdown (17).

**Noise floor:** a standing 10-seed robustness run per base `PipelineSpec`, re-run automatically whenever the base hash or `code_sha` changes. Experiments reference the latest one for their base instead of replicating arms.

## 6. One YAML for all of them

```yaml
# configs/evals/minutes_logistic_blend.yaml
name: minutes_logistic_blend
question: Does a stronger logistic blend improve minutes calibration, and is it worth points?
base: production                     # PipelineSpec of the primary line
arms:
  control: {}
  blend_07: {model.components.minutes.params.logistic_blend: 0.7}
  blend_09: {model.components.minutes.params.logistic_blend: 0.9}
  oracle:   {model.components.minutes: oracle}
context:
  mode: backtest
  windows: {primary: 2023-24, confirm: 2024-25, holdout: 2025-26}
  seeds: crn
  population: all                    # or decision_pool
measures:
  - stage: component:minutes
    metrics: [brier_played, brier_60, rps_buckets, team_xi_consistency]
    slices: [position, flag_at_deadline, horizon_step]
  - stage: prediction
    metrics: [rmse_points, haul_brier]
  - stage: season
    metrics: [season_points, attainable_share, regret]
decision:
  type: superiority
  primary: {stage: season, metric: season_points, window: holdout}
  gates:
    - {stage: component:minutes, metric: brier_60, rule: "diff <= 0", short_circuit: true}
  alpha: 0.05
  exclude_from_winner: [oracle]
```

Other kinds are the same shape. A snapshot comparison is `mode: snapshot`, one deadline, `measures: [{stage: prediction}]`, `decision: {type: none}`. A refactor is an `env` patch with `decision: {type: equivalence, margin: …}`. A horizon study is a `planner.horizon` grid measured at `season`.

## 7. What changes in the data model

- `PipelineSpec` and its hash replace `spec_hash` as the arm identity. `ModelSpec` hashes remain for component and model caching.
- `run_type` gains `component`. The component output cache key is `(component spec hash, upstream component hashes, code_sha, lockfile_hash, build_id, deadline)`.
- `ExperimentManifest` stores the resolved base, each arm's patch and resolved hash, the context, the measures and the decision, all frozen at submit.
- `results/` gets one `arm_metrics` / `paired` table per measured stage (`stage` column), plus `short_circuited` events.
- `kind` examples from the catalogue go in a free `tags` field, not a fixed enum. The definition is general; tags are only for browsing in the UI.

## 8. Decisions (2026-11-24)

- Adopt `PipelineSpec` with all five sections (data, model, planner, policy, env).
- Holdout: the latest complete season; required for sweeps and for experiments with more than 2 treatment arms; selected arm re-tested alone (rule 4).
- Noise floor: a standing robustness run per base (§5).
- Short-circuit on by default (§2).
- Multiple comparisons: holdout re-test only.

## 9. PR sequence

1. `PipelineSpec` schema, with `planner` / `policy` / `data` / `env` sections typed and hashed. Production lines expressed as PipelineSpecs. Test: today's ops behaviour is reproduced exactly (output provably unchanged).
2. Generic patch engine: dotted paths, component substitution, grid expansion, a validity check that the diff equals the patch, auto-labels.
3. Stage-tagged measures: adapters that wrap existing scorers (`component_eval`, `minutes_measurement`, `other_points_measurement`, the board, `simulated_season`) behind one `Measure` interface, `(stage, metrics, slices) → table`.
4. Component runs and the component output cache, plus the composite hook for precomputed outputs.
5. Decision engine: superiority, non-inferiority and equivalence, with gates, short-circuit and holdout reporting. Reuses the existing DM/HAC and bootstrap code.
6. Migrate existing EvalSpecs to the new shape (mechanical; the old loader stays until they're all moved).
