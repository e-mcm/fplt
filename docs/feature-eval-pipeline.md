# Feature / data / config / component eval pipeline

Standing design so **any model lever** — new staged data, a feature version, a hyperparam diff, a layered component swap, a full predictor fork, a lockfile upgrade, or a no-behaviour refactor — uses one path: **control vs treatment → walk-forward suites → a written recommendation**. Improvement levers are two ModelSpecs. Equivalence levers share the spec and differ in the lockfile or the git ref. Decision bakeoffs stay opt-in.

**Status:** P1+P2 promote-eval execute and the **Stats** package (HAC OLS + stationary-bootstrap intervals, SPA/MCS) are on main. **P2.5** splits `produce_ledger` from compare and caches each arm as content-addressed parquet (not `joblib`). **P3** adds slot-named `eval-components` (with `--skip-sims`), scoreline `per_match.parquet`, and the raw de-vig 1X2 market row. **P7** adds `kind: environment` and `kind: refactor` (worktree + `uv sync --frozen` + subprocess produce) with the §3.2 identical / different-not-worse / worse verdict. **Bakeoff** adds a thin multi-arm wrapper (`fplt bakeoff` / `BakeoffSpec`, plus optional EvalSpec `arms:`) that reuses compare + the ledger cache and writes SPA/MCS + a MAE rank table — descriptive, not a `keep`. Intervals are descriptive for improvement levers; equivalence may `keep` on identical predictions or `drop` when an interval is entirely on the bad side (not a `keep` rule for improvement). Numeric keep/drop thresholds for improvement wait on FDB-73.

**Related:** project-store `docs/odds-dc-implementation-plan.md` (W7–W9), `docs/model-compare-tooling.md`, `docs/model-hyperparameters-and-train-window.md` (S0–S5), `docs/defensive-contribution-eval-2025-26.md` (W9 numbers). Repo rules: `docs/design.md` §7.4 and §7.6, `AGENTS.md`.

---

## 1. Problem

Each lever had been inventing its own eval. The shape that survived the hand runs is always two YAMLs and one window. What varied, and what this doc now fixes, is which YAML is the control, what “coverage” counts, and which PIT clock the model actually reads.

| Lever | Arms that are real on main | What the hand run taught |
|---|---|---|
| Odds prior (W7) | `dc_layered_v2` vs `dc_layered_odds_v1` | Production strength is **0**. Treatment is the on-fork. The landed prior blends λ at predict time and cannot see closing lines. The next study is a history rating prior against raw de-vigged 1X2 (§5.1). |
| Lockfile, refactor | none yet | Same spec. Question is "did forecasts move, and did they get worse?" (§3.2). |
| `penalties_order` (W8) | `dc_layered_nopens_v1` vs `dc_layered_v2` | Production default is **on**. Control is the off-fork. Vaastav writes the column null, so the arms tie. |
| `defensive_contribution` (W9) | `dc_layered_nodefcon_v1` vs `dc_layered_v2` | Same opt-out shape. Only 2025-26+ rules award the points. MAE and RMSE disagreed; the call was `tune`. |
| Minutes swap, direct-model fork, `xi` / `seasons_back` | still future | Same EvalSpec. One lever. |

---

## 2. Core invariant

Promote-eval does not special-case “dataset” vs “XGBoost” in the runner. Both arms are named specs under `configs/models/`.

| `kind` | `change.lever` | What differs |
|---|---|---|
| `data` | `data_table` | Treatment is the first consumer of a new table. Rare once a flag exists. |
| `feature` | `feature_version` | `features: [name@N+1]` vs `@N` |
| `config` | `hyperparam` | Same `kind` and component graph; one param family |
| `component` | `component_swap` | One `components.<slot>.kind` |
| `model` | `model_fork` | Top-level `kind` (`lgbm_direct` → `xgb_direct`) |
| `environment` | `dependency_upgrade` | `uv.lock` only. Resolved specs byte-identical. Equivalence (§3.2). |
| `refactor` | `no_behaviour_change` | Git ref only. Specs and lockfile identical. Equivalence (§3.2). |

```text
off YAML  ──┐
            ├──► same window, same seasons_back, same suites
on YAML  ───┘         │
                      ├── coverage on the model's real read
                      ├── prediction (compare)
                      ├── components, when the slot has a metric
                      └── recommendation.md
```

**Control is the off state. Treatment is the on state.** Production is whichever of those two already matches the spec we ship. The fork is always the other one. Never edit the production YAML to manufacture the opposite arm.

Code defaults, absent key included:

| Param | Absent means | Production `dc_layered_v2` today |
|---|---|---|
| `use_penalties_order` | `true` | on |
| `use_defensive_contribution` | `true` | on |
| `market_prior_strength` | `0` | off |

`PRODUCTION_MODELS` in `fplt.automation.cycle` is the live-plan list (`dc_layered_v2` today, horizon `DEFAULT_PLAN_HORIZON` = 3). It is not the arm check. The arm check reads the layered spec under test.

Other hard rules, unchanged:

- New `kind` values go through `register_model` / `register_component` and the contract suite, including the leakage test, before an eval.
- Models read only through `PITView`. New staged tables carry `available_at`.
- Artifacts stay under `artifacts/<model_id>/as_of=…/` via that model’s `save` / `load`.
- One lever per EvalSpec. `change.diff` lists it. A dose of one continuous knob (`arms:`, or several EvalSpecs on one cached control) is still one lever. It is a descriptive readout. `keep` comes from a confirmation window that did not rank the arms (§9). A second slot, flag, feature, or an unlisted direct dependency is not one lever.
- Improvement levers (`data`, `feature`, `config`, `component`, `model`) ask whether forecasts got better. Equivalence levers (`environment`, `refactor`) ask whether they moved, and whether they got worse (§3.2).

The three flags above are `kind: config` once the consumer is on main. Staging the table is a prerequisite PR, not the A/B. `kind: data` remains for a consumer that has no off switch. `example_data_odds_prior.yaml` predates the flag and mis-labels W7 as `data`. Copy the specs in §4.2.

---

## 3. Arm orientation

P1 adds `change.orientation: opt_in | opt_out` and checks the loaded params. Effective value uses the code default when the key is absent. Both arms equal → fail (`no contrast`). Arms swapped relative to `orientation` → fail (`arms inverted`).

### Opt-out — production already has the lever

Control = new YAML with the flag off. Treatment = production YAML.

| Lever | Control | Treatment |
|---|---|---|
| `use_defensive_contribution` | `dc_layered_nodefcon_v1` (`false`) | `dc_layered_v2` |
| `use_penalties_order` | `dc_layered_nopens_v1` (`false`) | `dc_layered_v2` |
| `market_prior_strength` | a fork with `0`, **only after** production sets it > 0 | that production spec |

`configs/evals/pens_order_involvement_ab.yaml` already has the pens orientation. It still needs the coverage block in §6.

What the recommendation means:

| Call | Opt-out (treatment is production) |
|---|---|
| `keep` | Leave production on. Do not merge the off-fork as the default. |
| `drop` | A follow-up PR turns the production default off. |
| `tune` | Leave production on. Do not claim a metric win. |
| `inconclusive` | Same as `tune` until FDB-73 sets thresholds. |

W9 is the specimen. With-defcon MAE 0.991 vs without 0.983, RMSE 1.983 vs 1.994, Spearman flat, mean xPts closer to actuals on DEF/MID. Call: **`tune`**. A rule that keeps only on MAE would have turned the wire off.

### Opt-in — production does not have the lever

Control = production (off). Treatment = fork (on).

Odds today: control `dc_layered_v2` (`market_prior_strength` absent → 0), treatment `dc_layered_odds_v1` (`1.0`). Calling this opt-out would put the off state on both arms.

| Call | Opt-in (treatment is the fork) |
|---|---|
| `keep` | Follow-up PR adopts the treatment spec. This runner does not edit `PRODUCTION_MODELS`. |
| `drop` | Leave production unchanged. |
| `tune` | Keep the fork. Do not promote it. |
| `inconclusive` | P2’s only metric call until FDB-73. |

### 3.2 Equivalence levers

`kind: environment` and `kind: refactor` do not ask whether the change is better. Control is the production state (the lockfile on `main`, or the git ref we ship). Treatment is the candidate. An environment lever keeps the resolved specs byte-identical. A refactor keeps the specs and the lockfile identical. Orientation does not apply. A spec that also changes is a second lever and fails preflight unless `allow_multi_lever: true`.

Verdict, in this order:

1. **Identical.** Every prediction in every fold is within `equivalence.tolerance` (`xpts_abs` default `1e-6`, `prob_abs` default `1e-9`). Compare ledgers sorted by key with `polars.testing.assert_frame_equal` (`rtol=0`). Verdict: **`keep`**. The report is the count of rows compared.
2. **Different, not worse.** Some rows exceed tolerance, and no primary-metric interval (§9) lies entirely on the bad side. Verdict: **`inconclusive`**. List the largest diffs by fixture and player. A changed RNG stream, a changed default, or a changed tie-break lands here.
3. **Worse.** At least one primary metric has its interval entirely on the bad side. Verdict: **`drop`**.

This `keep` and this `drop` do not wait on FDB-73, and they do not apply to improvement levers. MAE can tie while individual forecasts move. The ledger diff is the check. Once FDB-73 sets a non-inferiority margin, step 3 may become a paired TOST on the per-gameweek loss differentials. Until then the interval rule stands.

```yaml
equivalence:
  tolerance:
    xpts_abs: 1.0e-6
    prob_abs: 1.0e-9
  compare: predictions
```

---

## 4. EvalSpec

Path: `configs/evals/<name>.yaml`. Loader: `fplt.backtest.promote_eval.EvalSpec` (`extra=forbid`).

Fields below marked **P1** are the contract for the next PR. They do not load on main today. Put them in the YAML only in that PR, together with the parser.

```yaml
name: minutes_lgbm_vs_empirical
kind: component
question: "Does a LightGBM minutes model beat empirical_minutes_v2 on Brier, then on xPts?"  # P1; runner refuses a blank question
description: "dc_layered minutes slot: empirical_minutes_v2 → lgbm_minutes"

change:
  lever: component_swap
  orientation: opt_in          # P1
  slot: minutes
  control_kind: empirical_minutes_v2
  treatment_kind: lgbm_minutes
  tables: []
  features: []
  allow_multi_lever: false     # P1; example_* scaffolds set true
  diff:
    - "components.minutes.kind: empirical_minutes_v2 → lgbm_minutes"
  notes: "same window; n_sims unchanged"

control: dc_layered_v2
treatment: dc_layered_minutes_lgbm_v1
# arms: [dc_layered_odds_strength_0_25, …]   # optional dose; SPA/MCS descriptive

window:
  seasons: ["2024-25"]
  from_gw: 1
  to_gw: 20
  seasons_back: 4
  fit_gw: null
  modes: [walk_forward]

suites:
  prediction: true
  components: true
  decisions: false
  steps: 1
  decisions_horizon: 3
  max_mean_ppg: 90

gates:
  min_coverage: null
  allow_low_coverage: false
  coverage:                    # P1; formula is selected by metric, not free text
    metric: null
    read: null
  keep_if:                     # columns to print; not an auto-keep until FDB-73
    prediction_any: [spearman, mae]
    components_any: [brier_played, brier_60]
  drop_if:
    coverage_below_min: true
    mean_ppg_gate_fail: true

report:
  title: "Minutes: LGBM vs empirical_v2"
  store_doc: null
```

`keep_if` names match the tools: prediction `rmse` (headline), `mae` (secondary continuity), `spearman`, `top10_precision` (mean per-GW), `brier_played`, `brier_60`, `brier_goal`, `brier_assist`, `brier_cs`, `brier_haul`, and the matching `log_*` event scores; components `rps`, `logloss`, `brier_cs_home`, `brier_cs_away`, `brier_played`, `brier_60`, `mae_goal_share`. Mean-seeking boards score the decision-relevant pool (`fplt.backtest.decision_pool`). Print per-GW win counts (RMSE headline; MAE beside for continuity). Compare / promote-eval also write calibration tables by xPts decile and position.

### 4.1 Default suites

| `kind` | Suites | Primary readout |
|---|---|---|
| `config` (flag or one knob) | prediction; components when the slot has a metric | slot metric, then xPts slices (RMSE headline) |
| `data` | prediction; components if the consumer has one | same, plus coverage |
| `feature` | prediction | Spearman, RMSE (MAE secondary) |
| `component` | components, then prediction | slot metric before xPts `keep` |
| `model` | prediction; decisions only when claiming squad points | RMSE, Spearman, Brier (MAE secondary) |
| `environment` | prediction ledger diff | count and size of diffs; metric intervals |
| `refactor` | resolved-spec hash first; ledger diff only if the hashes differ | identical yes/no |

`suites.decisions` defaults off. A data or config promo does not need a decision bakeoff. W9’s `tune` was a prediction-only call. `kind: model` still needs S3+ before a production flip that claims transfers. `kind: component` needs a slot-metric win before an xPts `keep`; decisions stay optional.

---

## 5. Concrete specs (odds, pens, defcon)

These are the P1 targets. `gates.coverage` and `change.orientation` will not validate until that PR.

### 5.1 Odds — two different studies

**Recorded arm (landed, do not treat as the next experiment).** `market_prior_strength` blends Dixon–Coles λ toward `odds_to_lambda` at predict time. Proportional devig. `MatchScoreModel` reads `PITView.latest("match_odds")` (`available_at <= deadline`). Football-data rows are `odds_kind=closing_proxy` with `available_at=kickoff`, so `test_odds_after_deadline_are_invisible` makes the prior match vanilla. `odds_poisson_v1` is a different read (`use_closing_proxy: true`). It is also the wrong stand-in for the book: on 2024-25 GW1–38 closing lines, raw proportional de-vig of 1X2 has RPS **0.366** and log-loss **0.967**; `dixon_coles_v2` is **0.334** / **1.235**; `odds_poisson_v1` is **0.363** / **1.286** (project-store `docs/dc-odds-prior-eval.md`). The Poisson map throws the bookie’s 1X2 calibration away. Strict-PIT join on the gameweek being predicted was **0/199**.

This spec still fails preflight on coverage until a model PR opts the score component into a labelled proxy. Do not lower `min_coverage`. Do not teach the runner a second cutoff.

**Next odds model (its own model PR, then one EvalSpec).** `market_prior_source: history` on the score component. `fit` does not read odds. `market_prior_strength` only blends λ inside `predict`, on the fixtures being scored. With `history`:

1. Take odds for **past** fixtures visible through `PITView` at the deadline. A past close has `available_at = kickoff`, and that kickoff is already before the deadline, so the row is visible. No second cutoff. No `use_closing_proxy`.
2. Convert with the existing `odds_to_lambda`. Production `de_vig` is proportional. Shin is a later param on that function, not a runner choice.
3. Fit market ratings `(a0, d0)` in the Dixon–Coles parametrisation, with the same `ξ` time-decay weights the fit already uses.
4. Add `0.5 · strength · (Σ(a − a0)² + Σ(d − d0)²)` to the objective. `strength → 0` matches vanilla. Home advantage and `ρ` stay unpenalised in v1.

`market_prior_source: history` is the same switch as `odds_history: pre_kickoff`. One param. The target-fixture blend stays the recorded arm above.

Coverage for this model is `odds_history_join` (§6), not the target-fixture `odds_join` gate. The EvalSpec is `kind: config`, `opt_in`, control `dc_layered_v2`, treatment the history fork (built with `extends`, §11). A dose such as 0.25 / 1 / 4 on one window is descriptive. Report Hansen's SPA and the model confidence set (§9). `keep` is a later EvalSpec on a window that did not rank the values. The benchmark row is **raw de-vigged 1X2** on the same clock as `coverage.read`. `odds_poisson_v1` may be printed as a model column. It is not the book.

Optuna, if used, tunes on a dev window that does not overlap the eval window and writes an `extends` fork. It does not run inside promote-eval.

A deadline snapshot table (`match_odds_deadline`, `odds_kind=deadline_snapshot`, `available_at` = capture time) is what a **target-fixture** prior still needs. Football-data has no `captured_at` and no book named for that fetch. The history prior does not wait on it.

```yaml
name: odds_prior_dc_layered
kind: config
question: "Does the landed predict-time λ blend move scorelines or xPts once the odds are actually visible at the deadline?"
description: "Recorded W7 arm. Not the rating-prior study."

change:
  lever: hyperparam
  orientation: opt_in
  slot: score
  control_kind: dixon_coles
  treatment_kind: dixon_coles
  tables: [match_odds]
  features: []
  diff:
    - "components.score.params.market_prior_strength: 0 → 1.0"
  notes: >
    Control dc_layered_v2 (key absent, default 0). Treatment
    dc_layered_odds_v1 at 1.0. Read is strict PIT. Closing-proxy
    kickoff rows are a labelled side rate, not the gate.

control: dc_layered_v2
treatment: dc_layered_odds_v1

window:
  seasons: ["2024-25"]
  from_gw: 1
  to_gw: 38
  seasons_back: 4
  modes: [walk_forward]

suites:
  prediction: true
  components: true       # P3: --score-models dixon_coles_v2 dixon_coles_odds_v1 --skip-sims
  decisions: false
  steps: 1

gates:
  min_coverage: 0.80
  allow_low_coverage: false
  coverage:
    metric: odds_join
    read: strict_pit
  keep_if:
    prediction_any: [spearman, mae]
    components_any: [rps, logloss]
  drop_if:
    coverage_below_min: true
    mean_ppg_gate_fail: true

report:
  title: "DC market prior (W7)"
  store_doc: dc-odds-prior-eval.md
```

`read: closing_proxy_labelled` is legal only when the treatment spec sets `use_closing_proxy: true`. On `dc_layered_odds_v1` today that key is absent, so P1 fails a proxy read as a mismatched clock.

### 5.2 `penalties_order` — opt-out

On snapshot rows, null means “not on the pecking order”. On vaastav, null means “we never had the column” (`source=vaastav`, design §4.5). An 80% non-null gate would fail a healthy snapshot season, because most players are legitimately null. The gate is a zero-detector.

```yaml
name: pens_order_involvement_ab
kind: config
description: "involvement ± penalties_order (W8). Opt-out."

change:
  lever: hyperparam
  orientation: opt_out
  slot: involvement
  control_kind: involvement_xg
  treatment_kind: involvement_xg
  tables: [player_status]
  features: []
  diff:
    - "components.involvement.params.use_penalties_order: false → true"
  notes: >
    Control dc_layered_nopens_v1. Treatment dc_layered_v2 (default true).
    Window must be snapshot-backed. Vaastav non-null rate is 0 and the
    arms tie via the penalties_missed fallback.

control: dc_layered_nopens_v1
treatment: dc_layered_v2

window:
  seasons: ["2025-26"]    # replace with the season that actually has fpl_api snapshots
  from_gw: 1
  to_gw: 38
  seasons_back: 4
  modes: [walk_forward]

suites:
  prediction: true
  components: true        # P3: involvement_xg_nopens_v1 vs involvement_xg_v1
  decisions: false
  steps: 1

gates:
  min_coverage: 0.01
  allow_low_coverage: false
  coverage:
    metric: penalties_order_non_null
    read: strict_pit
  keep_if:
    components_any: [mae_goal_share]
    prediction_any: [mae, spearman]
  drop_if:
    coverage_below_min: true
    mean_ppg_gate_fail: true

report:
  title: "penalties_order involvement lift (W8)"
  store_doc: penalties-order-involvement-eval.md
```

`0.01` fails an all-null vaastav window and passes a season where FPL filled a real pecking order. The report also prints the `penalties_order == 1` rate and the share of status rows with `source=fpl_api`. Those two are not the gate. The file on main is the same arms with `seasons: ["2024-25"]` and no coverage block; P1 should move the window onto snapshot gameweeks or the preflight will fail, which is the correct outcome.

### 5.3 Defensive contribution — opt-out, season filter

Rules award defensive contribution only where `configs/scoring/<season>.yaml` has a non-empty `defensive_contributions` list (2025-26 and 2026-27). Earlier seasons make `use_defensive_contribution` a no-op. W9’s award rate was about 12% of played points (20.8% of DEF appearances). That rate is the interpretation. Gating on it would drop a real lever. `eval-components` has no other-points metric, so the suite stays off until one exists.

```yaml
name: defcon_other_2025_26
kind: config
description: "other_player ± defensive_contribution (W9). Opt-out. 2025-26 only."

change:
  lever: hyperparam
  orientation: opt_out
  slot: other
  control_kind: other_player
  treatment_kind: other_player
  tables: [player_fixture_stats]
  features: []
  diff:
    - "components.other.params.use_defensive_contribution: false → true"
  notes: >
    Control dc_layered_nodefcon_v1. Treatment dc_layered_v2 (default true).
    seasons_back 4 is fine: pre-2025 training rows contribute 0 DC by rules.
    The gate is the test window, not the train window.

control: dc_layered_nodefcon_v1
treatment: dc_layered_v2

window:
  seasons: ["2025-26"]
  from_gw: 1
  to_gw: 38
  seasons_back: 4
  modes: [walk_forward]

suites:
  prediction: true
  components: false
  decisions: false
  steps: 1

gates:
  min_coverage: 1.0
  allow_low_coverage: false
  coverage:
    metric: dc_season_eligible
    read: strict_pit
  keep_if:
    prediction_any: [mae, rmse, spearman]
  drop_if:
    coverage_below_min: true
    mean_ppg_gate_fail: true

report:
  title: "Defensive contribution (W9)"
  store_doc: defensive-contribution-eval-2025-26.md
```

`min_coverage: 1.0` means every target fixture sits in a season whose scoring YAML awards DC. Mixing in 2024-25 fails the preflight.

---

## 6. Coverage

One formula per `gates.coverage.metric`. The YAML does not redefine the numerator. Denominators are fixtures or player–fixtures in the test window, never gameweeks (blanks add nothing; doubles are two fixtures).

| `metric` | Denominator | Numerator | `min_coverage` | Also print, do not gate |
|---|---|---|---|---|
| `odds_join` | Fixtures the arm predicts | 1X2 non-null after the **model’s** join, on the read in `coverage.read` | `0.80` | Closing-proxy join %; fixture_id-only join % |
| `odds_history_join` | Training-window fixtures, weighted by the Dixon–Coles time-decay weight | 1X2 non-null via a date-safe `fixture_id` join | `0.80` | Target-fixture strict join % (not the gate for `market_prior_source: history`) |
| `penalties_order_non_null` | Player–fixtures scored | Latest PIT `player_status.penalties_order` non-null at that deadline | `0.01` | `== 1` rate; share of rows with `source=fpl_api` |
| `dc_season_eligible` | Target fixtures | Season has non-empty `defensive_contributions` in scoring YAML | `1.0` | Award rate among `minutes > 0`; share of played rows with the staged column null |

Odds join keys, in order: `fixture_id`, else `season + team_h_key + team_a_key + match date`. Teams alone are not a key (`odds_poisson` already rejects that; `_align_odds` in `dixon_coles.py` still falls through to teams). P1 measures both rates. If the team-key rate exceeds the fixture_id rate by more than 5 percentage points, **warn**. The runner does not patch the join.

`allow_low_coverage: true` turns a coverage miss into a warning and forces `inconclusive`. It never produces `keep`. Use it only to see a no-op in the tables.

`change.tables` is checked whenever it is non-empty, including on `kind: config`. Each table must exist under staged storage and must have `available_at`. That check is P1; the dry-run on main does not open the store.

---

## 7. PIT

Predicted fixtures use one clock: `PITView` keeps rows with `available_at <=` the gameweek deadline. A feature row for a fixture being scored whose timestamp is after that deadline **fails the run** (exit 2), even when coverage is high. Promote-eval does not silently call `closing_proxy_match_odds()` for models that use `latest`.

Past closes are already on that clock. A finished match has `available_at = kickoff`, and that kickoff is before the next deadline, so `PITView.latest` returns the row. The history prior (§5.1) is the model that reads those rows. `market_prior_source: history` is the same switch as `odds_history: pre_kickoff`. The component YAML sets it. The runner does not flip it, and it does not invent a second cutoff. The target-fixture blend does not read them.

When `change.tables` includes `match_odds`, or any joined row has `odds_kind=closing_proxy`, the plan and `recommendation.md` print this caveat plus both rates, including when the run then fails:

> Closing lines are stamped `available_at=kickoff`, after the FPL deadline. This run did not move the PIT cutoff. Strict-PIT odds join = {strict:.1%}. Closing-proxy join (labelled; used only by specs with `use_closing_proxy: true`) = {proxy:.1%}. market_prior_source = {market_prior_source}.

Football-data has no `captured_at`. A deadline-honest **target-fixture** prior needs a snapshot taken at or before the deadline, from a named book. Until that feed exists the strict-PIT `odds_join` gate fails, and the closing-proxy rate says whether the file itself joined. The history prior uses `odds_history_join` instead, and can run on the closes already staged.

A model change that wants the close on the fixtures being predicted sets `use_closing_proxy: true` on that spec, same as `odds_poisson_v1`, and the EvalSpec sets `coverage.read: closing_proxy_labelled`. The caveat still prints. The leakage test still owns deadline safety. The report must not call that run a deadline result.

---

## 8. Preflight: fail vs warn

Exit `2` on fail, before any fit. Exit `1` later only for a coverage `drop` or a decisions mean-PPG gate. Exit `0` for `inconclusive`, `tune`, and `keep`.

**Fail**

| Check | Why it is fatal |
|---|---|
| Spec name missing, or `control == treatment` | No A/B |
| `control_kind` / `treatment_kind` disagree with the loaded YAML (top-level, or `components[slot]`) | The diff is a story |
| `orientation` disagrees with effective params, or both arms share the effective value | Inverted or empty contrast. Absent bool → `true`; absent `market_prior_strength` → `0` |
| Another slot’s `kind` or params differ, and `allow_multi_lever` is false | CACE. W8 and W9 must stay separate even though both treatments are `dc_layered_v2` |
| Listed table missing, or no `available_at` | The consumer cannot be PIT-safe |
| `kind` not in `registered_kinds()` / `registered_components()` | Contract suite never saw it |
| Coverage &lt; `min_coverage` and `allow_low_coverage` is false | The lever did not run. Odds-on-closing-lines fails here on purpose |
| A predicted-match feature is timestamped after the deadline | Leak. Print the coverage rates, then exit 2 |
| `coverage.read` disagrees with the treatment spec (`closing_proxy_labelled` without `use_closing_proxy: true`) | The gate would count rows the model never sees |
| `kind: config` diff touches more than the declared param family | Same as multi-lever |
| Spec and environment both differ, and `allow_multi_lever` is false | Two levers |
| `kind: environment` and the resolved spec hashes differ | Not an environment lever |
| `kind: environment` and the lockfiles are identical | No contrast |
| `kind: environment` and a changed **direct** dependency is missing from `change.packages` | The diff is a story |
| `kind: refactor` and the specs or the lockfile differ | Not a refactor |
| `extends` target missing, or a chain deeper than one | Unresolvable fork |
| Effective diff of the resolved specs ≠ `change.diff` | The diff is a story. Applies once `extends` exists (P1.5) |
| Dirty tree without `--allow-dirty` | Not reproducible |
| Cached ledger whose stored key ≠ the key just computed | Stale cache (P2.5) |

**Warn, and continue**

| Check | What the operator still does |
|---|---|
| Tier S0–S2 with `suites.decisions: true` | Decisions are forced off (already implemented) |
| Closing-proxy rows exist | Print the caveat and both rates. Existence alone is not a fail |
| Team-key odds rate exceeds fixture_id rate by &gt; 5pp | Fix stays in the model |
| Printed `eval-components` line has no arm-specific model flags | Unresolved arm YAML / missing `components.<slot>.name`; the line is not an A/B |
| Vaastav `source` on a minutes slice | Report that slice separately; pens still fails if the non-null rate is ~0 |
| Played rows with null `defensive_contribution` &gt; 20% inside an awarding season | Column is degraded. Season eligibility can still pass |
| Optional shared table empty on both arms (Understat missing in W9) | Not a differential |
| `kind: model` on a single season | Fine for S1. Decisions claim still needs a later tier |
| Pooled `top10_precision` listed in `keep_if` | Print it; do not let it decide |
| Transitive dependency changes on an environment lever | List them. Direct omissions still fail |
| Several EvalSpecs, same control, same lever, one window | Print the selection caveat. `keep` needs a fresh window (§9) |
| Learner swap with non-deterministic threading (`deterministic: false` and similar) | Report the seed-to-seed spread |

Scaffold files `example_config_dc_v1_vs_v2.yaml` and `example_component_minutes_v1_vs_v2.yaml` compare `dc_layered_v1` with `dc_layered_v2` (several slots). They exist so the loader has something to resolve. P1 sets `allow_multi_lever: true` on those two names. A real minutes swap copies `dc_layered_v2` and changes one slot.

---

## 9. Metrics, commands, artifacts

Reuse `compare`, `eval-components`, and `decisions`. No forked metric math.

Produce and compare are separate:

```text
produce_ledger(arm, window, seed) -> ledger.parquet
compare(ledger_control, ledger_treatment, suites) -> summary + recommendation.md
```

`fplt compare` keeps its CLI. Internally it becomes produce twice, then compare. Cache each ledger in a content-addressed directory keyed by `(resolved spec hash, uv.lock hash, staged-data hashes, window, seasons_back, n_sims, seed)`. Store the key beside the parquet and verify it on read. A mismatch fails preflight. This is parquet on disk, not `joblib.Memory`. P2.5. The control is usually production, so most EvalSpecs reuse it. Environment arms run each produce in its own subprocess, under `uv sync --frozen`, in a worktree at that arm's `lock_ref` (P7).

**Paired seeds.** `seed_for(model_id, as_of, fixture_id)` stays the single-model stream: SHA-256 of that triple, one `Generator` per fixture, then scorelines, every player's minutes in `player_key` order, and the match simulator. Two forks do not share it, because `model_id` is in the key, and one extra player shifts every later draw. Promote-eval pairing uses an eval seed that does not include `model_id`, and a player stream spawned from `(seed, season, gw, fixture_id, player_key)` (`numpy.random.SeedSequence`). That is its own PR. It does not rewrite `seed_for`. Acceptance: identical resolved spec and environment produce bit-identical ledgers. A NumPy or RNG upgrade that changes streams is "different, not worse" (§3.2).

**Intervals.** For each primary metric, `d_t` is the per-gameweek mean loss of the control minus the loss of the treatment, signed so positive means the treatment is better. The unit is the gameweek, so doubles and blanks fall out. Report:

- Diebold–Mariano / Giacomini–White: `statsmodels` OLS of `d_t` on a constant, `cov_type="HAC"`. Mean, HAC standard error, p-value. With a refit every deadline this tests the method as run, estimation noise included (Giacomini and White, Econometrica 2006; Diebold and Mariano, JBES 1995).
- A 95% interval from `arch.bootstrap.StationaryBootstrap` on that same gameweek series (default 2,000 resamples, fixed seed). Thirty-eight gameweeks is a small sample for the HAC limit (Politis and Romano, JASA 1994).
- Per-GW win counts, as now.
- A position slice, with its own interval, when the lever is position-specific.

These columns are descriptive for improvement levers until FDB-73. They may `drop` an equivalence lever (§3.2). When several arms share a window, also report `arch.bootstrap.SPA` ("does any arm beat the control?") and `arch.bootstrap.MCS` (arms not separable from the best). `keep` still comes from a confirmation window. Implemented in `fplt.backtest.intervals` via `statsmodels` and `arch`; the runner does not reimplement Newey–West or a stationary bootstrap.

P2 prediction command is already the right shape: `fplt compare --models <control> <treatment>` with the shared window. P2 does not override `n_sims`. W9 at `n_sims: 10000` was about three minutes an arm. A cheaper screen is a pair of specs that both set the same lower `n_sims`, declared in `change.diff` only when `n_sims` is the lever.

P3 components command must name the slot specs it reads off each parent (`components.<slot>.name`), pass `--skip-sims` unless `n_sims` is the lever, and stop calling the defaults (`dixon_coles_v1`, `empirical_minutes_v1`, `involvement_shares_v1`). The dry-run on main omits those flags, so it does not compare the two arms.

S0 load + preflight. S1 clamps the window to GW1–10. S2 to GW1–20. S3 adds decisions on GW1–10. S4 decisions on the full window. S5 multi-season decisions stays outside v1. Dev season for in-house models remains 2024-25, except live-only fields (defcon, snapshot `penalties_order`), which set `window.seasons` explicitly. OpenFPL still starts at 2024-25.

```text
$FPLT_DATA_DIR/promote_eval/<eval_name>/<stamp>/
  plan.json
  audit.json
  summary.json
  summary.parquet
  per_match.parquet       # scoreline levers: one row per fixture × arm × metric
  spec_diff.json
  recommendation.md
```

`audit.json` is the reproducibility record. It does not need a tracking server.

| Field | How |
|---|---|
| `git_sha`, `git_dirty` | Dirty tree fails unless `--allow-dirty`. That flag forces `inconclusive` and is printed on line 1 of `recommendation.md` |
| `uv_lock_sha256` | Hash of the lockfile the arm actually ran under. The two arms differ only for `kind: environment` |
| `python`, `os`, `fplt_version`, `argv` | Version strings and the command line |
| `data` | Per staged parquet actually read: SHA-256, row count, max `available_at` |
| `config` | SHA-256 of the resolved ModelSpec per arm, after `extends` |
| `seed`, `n_sims` | Eval seed for pairing. Single-model draws still use `seed_for` |
| `folds` | `{season, gw, as_of, n_train, n_test}` |
| `market_prior_source` | The score-component param, when the lever reads odds |
| `wall_clock_s`, `peak_rss` | Per arm. Reported for learner swaps and upgrades. Never a gate |

Same SHA, data hashes, and config must reproduce per-match metrics bit for bit. Seeded simulation is included; an unseeded draw fails the audit check.

Scoreline levers (`slot: score`, or `change.tables` contains `match_odds`) also write:

- The **raw de-vig 1X2** row on the same clock as `coverage.read`. Proportional devig unless the arm's spec says otherwise. `odds_poisson_v1` is a model column, not this row.
- Per-gameweek paired deltas versus the control and versus that market row, with the HAC interval and the stationary-bootstrap interval from above.
- Breakdowns by season. This is a report, not an automatic `keep`, until FDB-73.

Player metrics stay on per-GW win counts, mean xPts versus mean actual, and position slices. A bootstrap CI does not replace the W9 reading (MAE and RMSE can disagree).

`recommendation.md` always has these lines, in this order:

1. Title, `question`, question type (`improvement` or `equivalence`), `kind`, `orientation` (improvement only), lever, `change.diff`. Dirty-tree flag on this line when `--allow-dirty` was set.
2. Arms (spec name, effective flag, which one is production, lockfile hash when the lever is `environment`). A dose lists every arm. The call does not promote the winner of this window.
3. Audit: git SHA, dirty flag, `uv.lock` hash, data hashes, resolved-spec hash, wall-clock, peak RSS. Full record is `audit.json`.
4. Window: seasons, GWs, `seasons_back`, mode, `n_sims`. Folds live in `audit.json`.
5. Coverage: metric name, numerator, denominator, rate, `min_coverage`, pass or fail.
6. Join key and the `available_at` rule actually used. `market_prior_source` when the lever reads odds.
7. PIT caveat, when §7 says to print it, with both odds rates.
8. Deltas: **RMSE** (headline), MAE (continuity), Spearman, per-GW win counts on RMSE (MAE wins listed beside), mean predicted xPts vs mean actual, HAC interval, bootstrap interval, signed so positive means the treatment is better. Scored on the decision-relevant pool. Position slices when the lever is position-specific (defcon). Scoreline levers add the raw de-vig row. Several arms on one window add SPA and the model confidence set (RMSE losses). W9-style MAE-worse / RMSE-better splits: prefer RMSE for mean-seeking calls.
9. Component deltas, when that suite ran.
10. `decisions: not run` when the suite is off. This is a complete promo for data and config.
11. Equivalence only: predictions compared, count beyond tolerance, max absolute diff, top 10 offenders.
12. Call (`keep` / `tune` / `drop` / `inconclusive`) and one paragraph. Improvement P2 may only emit `drop` (coverage or a leak) or `inconclusive`. Equivalence uses §3.2.

---

## 10. Keep / drop until FDB-73

1. Preflight fail → exit 2. No recommendation file required.
2. Coverage under `min_coverage` → `drop`, unless `allow_low_coverage` (then `inconclusive`).
3. Otherwise P2 writes **`inconclusive`** and the delta table. It does not apply `keep_if`.
4. After FDB-73 publishes the three hand results, a follow-up may turn `keep_if` into a rule. W9 says that rule needs RMSE, bias, and per-GW counts, not MAE alone. No ε in the P1+P2 PR.
5. Decisions mean-PPG above `max_mean_ppg` → `drop`, and only when that suite ran.
6. Component slot metric gates xPts `keep` once P3 exists. An xPts-only win with a worse slot metric is `tune`.
7. Quality numbers stay out of pytest.
8. Equivalence (§3.2): identical predictions → `keep`; an interval entirely on the bad side → `drop`; otherwise `inconclusive`. That rule does not promote an improvement lever.

---

## 11. Ship order

| Next | What lands | What stays out |
|---|---|---|
| **P1+P2** | Preflight in §8 for the levers that exist today: orientation, one lever, coverage, leak, missing spec. `question`. `audit.json` with the §9 fields that do not need `extends` or the cache. `--allow-dirty` forces `inconclusive`. Compare both arms. `recommendation.md`. | `statsmodels` / `arch`. Paired-seed change. `extends`. Numeric `keep_if`. Decisions. Environment subprocesses. |
| **CRN** | Eval-only paired streams (§9). `seed_for` unchanged. Bit-identical ledgers when the resolved spec and the environment match. | Landing it inside the P1+P2 PR. |
| **Stats** | Done: HAC and stationary-bootstrap columns on the per-gameweek series (`fplt.backtest.intervals`). SPA and MCS when several arms share a window. Descriptive for improvement levers; equivalence may `drop` on an entirely-bad interval. | Using an interval as a `keep`. |
| **P1.5** | `extends` / `overrides` on the ModelSpec loader. Deep-merge a dict, then the existing pydantic model. One parent. Migrate `dc_layered_nopens_v1`, `dc_layered_nodefcon_v1`, and `dc_layered_odds_v1`. Resolved hash equals the old copy. No model run. | OmegaConf. Chains. |
| **P2.5** | Done: `produce_ledger` and the content-addressed parquet cache under `ledger_cache/<digest>/`. `fplt compare` CLI unchanged (produce twice, then summarise). | `joblib`. |
| **P3** | Done: components command names the two slot specs; `--skip-sims` by default. Scoreline levers: `per_match.parquet` and the raw de-vig row. | Using `odds_poisson_v1` as the book. |
| **P4** | Decisions suite. Still opt-in. Data and config promos do not wait for it. | — |
| **P7** | Done: `kind: environment` and `kind: refactor`, including the §3.2 verdict. Worktree, `uv sync --frozen`, subprocess produce per arm; ledger identity via `assert_frame_equal` then intervals. | `nox`, until more than one such eval exists. An unlisted `uv lock --upgrade` as a single lever. |
| **Bakeoff** | Done: thin multi-arm wrapper on compare / promote-eval. `fplt bakeoff` + `BakeoffSpec` (≥3 ModelSpecs, shared window, SPA/MCS + **RMSE** rank → `ranking.md`; MAE still in the metric table). EvalSpec optional `arms:` for a dose on the promote-eval compare. Shared ledgers via the content-addressed cache. Optional `pairwise_evals:` compose N promote-eval runs. | Running the Deep/PPDA + odds championship itself (FDB-99). A second framework. Emitting `keep` from the ranking window. |
| **History prior** | `market_prior_source: history` as specified in §5.1. Then one EvalSpec. A dose on that window is descriptive. | Re-running the predict-time λ blend as that study. Optuna inside the runner. Blocking this PR on `match_odds_deadline`. |
| **Tuning** | After P1.5. Optuna on a dev window that does not overlap the eval window. It writes an `extends` fork. Confirmation is a fresh EvalSpec. | Searching over which component or feature exists. |
| **Deadline odds** | A staging PR only after a book and a capture time exist. | A fetcher aimed at football-data closing files. |
| **Thresholds** | After FDB-73. The intervals are an input. | Inventing ε from W9 alone. |
| **P5** | Commit the recorded odds spec and `defcon_other_2025_26`. Extend the pens EvalSpec with coverage. | Running them is not a blocker for P1+P2. |
| **P6** | None. No Dagster job in v1. | Hydra, a second `src/fpl` tree, MLflow. |

P1 and P2 ship together: a compare run that skips the vacuous-A/B check will happily score two identical predictions. The scaffold examples keep loading; `allow_multi_lever` is how.

---

## 12. Sharp edges on main

- **The landed λ blend is a no-op on ingested closing lines.** `view.latest` hides kickoff timestamps. A labelled `use_closing_proxy` flag is a model change. The runner must not call the escape hatch on the model's behalf.
- **The landed odds prior is the wrong next experiment.** Predict-time λ blend does not test "does the book help?". The next model is `market_prior_source: history` (§5.1). Raw de-vig 1X2 is the bar. `odds_poisson_v1` is not.
- **Fork YAMLs are full copies.** `dc_layered_nodefcon_v1` repeats `dc_layered_v2` to flip one flag. `extends` (P1.5) is how new forks are written. Migrating the three existing copies is a hash check, not a model run.
- **`seed_for` includes `model_id` and is shared across players in a fixture.** A pen or defcon delta mixes the lever with a different Monte Carlo stream, and one extra player shifts later draws. Pairing is the CRN PR (§9). Do not change `seed_for` to get there.
- **`_align_odds` can attach another fixture’s line** when `fixture_id` is missing, because the fallback key drops the date. Coverage reports both rates and warns. The repair belongs in the model.
- **`eval-components` in the dry-run is not the A/B.** Defaults are `dixon_coles_v1`, `poisson_score_v1`, `odds_poisson_v1`, `empirical_minutes_v1`, `involvement_shares_v1`.
- **`example_data_odds_prior.yaml` is the wrong `kind`.** W7 is `config`.
- **v1-vs-v2 examples are multi-lever.** They are loader fixtures.
- **Pens and defcon both treat `dc_layered_v2` as treatment.** One EvalSpec cannot move both flags.
- **Pooled top-10 and a single MAE hide a `tune`.** W9: without-defcon won MAE on 34 of 38 gameweeks; with-defcon won Spearman on 22; bias improved.
- **`n_sims: 10000` is part of the spec.** P2 runs it. Override only with two specs that share the new value.

---

## 13. Use a library when it already has ~90%

Preference: if an existing library already does about 90% of this job, use it. Scored against the promote-eval job, nothing external clears that bar. The code that does clear it is already in this repo.

| Piece of the job | Share | Where it lives |
|---|---:|---|
| Walk-forward refit and predict, both arms, one window | ~40% | `fplt compare` |
| Slices by GW, position, horizon | ~15% | compare metrics + the parquet ledger |
| Component metrics (RPS, minutes Brier, involvement MAE) | ~10% | `eval-components` (P3 still has to pass the two slot names) |
| Squad replay | ~10%, opt-in | `fplt decisions` |
| Preflight: orientation, one lever, coverage, PIT read | ~15% | not written yet; no library has it |
| `recommendation.md` | ~5% | not written yet |
| A UI that lists runs | ~5% | the promote-eval directory; Streamlit already reads the ledger |

P1+P2 is a thin caller of `compare_models`, plus the preflight. It does not reimplement MAE. That is the 90% rule applied to code we already maintain. The rest of this section is what you would still have to write if a library were the runner.

### 13.1 MLflow evaluate — closest external API

[`mlflow.models.evaluate`](https://mlflow.org/docs/latest/ml/evaluation/) is the nearest fit. It takes a fitted pyfunc URI, one dataframe, and a target column. Builtin regressor metrics are MAE, MSE, RMSE, R², and MAPE. `extra_metrics` / `make_metric` add scalars. `baseline_model` plus `validation_thresholds` (`threshold`, `min_absolute_change`, `min_relative_change`, `greater_is_better`) compare a candidate to a baseline. Artifacts land on the tracking run. Model Registry aliases (`@champion`) are a separate deploy pointer.

What it does not do:

- **No walk-forward.** One `data` argument, one predict. A gameweek loop would be our code, with `evaluate` called at the bottom as a logger.
- **No refit.** The pyfunc is already fitted. `dc_layered_*` refits every deadline on a `PITView`.
- **No deadline.** The frame is whatever rows the caller put in it. `available_at` is invisible.
- **No FPL slices.** Spearman within a gameweek, top-10 precision, Brier on `p_haul` / `p_60`, RPS, clean-sheet Brier, minutes Brier, and involvement MAE are not builtin. A custom metric can return more scalars; those functions already live in `fplt.backtest.metrics`.
- **Thresholds only know “candidate beats baseline”.** There is no `tune`, and no rule for the W9 split (MAE worse, RMSE better, bias better). ε is also deferred until FDB-73, so the validation feature is the part we would leave off.
- **Baseline metrics are second-class.** They come back on `EvaluationResult.baseline_model_metrics`. The logged artifacts are the candidate’s.
- **The model shape does not match.** `PointsPredictor.predict(view, fixtures)` is not a feature matrix in, vector out. The pyfunc adapter is the real work, and it would be a pandas boundary. Polars stays the rule everywhere else.
- **A registry alias is a second production list** beside `PRODUCTION_MODELS`.

About **15%** of the job if the adapter were free. Under **10%** once the gameweek loop and the adapter are counted as ours. M10 already removed the `mlflow` dependency, `mlruns/`, and `mlflow.db`. P1+P2 does not add them back.

Weights & Biases and DVC (`dvc exp diff`) are the same leftover **5%**: a place to file the summary. They add a hosted account or a second experiment table on top of `summary.parquet`.

### 13.2 Deepchecks and Giskard — two models, one frozen test set

[Deepchecks `ModelComparisonSuite`](https://docs.deepchecks.com/stable/tabular/auto_checks/model_evaluation/plot_multi_model_performance_report.html) runs checks on two or more **already fitted** scikit-learn estimators, each with a train `Dataset` and a test `Dataset`. `MultiModelPerformanceReport` defaults to negative RMSE, negative MAE, and R².

Missing: per-deadline refit, a gameweek axis (a slice is a column filter on that one test set), `predict(view, fixtures)`, Spearman-by-GW, top-10, Brier, RPS, coverage, and `available_at`. Its train/test checks look for mixed rows, not a deadline. The default scorers crown the lower MAE, so the W9 table would pick `dc_layered_nodefcon_v1` and drop the wire the hand eval kept. Pandas `Dataset`. About **10–15%**, after we had built the frame ourselves.

[Giskard’s suite](https://docs.giskard.ai/) reruns tests you pass a model into. The shipped tests are accuracy, drift, and perturbation scans on one dataframe. The runner is pytest-shaped. Coverage, PIT, and walk-forward would still be our tests. About **5%**.

### 13.3 Every other candidate, and the hole

| Library | What it covers | What is missing |
|---|---|---|
| **Evidently** Test Suite ([tests](https://docs.evidentlyai.com/docs/library/tests)) | Pass / Fail / Warning (`is_critical=False`) / Error on a frame that already has predictions and a target. Regression preset: MAE, RMSE, mean error. | Does not fit or walk. Two arms means we concat a model column first. No RPS, minutes Brier, or Spearman-by-GW unless we write them. No odds-join coverage. Using it after `compare` recomputes numbers the ledger already stores. ~10%. |
| **TFX Evaluator** + Pusher ([guide](https://www.tensorflow.org/tfx/guide/evaluator)) | Candidate vs blessed baseline, `change_threshold`, slice specs, a blessing artifact that gates the push. | TF Examples and a SavedModel. No refit loop. Thresholds mean “beat the baseline”, which is opt-in only. A missing baseline is rubber-stamped. No FPL metrics. The dependency is larger than the orchestrator. ~10% of the gate, ~0% of the runner. |
| **Feast** [`get_historical_features`](https://docs.feast.dev/getting-started/concepts/point-in-time-joins) | Point-in-time join on `event_timestamp`, a TTL, and an opt-in `created_timestamp <= entity_timestamp`. | The entire eval. The default join is event time, so a backfill stays visible until that flag is on. Our clock is `available_at <= deadline`, and `PITView` already is that clock. Swapping it is a data-layer migration, not this project. |
| **Tecton** | The same feature-store pitch, plus monitoring. | The same hole as Feast, plus a hosted platform. |
| **Great Expectations** | Expectations on a table you hand it. | Does not run the model’s odds join or the scoring-YAML season check. We would compute the rate, then assert it. Pandera already gates schemas. |
| **SageMaker / Vertex model registry** | Champion/challenger alias and an approval step in front of an endpoint. | The challenger is assumed to be the new model. For pens and defcon, production is the treatment. The split is online traffic, not a squad replay. |
| **Kubeflow, ZenML, Metaflow, Kedro** | DAGs. | Dagster is already the orchestrator, and v1 of this command is not a job. No `ModelSpec`, no `available_at`. |
| **whylogs / WhyLabs, NannyML, Arize Phoenix** | Profiles, drift, label-free performance estimates, LLM traces. | We have actuals and a paired refit. Different product. |
| **sklearn `TimeSeriesSplit`** | An expanding or rolling index splitter. | No deadlines, no `PITView`, no FPL metrics. Our walk-forward is the stricter version of the same idea. |
| **socceraction, OpenFPL, penaltyblog, vaastav, football-data.co.uk** | VAEP, a trained FPL ensemble, a Dixon–Coles fitter, a historical dump, closing prices. | None of them is a promote step. OpenFPL stays a 2024-25+ benchmark. |

Papers behind checks we keep in our own runner:

- **Breck, Cai, Nielsen, Salib, Sculley**, “The ML Test Score”, IEEE Big Data 2017. Coverage is a blocking data test. Metric ε waits for FDB-73.
- **Sculley et al.**, “Hidden Technical Debt in Machine Learning Systems”, NeurIPS 2015. One lever, or preflight fails.
- **Kaufman, Rosset, Perlich**, “Leakage in Data Mining”, ACM TKDD 2012. A feature is legitimate when it is knowable at the decision time. That time is the deadline.
- **Bottou et al.**, “Counterfactual Reasoning and Learning Systems”, JMLR 2013. Inverse-propensity methods are for policies you cannot rerun. We can refit both specs.
- **Decroos, Bransen, Van Haaren, Davis**, “Actions Speak Louder than Goals”, KDD 2019. Score the component that has a label before trusting the composite. Same reason a worse MAE can still be `tune`.
- **Dixon and Coles**, JRSS-C 1997, and **Štrumbelj**, International Journal of Forecasting 2014. The scoreline model and the de-vig. `odds_to_lambda` already is the de-vig.
- **Diebold and Mariano**, JBES 1995, and **Giacomini and White**, Econometrica 2006. Paired loss differentials, including re-estimation. This is the test §9 runs through `statsmodels`.
- **Politis and Romano**, JASA 1994. Stationary bootstrap, via `arch`.
- **White**, Econometrica 2000; **Hansen**, JBES 2005; **Hansen, Lunde, and Nason**, Econometrica 2011. Reality Check, SPA, and the model confidence set, for a dose compared on one window.

### 13.4 Recommendation

Do not add an external library as the promote-eval runner. Call `compare`, `eval-components`, and `decisions`.

Do add `statsmodels` and `arch` for the paired intervals and for SPA / MCS (§9). That is the 90% bar applied to the statistics, which is a different job from walking deadlines. Neither package is a runner, a metric store, or a config system.

That runner preference flips if a library walks a deadline calendar, refits a caller-supplied model each step, scores two arms on that window, and accepts a coverage fraction plus Spearman-by-GW as gates. None of the libraries above do those four things. Wrapping `PointsPredictor` so one of them can score a frame we already built with `PITView` is more code than the orchestrator, and the clock ends up outside the library.

Copy these habits into the thin runner, without those dependencies: Evidently’s critical vs non-critical status (§8), TFX’s blessing file in front of the production edit, Feast’s printed as-of and unmatched count (the coverage block), and VAEP’s component-before-composite order. Deep-merge forks with a dict and pydantic. Do not add OmegaConf. Cache ledgers as parquet. Do not add `joblib`.

---

## 14. Operator card

```text
1. Off state and on state are the two ModelSpecs. Production stays the one
   it already is. Fork the other. Never edit production YAML in place.
2. configs/evals/<name>.yaml — question, one lever, orientation, coverage metric.
3. make check (new kinds need the leakage suite).
4. fplt promote-eval --eval <name> --tier S1
     preflight fails a vacuous A/B, a leak, or a dirty tree
     --allow-dirty still runs, and the verdict is inconclusive
     compare writes deltas
     scoreline arms also write the raw de-vig row and the HAC / bootstrap intervals
     improvement recommendation is inconclusive, or drop on coverage
     equivalence recommendation follows the identical / not-worse / worse rule
5. Decisions only if the claim is the squad (kind: model, or an explicit opt-in).
6. Production change is a later PR. keep on an opt-out means "leave it on".
   A dose on one window does not promote. Confirmation is a fresh window.
7. Multi-arm championship / dose ranking:
     fplt bakeoff --bakeoff <name> --execute
       or: fplt bakeoff --models a b c --control a --season … --execute
       or: EvalSpec arms: […] on promote-eval (same compare + SPA/MCS)
     ranking.md is descriptive — not a keep
     decision season totals stay on fplt decisions

ODDS     recorded arm: control dc_layered_v2, treatment dc_layered_odds_v1
         gate: strict-PIT odds join >= 80%. Closing kickoff lines fail this.
         benchmark row: raw de-vig 1X2. odds_poisson is only a model column.
         next model: market_prior_source history. Gate is odds_history_join.
PENS     control dc_layered_nopens_v1, treatment dc_layered_v2
         gate: penalties_order non-null rate >= 1% (zero-detector, not 80%).
DEFCON   control dc_layered_nodefcon_v1, treatment dc_layered_v2
         gate: 100% of target fixtures in a season whose rules award DC.
ENV      same spec, two lockfiles. Verdict from prediction identity, then intervals.
BAKEOFF  ≥3 ModelSpecs, shared window; SPA/MCS + RMSE rank (MAE secondary). Not a keep.
```

---

## 15. Out of scope for the runner

- Choosing a treatment value inside the runner. Optuna on a disjoint dev window is the Tuning row in §11. A dose compared on the eval window is descriptive.
- Cartesian grids, and search over which component or feature is included.
- OmegaConf, Hydra, a parallel `src/fpl` tree, MLflow, and `joblib.Memory`.
- Editing `PRODUCTION_MODELS`.
- Letting the runner invent a closing-odds cutoff, or fetch a deadline snapshot before a book and a capture time exist.
- Rewriting `seed_for`. Paired streams are an eval seed (§9).
- Numeric keep/drop thresholds for improvement levers before FDB-73.
- An external eval library as the runner (§13). The metric loop stays `compare` / `eval-components` / `decisions`. `statsmodels` and `arch` are the interval layer only.
- Network in tests. Quality asserts in pytest.
- Replacing the Streamlit leaderboard.

---

## 16. Decision log

| Choice | Why |
|---|---|
| Off vs on, production stays put | Pens and defcon are already on. Odds is off. One sentence covers both. |
| `orientation` checked against effective params | Absent `use_*` is true. Absent `market_prior_strength` is 0. |
| Coverage formula named by the metric | Stops an 80% gate on a pecking order, and a 12% gate on defcon awards. |
| Odds gate is strict PIT | The model under test cannot see kickoff stamps. Failing the eval is the honest result. Print the closing-proxy rate anyway. |
| Predicted fixtures vs rating history | A post-deadline row on a fixture being scored fails the run. Past closes are already visible at the next deadline. `market_prior_source: history` is the model that reads them. |
| Dose on one window is not a promotion | SPA and the model confidence set describe the set. `keep` is a fresh window. |
| Scoreline reports include the book | Raw de-vig 1X2, plus HAC and a stationary bootstrap on gameweeks. `odds_poisson` is a model column. Not an auto-keep. |
| `audit.json`, not MLflow | Git SHA, lockfile hash, data hashes, resolved spec, folds, wall-clock, peak RSS. `--allow-dirty` forces `inconclusive`. |
| Next odds model is the history rating prior | W7: raw closing 1X2 RPS 0.366, DC 0.334, odds-Poisson 0.363. The landed λ blend only sees the fixtures passed to `predict`. |
| Equivalence is a second verdict | Identical predictions adopt a lockfile or a refactor. An interval entirely on the bad side pins the old one. Improvement levers do not use this `keep`. |
| `extends`, one parent | The off-forks are full copies today. Deep-merge, then pydantic. Hash-check the migration. |
| Paired seeds do not change `seed_for` | The production stream includes `model_id`. The eval stream does not. |
| `statsmodels` and `arch` for intervals | Diebold–Mariano / Giacomini–White and a stationary bootstrap. Not a runner and not a metric store. |
| P2 emits `inconclusive` unless coverage drops | W9 MAE-only keep would have been the wrong call. Equivalence is the exception in §3.2. |
| Decisions opt-in for data and config | W9 did not need a bakeoff to recommend `tune`. |
| P1+P2 together | Compare without the vacuous-A/B fail scores a no-op as a result. |
| No external eval library as the runner | None of MLflow evaluate, Deepchecks, Giskard, Evidently, or TFX walks deadlines and refits. They cover a static frame. ~90% of the metric loop is already `fplt compare`. `statsmodels` and `arch` cover the paired test, which is the piece that cleared the bar. |
| Fail vs warn as in §8 | Evidently’s critical bit, applied to our checks, without the library. |

---

## 17. What the second draft changed

The Hydra / MLflow / new-`src/fpl` draft is discarded. The follow-up ("one change, one verdict") extends this file. The sections above are that extension. This section is the conflict list, so a later edit does not revive a rejected line.

Taken:

- Equivalence kinds and the three-step verdict (§3.2).
- `extends` / `overrides`, one parent (P1.5).
- Ledger split and a content-addressed parquet cache (P2.5).
- Diebold–Mariano / Giacomini–White and a stationary bootstrap, via `statsmodels` and `arch`. SPA and MCS when several arms share a window.
- The history rating prior (§5.1). Checked against the code: `market_prior_strength` blends λ only for the fixtures passed to `predict`. `fit` never reads `match_odds`.
- Provenance extras, and `--allow-dirty` forces `inconclusive`.
- Optuna only upstream of the runner, on a window that does not overlap the eval.

Left:

- `odds_poisson_v1` as the market column. The book is raw de-vigged 1X2 (RPS 0.366 against 0.363 for odds-Poisson, 2024-25 GW1–38).
- OmegaConf and `joblib.Memory`.
- CRN, `statsmodels`, and `arch` inside the P1+P2 PR.
- Replacing `seed_for`. Today it hashes `(model_id, season, gw, fixture_id)` and one generator serves every player in that fixture, in `player_key` order.
- A `match_odds_deadline` fetcher before a book and a capture time exist.
- Promoting the best of a dose from the window that ranked it. `plan.json` on main is still the dry-run command list. It does not record provenance. The three off-forks are full YAML copies. There is no `extends` loader.
