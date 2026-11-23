# Dynamic/covariate team ratings vs dc_layered_v4 (RPS / log-loss)

question: Do predict-time rest / congestion / form covariates on Dixon–Coles improve scoreline RPS and log-loss vs production dc_layered_v4?
question_type: improvement
kind: config
orientation: opt_in
lever: hyperparam
diff:
  - components.score.params.rest_coef: 0 → 0.05
  - components.score.params.congestion_coef: 0 → 0.05
  - components.score.params.form_window: 0 → 5
  - components.score.params.form_strength: 0 → 0.2

## Arms
- control: dc_layered_v4 (production) [congestion_coef=0.0, form_strength=0.0, rest_coef=0.0] model_id=1a1a9fd52db7250e
- treatment: dc_layered_covariate_v1 [congestion_coef=0.05, form_strength=0.2, rest_coef=0.05] model_id=65f462314f577b0a

## Audit
- git_sha: 4f81ccf3af0c7731d5b521d0a2b342a5f9abfacc
- git_dirty: False
- uv_lock_sha256: e9ae5f72e824c4d614230105110022eb159fb9c422befd6ad0c0ea8e26a5dd73
- full record: audit.json

## Window
- seasons: 2024-25
- gw: 1-20
- seasons_back: 4
- modes: walk_forward
- n_sims: 10000
- note: Development window (eval protocol §1). Confirmation = 2024-25 GW21–38 via --from-gw 21 --to-gw 38 (disjoint). Optional second confirm: --seasons 2025-26 on finished GWs only. Sealed: unfinished 2025-26 / 2026-27 GWs. Do not nest GW1–38 primary + GW20–38 confirm.

## Coverage
- (no coverage metric configured)

## Join / available_at
- coverage.read: strict_pit
- rule: available_at <= gameweek deadline (PITView)

## PIT caveat
- (none)

## Prediction deltas
- rmse_delta (headline; treatment better > 0): -0.006324491649629671
- mae_delta (secondary; treatment better > 0): -0.006031921454073386
- spearman_delta (secondary; treatment better > 0): -0.001246102472409305
- top10_precision_delta (secondary; treatment better > 0): -0.009999999999999981
- brier_60_delta (treatment better > 0): -3.5416147354527805e-06
- brier_assist_delta (treatment better > 0): -0.00012948929243751517
- brier_cs_delta (treatment better > 0): -0.0011440400016228536
- brier_goal_delta (treatment better > 0): -0.00015318015903927062
- brier_haul_delta (treatment better > 0): -0.0001323980120090905
- brier_played_delta (treatment better > 0): -5.9004865303469156e-05
- log_60_delta (treatment better > 0): -3.783848010208146e-05
- log_assist_delta (treatment better > 0): -0.0011578461120841377
- log_cs_delta (treatment better > 0): -0.0037975886841675743
- log_goal_delta (treatment better > 0): -0.0008086341819513732
- log_haul_delta (treatment better > 0): -0.0014786421687643936
- log_played_delta (treatment better > 0): -7.5518241797079e-05
- per_gw_wins (RMSE): control=12 treatment=8 ties=0
- per_gw_wins (MAE, continuity): control=13 treatment=7 ties=0
- rmse: mean_d=-0.007242064271043058 HAC_se=0.002323969272892471 HAC_p=0.0018316824594112793 HAC_CI=[-0.011796960347090036, -0.002687168194996076] bootstrap_CI=[-0.010797754391104409, -0.0035733977561417715]
- mae: mean_d=-0.00736737590086598 HAC_se=0.002424903693042975 HAC_p=0.00237985031380266 HAC_CI=[-0.01212009980520838, -0.0026146519965235763] bootstrap_CI=[-0.011303630306681514, -0.003680551229056221]
- spearman: mean_d=-0.002944149885138814 HAC_se=0.001936160573264055 HAC_p=0.128357119072007 HAC_CI=[-0.006738954877022785, 0.0008506551067451601] bootstrap_CI=[-0.00618429066951343, 5.25570222848419e-05]
- intervals are descriptive for improvement levers (not a keep rule; equivalence may drop on entirely-bad intervals)
- player pool: decision-relevant pool: as-of price >= 45 (£4.5m) and (price >= 70 or prior season minutes >= 90 or as_of_gw <= 2); PIT via available_at <= deadline
- W9-style MAE/RMSE split: RMSE is the mean-seeking headline (optimiser consumes expected points); MAE alone can prefer median-seeking forecasts that underprice haul upside — read both, prefer RMSE when they disagree on a mean-seeking board
- raw de-vig 1X2 (raw_de_vig_1x2) mean rps: 0.27777777777777773
- odds_poisson_v1 is not the market row (book = raw proportional de-vig 1X2)

## Calibration
Event reliability by xPts decile and position (see also compare/calibration.parquet):

```
shape: (40, 9)
┌──────────────────┬─────────┬────────┬─────────────┬───┬──────┬──────────┬──────────┬──────────┐
│ model_id         ┆ horizon ┆ event  ┆ slice_kind  ┆ … ┆ n    ┆ mean_p   ┆ mean_y   ┆ brier    │
│ ---              ┆ ---     ┆ ---    ┆ ---         ┆   ┆ ---  ┆ ---      ┆ ---      ┆ ---      │
│ str              ┆ i64     ┆ str    ┆ str         ┆   ┆ i64  ┆ f64      ┆ f64      ┆ f64      │
╞══════════════════╪═════════╪════════╪═════════════╪═══╪══════╪══════════╪══════════╪══════════╡
│ 1a1a9fd52db7250e ┆ 1       ┆ played ┆ position    ┆ … ┆ 1540 ┆ 0.682593 ┆ 0.761039 ┆ 0.153651 │
│ 1a1a9fd52db7250e ┆ 1       ┆ played ┆ position    ┆ … ┆ 3465 ┆ 0.619644 ┆ 0.735931 ┆ 0.17827  │
│ 1a1a9fd52db7250e ┆ 1       ┆ played ┆ position    ┆ … ┆ 372  ┆ 0.837084 ┆ 0.776882 ┆ 0.131101 │
│ 1a1a9fd52db7250e ┆ 1       ┆ played ┆ position    ┆ … ┆ 785  ┆ 0.654754 ┆ 0.743949 ┆ 0.151488 │
│ 1a1a9fd52db7250e ┆ 1       ┆ played ┆ xpts_decile ┆ … ┆ 617  ┆ 0.582697 ┆ 0.690438 ┆ 0.217873 │
│ …                ┆ …       ┆ …      ┆ …           ┆ … ┆ …    ┆ …        ┆ …        ┆ …        │
│ 1a1a9fd52db7250e ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 616  ┆ 0.101878 ┆ 0.121753 ┆ 0.100507 │
│ 1a1a9fd52db7250e ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 616  ┆ 0.029756 ┆ 0.040584 ┆ 0.03855  │
│ 1a1a9fd52db7250e ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 616  ┆ 0.198855 ┆ 0.204545 ┆ 0.138084 │
│ 1a1a9fd52db7250e ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 617  ┆ 0.052696 ┆ 0.053485 ┆ 0.048352 │
│ 1a1a9fd52db7250e ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 616  ┆ 0.064707 ┆ 0.079545 ┆ 0.071307 │
└──────────────────┴─────────┴────────┴─────────────┴───┴──────┴──────────┴──────────┴──────────┘
```

## Components
- brier_cs_away_delta: -0.007648896076368983
- brier_cs_home_delta: -0.008839075876462932
- logloss_delta: -0.03411744102552827
- rps_delta: -0.006531903897732172
- market arm: raw_de_vig_1x2 (same clock as coverage.read; not odds_poisson_v1)
- per_match rows: 2388 (fixture × arm × metric → per_match.parquet)

## Decisions
- decisions: not run

## Equivalence
- n/a (improvement lever)

## Call
**inconclusive**

Improvement levers stay auto-inconclusive until numeric keep_if thresholds land (FDB-73). Deltas are descriptive only; keep_if is not applied. Written keep/tune/drop follows docs/eval-protocol.md §2 (confirmation window + SPA budget). Headline point metric is rmse on the decision-relevant pool.

(compare output: /cursor/stores/bc-8a102b3d-bfb7-40d0-ac50-e35a866b399f/internal/dynamic-covariate-ratings-promote/development/compare)
