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
- gw: 21-38
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
- rmse_delta (headline; treatment better > 0): -0.004728526248875209
- mae_delta (secondary; treatment better > 0): -0.006178298086606082
- spearman_delta (secondary; treatment better > 0): -0.005250721933969804
- top10_precision_delta (secondary; treatment better > 0): 0.016666666666666677
- brier_60_delta (treatment better > 0): -0.00013661928835179138
- brier_assist_delta (treatment better > 0): -0.0001317933736153104
- brier_cs_delta (treatment better > 0): -0.0013668920174555177
- brier_goal_delta (treatment better > 0): -8.256365391072495e-05
- brier_haul_delta (treatment better > 0): -1.053721718698164e-05
- brier_played_delta (treatment better > 0): -1.3144013091603846e-05
- log_60_delta (treatment better > 0): -0.0004688070184504234
- log_assist_delta (treatment better > 0): -0.001188123658929785
- log_cs_delta (treatment better > 0): -0.006813003760236858
- log_goal_delta (treatment better > 0): -0.001345044574767723
- log_haul_delta (treatment better > 0): -0.0010349356377181984
- log_played_delta (treatment better > 0): 3.1005598992273065e-05
- per_gw_wins (RMSE): control=9 treatment=9 ties=0
- per_gw_wins (MAE, continuity): control=12 treatment=6 ties=0
- rmse: mean_d=-0.0037383542895911403 HAC_se=0.002645233713969754 HAC_p=0.1575847028256938 HAC_CI=[-0.008922917099662982, 0.0014462085204807063] bootstrap_CI=[-0.008358487075328177, 0.0008166737882935563]
- mae: mean_d=-0.005697973117459145 HAC_se=0.002067742599801688 HAC_p=0.005857579325742458 HAC_CI=[-0.009750674142369669, -0.001645272092548617] bootstrap_CI=[-0.009419460913690903, -0.001718391061912765]
- spearman: mean_d=-0.004509044729613822 HAC_se=0.001338150693413129 HAC_p=0.00075274890533547 HAC_CI=[-0.007131771894590853, -0.0018863175646367885] bootstrap_CI=[-0.006688153117900398, -0.0024148904435800454]
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
│ 1a1a9fd52db7250e ┆ 1       ┆ played ┆ position    ┆ … ┆ 3634 ┆ 0.614972 ┆ 0.68492  ┆ 0.172433 │
│ 1a1a9fd52db7250e ┆ 1       ┆ played ┆ position    ┆ … ┆ 304  ┆ 0.855005 ┆ 0.825658 ┆ 0.096118 │
│ 1a1a9fd52db7250e ┆ 1       ┆ played ┆ position    ┆ … ┆ 863  ┆ 0.616899 ┆ 0.604867 ┆ 0.198879 │
│ 1a1a9fd52db7250e ┆ 1       ┆ played ┆ position    ┆ … ┆ 1157 ┆ 0.672538 ┆ 0.702679 ┆ 0.152265 │
│ 1a1a9fd52db7250e ┆ 1       ┆ played ┆ xpts_decile ┆ … ┆ 595  ┆ 0.588501 ┆ 0.72605  ┆ 0.21226  │
│ …                ┆ …       ┆ …      ┆ …           ┆ … ┆ …    ┆ …        ┆ …        ┆ …        │
│ 1a1a9fd52db7250e ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 596  ┆ 0.018647 ┆ 0.018456 ┆ 0.018049 │
│ 1a1a9fd52db7250e ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 596  ┆ 0.060864 ┆ 0.075503 ┆ 0.06794  │
│ 1a1a9fd52db7250e ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 596  ┆ 0.009079 ┆ 0.016779 ┆ 0.016523 │
│ 1a1a9fd52db7250e ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 596  ┆ 0.053186 ┆ 0.065436 ┆ 0.059276 │
│ 1a1a9fd52db7250e ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 596  ┆ 0.037997 ┆ 0.033557 ┆ 0.032525 │
└──────────────────┴─────────┴────────┴─────────────┴───┴──────┴──────────┴──────────┴──────────┘
```

## Components
- brier_cs_away_delta: -0.008490301347414173
- brier_cs_home_delta: -0.010807583948832855
- logloss_delta: -0.023003298549991946
- rps_delta: -0.003575986120801178
- market arm: raw_de_vig_1x2 (same clock as coverage.read; not odds_poisson_v1)
- per_match rows: 2172 (fixture × arm × metric → per_match.parquet)

## Decisions
- decisions: not run

## Equivalence
- n/a (improvement lever)

## Call
**inconclusive**

Improvement levers stay auto-inconclusive until numeric keep_if thresholds land (FDB-73). Deltas are descriptive only; keep_if is not applied. Written keep/tune/drop follows docs/eval-protocol.md §2 (confirmation window + SPA budget). Headline point metric is rmse on the decision-relevant pool.

(compare output: /cursor/stores/bc-8a102b3d-bfb7-40d0-ac50-e35a866b399f/internal/dynamic-covariate-ratings-promote/confirm/compare)
