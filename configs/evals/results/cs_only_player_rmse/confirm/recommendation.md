# CS-only market λ player RMSE vs dc_layered_v3 (prematch, strict PIT)

question: Does borrowing the market's zero-goal (clean-sheet) margin from prematch_scheduled odds at strength 0.5 improve player-fixture RMSE (and keep CS Brier better / RPS not worse) vs dc_layered_v3 under strict PIT?
question_type: improvement
kind: data
orientation: opt_in
lever: data_table
diff:
  - consume staged match_odds via market_cs_strength=0.5 (CS margins only)

## Arms
- control: dc_layered_v3 (production) [market_cs_strength=0.0] model_id=30367a901721a5d6
- treatment: dc_layered_cs_v1 [market_cs_strength=0.5] model_id=b16a511a480735f5

## Audit
- git_sha: 98d06a216437673efd4336a8f531990b2444c365
- git_dirty: False
- uv_lock_sha256: e9ae5f72e824c4d614230105110022eb159fb9c422befd6ad0c0ea8e26a5dd73
- full record: audit.json

## Window
- seasons: 2024-25
- gw: 21-38
- seasons_back: 4
- modes: walk_forward
- n_sims: 10000
- note: Development window (eval protocol §1). Confirmation = 2024-25 GW21–38 via --from-gw 21 --to-gw 38 (disjoint). Optional second confirm: --seasons 2025-26 on finished GWs only. Do not use nested GW1–38 primary + GW20–38 confirm. Sealed: unfinished 2025-26 / 2026-27 GWs.

## Coverage
- metric=odds_join numerator=165 denominator=181 rate=0.9116 min_coverage=0.8 → pass
- closing_proxy_rate: 1.0
- fixture_id_join_rate: 1.0
- prematch_dropped_fixture_ids: [144, 213, 234, 279, 282, 289, 307, 312, 329, 331, 336, 338, 346, 364, 366, 369]
- prematch_scheduled_rate: 0.9116022099447514
- read: prematch_scheduled
- strict_pit_rate: 0.9116022099447514
- team_key_join_rate: 1.0

## Join / available_at
- coverage.read: prematch_scheduled
- rule: available_at <= gameweek deadline (PITView)
- market_cs_strength (treatment): 0.5

## PIT caveat
Closing lines are stamped available_at=kickoff, after the FPL deadline. This run did not move the PIT cutoff. Strict-PIT odds join = 91.2%. Closing-proxy join (labelled; used only by specs with use_closing_proxy: true) = 100.0%. Prematch-scheduled deadline coverage = 91.2%. market_prior_source = latest. market_cs_strength = 0.5.

## Prediction deltas
- rmse_delta (headline; treatment better > 0): 0.0021332292376885498
- mae_delta (secondary; treatment better > 0): 0.0011684457871770082
- spearman_delta (secondary; treatment better > 0): 0.0009733746188538284
- top10_precision_delta (secondary; treatment better > 0): 0.005555555555555564
- brier_60_delta (treatment better > 0): -7.652254615639342e-05
- brier_assist_delta (treatment better > 0): 3.3645229942931565e-05
- brier_cs_delta (treatment better > 0): 0.0007262368697549676
- brier_goal_delta (treatment better > 0): 1.2746141322593496e-05
- brier_haul_delta (treatment better > 0): 3.705941591134604e-06
- brier_played_delta (treatment better > 0): 1.0056074185971209e-05
- log_60_delta (treatment better > 0): 1.3694086190563226e-06
- log_assist_delta (treatment better > 0): 0.0004911553320518791
- log_cs_delta (treatment better > 0): 0.002207296463407671
- log_goal_delta (treatment better > 0): -9.45712438360502e-05
- log_haul_delta (treatment better > 0): -0.0002187215176159274
- log_played_delta (treatment better > 0): 9.769776454793266e-05
- per_gw_wins (RMSE): control=6 treatment=12 ties=0
- per_gw_wins (MAE, continuity): control=9 treatment=9 ties=0
- rmse: mean_d=0.002215300437983508 HAC_se=0.0008437163225942791 HAC_p=0.008648469496953177 HAC_CI=[0.0005616468325301421, 0.003868954043436872] bootstrap_CI=[0.0008282592168429594, 0.0037765577492227024]
- mae: mean_d=0.00120660420407464 HAC_se=0.0005556960806900966 HAC_p=0.02990560633028722 HAC_CI=[0.00011745989957198623, 0.0022957485085772925] bootstrap_CI=[0.00016834745337684521, 0.0022823658857794443]
- spearman: mean_d=0.001108182966279955 HAC_se=0.00032444042471276653 HAC_p=0.0006362419152519004 HAC_CI=[0.00047229141871405324, 0.001744074513845856] bootstrap_CI=[0.0005470202747919924, 0.0016531879710344516]
- intervals are descriptive for improvement levers (not a keep rule; equivalence may drop on entirely-bad intervals)
- player pool: decision-relevant pool: as-of price >= 45 (£4.5m) and (price >= 70 or prior season minutes >= 90 or as_of_gw <= 2); PIT via available_at <= deadline
- W9-style MAE/RMSE split: RMSE is the mean-seeking headline (optimiser consumes expected points); MAE alone can prefer median-seeking forecasts that underprice haul upside — read both, prefer RMSE when they disagree on a mean-seeking board
- raw de-vig 1X2 (raw_de_vig_1x2) mean rps: 0.36052346348741765
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
│ 30367a901721a5d6 ┆ 1       ┆ played ┆ position    ┆ … ┆ 3634 ┆ 0.615087 ┆ 0.68492  ┆ 0.172449 │
│ 30367a901721a5d6 ┆ 1       ┆ played ┆ position    ┆ … ┆ 304  ┆ 0.854928 ┆ 0.825658 ┆ 0.096303 │
│ 30367a901721a5d6 ┆ 1       ┆ played ┆ position    ┆ … ┆ 863  ┆ 0.616707 ┆ 0.604867 ┆ 0.198966 │
│ 30367a901721a5d6 ┆ 1       ┆ played ┆ position    ┆ … ┆ 1157 ┆ 0.672521 ┆ 0.702679 ┆ 0.15217  │
│ 30367a901721a5d6 ┆ 1       ┆ played ┆ xpts_decile ┆ … ┆ 595  ┆ 0.589669 ┆ 0.736134 ┆ 0.211474 │
│ …                ┆ …       ┆ …      ┆ …           ┆ … ┆ …    ┆ …        ┆ …        ┆ …        │
│ 30367a901721a5d6 ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 597  ┆ 0.018634 ┆ 0.021776 ┆ 0.021168 │
│ 30367a901721a5d6 ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 596  ┆ 0.061382 ┆ 0.075503 ┆ 0.066866 │
│ 30367a901721a5d6 ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 596  ┆ 0.053132 ┆ 0.063758 ┆ 0.058695 │
│ 30367a901721a5d6 ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 596  ┆ 0.009062 ┆ 0.016779 ┆ 0.016535 │
│ 30367a901721a5d6 ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 596  ┆ 0.037668 ┆ 0.035235 ┆ 0.03436  │
└──────────────────┴─────────┴────────┴─────────────┴───┴──────┴──────────┴──────────┴──────────┘
```

## Components
- brier_cs_away_delta: 0.004881570217028758
- brier_cs_home_delta: 0.001726820408872426
- logloss_delta: -1.568897571502248e-05
- rps_delta: 1.829592408686409e-07
- market arm: raw_de_vig_1x2 (same clock as coverage.read; not odds_poisson_v1)
- per_match rows: 2172 (fixture × arm × metric → per_match.parquet)

## Decisions
- decisions: not run

## Equivalence
- n/a (improvement lever)

## Call
**inconclusive**

Improvement levers stay auto-inconclusive until numeric keep_if thresholds land (FDB-73). Deltas are descriptive only; keep_if is not applied. Written keep/tune/drop follows docs/eval-protocol.md §2 (confirmation window + SPA budget). Headline point metric is rmse on the decision-relevant pool.

(compare output: /tmp/fplt-cs-player-rmse/confirm-clean/compare)
