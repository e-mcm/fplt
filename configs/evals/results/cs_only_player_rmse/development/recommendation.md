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
- gw: 1-20
- seasons_back: 4
- modes: walk_forward
- n_sims: 10000
- note: Development window (eval protocol §1). Confirmation = 2024-25 GW21–38 via --from-gw 21 --to-gw 38 (disjoint). Optional second confirm: --seasons 2025-26 on finished GWs only. Do not use nested GW1–38 primary + GW20–38 confirm. Sealed: unfinished 2025-26 / 2026-27 GWs.

## Coverage
- metric=odds_join numerator=185 denominator=199 rate=0.9296 min_coverage=0.8 → pass
- closing_proxy_rate: 1.0
- fixture_id_join_rate: 1.0
- prematch_dropped_fixture_ids: [10, 52, 77, 92, 119, 150, 152, 171, 173, 181, 182, 186, 188, 200]
- prematch_scheduled_rate: 0.9296482412060302
- read: prematch_scheduled
- strict_pit_rate: 0.9296482412060302
- team_key_join_rate: 1.0

## Join / available_at
- coverage.read: prematch_scheduled
- rule: available_at <= gameweek deadline (PITView)
- market_cs_strength (treatment): 0.5

## PIT caveat
Closing lines are stamped available_at=kickoff, after the FPL deadline. This run did not move the PIT cutoff. Strict-PIT odds join = 93.0%. Closing-proxy join (labelled; used only by specs with use_closing_proxy: true) = 100.0%. Prematch-scheduled deadline coverage = 93.0%. market_prior_source = latest. market_cs_strength = 0.5.

## Prediction deltas
- rmse_delta (headline; treatment better > 0): 0.001698006562299792
- mae_delta (secondary; treatment better > 0): 0.0019831710483608767
- spearman_delta (secondary; treatment better > 0): 0.0013983072519065942
- top10_precision_delta (secondary; treatment better > 0): 0.0050000000000000044
- brier_60_delta (treatment better > 0): -2.874115546669387e-07
- brier_assist_delta (treatment better > 0): 4.319620740018959e-05
- brier_cs_delta (treatment better > 0): 0.0006416759331385885
- brier_goal_delta (treatment better > 0): 4.528288055827279e-05
- brier_haul_delta (treatment better > 0): -4.824878286274065e-06
- brier_played_delta (treatment better > 0): -3.7297908146716185e-05
- log_60_delta (treatment better > 0): 0.00030145906691980695
- log_assist_delta (treatment better > 0): 0.0002680653707546332
- log_cs_delta (treatment better > 0): 0.0019275148298502631
- log_goal_delta (treatment better > 0): 0.00039765912213785004
- log_haul_delta (treatment better > 0): 0.004133113269694705
- log_played_delta (treatment better > 0): 1.2448105850992874e-05
- per_gw_wins (RMSE): control=7 treatment=13 ties=0
- per_gw_wins (MAE, continuity): control=5 treatment=15 ties=0
- rmse: mean_d=0.001519833778646329 HAC_se=0.0008760730364957116 HAC_p=0.08277173120119734 HAC_CI=[-0.00019723782071191082, 0.0032369053780045677] bootstrap_CI=[-3.4073325849526605e-05, 0.00285493188275148]
- mae: mean_d=0.00205263718816594 HAC_se=0.000662025193426195 HAC_p=0.0019316654991085854 HAC_CI=[0.0007550916521924342, 0.0033501827241394447] bootstrap_CI=[0.0010042678677982335, 0.0031341184966390996]
- spearman: mean_d=0.0021195831921767714 HAC_se=0.00037767609331947717 HAC_p=1.9980270074996513e-08 HAC_CI=[0.0013793516514488072, 0.0028598147329047348] bootstrap_CI=[0.001372099030666852, 0.002840001912350669]
- intervals are descriptive for improvement levers (not a keep rule; equivalence may drop on entirely-bad intervals)
- player pool: decision-relevant pool: as-of price >= 45 (£4.5m) and (price >= 70 or prior season minutes >= 90 or as_of_gw <= 2); PIT via available_at <= deadline
- W9-style MAE/RMSE split: RMSE is the mean-seeking headline (optimiser consumes expected points); MAE alone can prefer median-seeking forecasts that underprice haul upside — read both, prefer RMSE when they disagree on a mean-seeking board
- raw de-vig 1X2 (raw_de_vig_1x2) mean rps: 0.35466482460136045
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
│ 30367a901721a5d6 ┆ 1       ┆ played ┆ position    ┆ … ┆ 1540 ┆ 0.682589 ┆ 0.761039 ┆ 0.153828 │
│ 30367a901721a5d6 ┆ 1       ┆ played ┆ position    ┆ … ┆ 3465 ┆ 0.619689 ┆ 0.735931 ┆ 0.178334 │
│ 30367a901721a5d6 ┆ 1       ┆ played ┆ position    ┆ … ┆ 372  ┆ 0.837094 ┆ 0.776882 ┆ 0.131311 │
│ 30367a901721a5d6 ┆ 1       ┆ played ┆ position    ┆ … ┆ 785  ┆ 0.655095 ┆ 0.743949 ┆ 0.151281 │
│ 30367a901721a5d6 ┆ 1       ┆ played ┆ xpts_decile ┆ … ┆ 616  ┆ 0.583163 ┆ 0.696429 ┆ 0.217904 │
│ …                ┆ …       ┆ …      ┆ …           ┆ … ┆ …    ┆ …        ┆ …        ┆ …        │
│ 30367a901721a5d6 ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 616  ┆ 0.100356 ┆ 0.113636 ┆ 0.095477 │
│ 30367a901721a5d6 ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 616  ┆ 0.030083 ┆ 0.040584 ┆ 0.038464 │
│ 30367a901721a5d6 ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 616  ┆ 0.199668 ┆ 0.211039 ┆ 0.141609 │
│ 30367a901721a5d6 ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 616  ┆ 0.063932 ┆ 0.082792 ┆ 0.074113 │
│ 30367a901721a5d6 ┆ 1       ┆ goal   ┆ xpts_decile ┆ … ┆ 617  ┆ 0.006376 ┆ 0.011345 ┆ 0.011123 │
└──────────────────┴─────────┴────────┴─────────────┴───┴──────┴──────────┴──────────┴──────────┘
```

## Components
- brier_cs_away_delta: 0.006148180882303733
- brier_cs_home_delta: 0.004896909132791616
- logloss_delta: 1.6577440648646302e-05
- rps_delta: 2.916488204662837e-06
- market arm: raw_de_vig_1x2 (same clock as coverage.read; not odds_poisson_v1)
- per_match rows: 2388 (fixture × arm × metric → per_match.parquet)

## Decisions
- decisions: not run

## Equivalence
- n/a (improvement lever)

## Call
**inconclusive**

Improvement levers stay auto-inconclusive until numeric keep_if thresholds land (FDB-73). Deltas are descriptive only; keep_if is not applied. Written keep/tune/drop follows docs/eval-protocol.md §2 (confirmation window + SPA budget). Headline point metric is rmse on the decision-relevant pool.

(compare output: /tmp/fplt-cs-player-rmse/dev/compare)
