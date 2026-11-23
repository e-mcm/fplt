# Minutes P1–P6 package vs pre-toggle control (FDB-257)

question: Do the P1–P6 minutes toggles (calibrated chance flags, return ramp, horizon availability recovery) improve minutes Brier and player-fixture RMSE versus the pre-toggle empirical_minutes_v2 baseline?
question_type: improvement
kind: config
orientation: opt_out
lever: hyperparam
diff:
  - components.minutes.params.calibrate_chance_flags: false → true
  - components.minutes.params.return_ramp_games: 0.0 → 2.0
  - components.minutes.params.availability_recovery_games: 0.0 → 3.0

## Arms
- control: dc_layered_minutes_pre_p1p6_v1 model_id=9cffba1c4d773ca7
- treatment: dc_layered_v3 (production) model_id=30367a901721a5d6

## Audit
- git_sha: 106dc1818868493a56b3a92861920354be6cb91f
- git_dirty: False
- uv_lock_sha256: e9ae5f72e824c4d614230105110022eb159fb9c422befd6ad0c0ea8e26a5dd73
- full record: audit.json

## Window
- seasons: 2024-25
- gw: 1-20
- seasons_back: 4
- modes: walk_forward
- n_sims: 10000
- note: Development window (eval protocol §1). Confirmation is GW21–38 on the same season (disjoint). Do not re-select doses on confirmation. Snapshot density is highest on live 2026-27 but those GWs are sealed / unfinished — not a keep window.

## Coverage
- metric=fpl_api_status_share numerator=0 denominator=13454 rate=0.0000 min_coverage=0.01 → fail
- source_fpl_api_share: 0.0

## Join / available_at
- coverage.read: strict_pit
- rule: available_at <= gameweek deadline (PITView)

## PIT caveat
- (none)

## Prediction deltas
- prediction: not run

## Calibration
- prediction: not run

## Components
- components: not run

## Decisions
- decisions: not run

## Equivalence
- n/a (improvement lever)

## Call
**drop**

Coverage 0.000 is below min_coverage=0.01 for metric=fpl_api_status_share. Drop until the lever actually runs on the model's read.
