# Docs index

Authoritative architecture is **[`design.md`](design.md)**. Everything else here is a focused companion — do not fork a second design doc.

## Core companions

| Doc | Audience | Use when |
|---|---|---|
| [`design.md`](design.md) | Humans + agents | Contracts, PIT, schemas, models, optimiser, Dagster §8, storage layout |
| [`dagster-cadence-and-setup.md`](dagster-cadence-and-setup.md) | Ops / agents | GHA capture cadence, sensors, always-on setup |
| [`blank-week-prod-plan.md`](blank-week-prod-plan.md) | Ops / agents | Empty-squad / blank-week `fplt plan` with production model + live defaults |
| [`next-gw-actions.md`](next-gw-actions.md) | Ops / agents | Logged-in squad: browser auth → production `fplt predict` → next-deadline `fplt plan` (not `apply --confirm`) |
| [`feature-eval-pipeline.md`](feature-eval-pipeline.md) | Modelling / agents | Promote-eval, bakeoffs, control vs treatment, keep/drop |
| [`evaluation-north-star.md`](evaluation-north-star.md) | Eval / agents | Forecast / decision / live aims; RMSE headline; points-at-rank bars (2400 / 2235) |
| [`eval-protocol.md`](eval-protocol.md) | Eval / agents | Windows, interim written keep, reference board, HiGHS; `claim: squad_points` decisions gate |
| [`w9-mae-rmse-reading.md`](w9-mae-rmse-reading.md) | Eval / agents | How to read MAE vs RMSE splits on mean-seeking boards |
| [`benchmark-archives.md`](benchmark-archives.md) | Eval / agents | `benchmark_projections` table (ep_next / OpenFPL / FPL Review) |
| [`openfpl.md`](openfpl.md) | Ops / agents | OpenFPL subprocess setup and season floor |
| [`fpl_review.md`](fpl_review.md) | Ops | Manual FPL Review CSV ingest (no scrape) |

## Promote / decision reference (committed from Project store)

Written promote calls, plans, and decision inventory. Board parquet/logs stay in the Project store — see [`promote-archives.md`](promote-archives.md).

| Doc | Use when |
|---|---|
| [`promote-pipeline-unified.md`](promote-pipeline-unified.md) | End-to-end promote pipeline / process |
| [`promo-evals-consolidated.md`](promo-evals-consolidated.md) | One-page W7–W9 + Wave B + A3 + rate-model rollup |
| [`unpromoted-improvements.md`](unpromoted-improvements.md) | Backlog of levers (kept / dropped / in flight) |
| [`stacked-prod-2400-board.md`](stacked-prod-2400-board.md) | Stacked production board vs points-at-rank bars (2400 / 2235) |
| [`live-prediction-regen.md`](live-prediction-regen.md) | Live prediction regen notes |
| [`price-within-horizon-design.md`](price-within-horizon-design.md) | Within-horizon list-price forecast design |
| [`joint-samples-optimiser-design.md`](joint-samples-optimiser-design.md) | Joint MC samples → optimiser SAA (FDB-241) |
| [`promote-archives.md`](promote-archives.md) | Where bulky `internal/` board archives live |

### Promote write-ups (`*-promote.md`)

| Doc | Call (summary) |
|---|---|
| [`availability-overlays-promote.md`](availability-overlays-promote.md) | Availability overlays (injury + predicted XI) |
| [`predicted-xi-audit.md`](predicted-xi-audit.md) | Predicted XI accuracy audit + late re-predict hooks (FDB-253) |
| [`minutes-measurement.md`](minutes-measurement.md) | Snapshot minutes measurement split + misses (FDB-246) |
| [`cs-only-player-rmse-promote.md`](cs-only-player-rmse-promote.md) | CS-only market λ player/RMSE — **keep** (flip PR pending) |
| [`other-points-measurement.md`](other-points-measurement.md) | Other-points / haul measurement + match residual (FDB-240) |
| [`defcon-rate-promote.md`](defcon-rate-promote.md) | Defcon rate — **keep (provisional)** pending FDB-73 |
| [`devig-1x2-prior-promote.md`](devig-1x2-prior-promote.md) | De-vig 1X2 — stop (no runnable prematch arm) |
| [`force-play-eval-promote.md`](force-play-eval-promote.md) | H=8 + force-off candidate; live stays H=3 force-on |
| [`forecast-combine-promote.md`](forecast-combine-promote.md) | Equal forecast combine — **drop** |
| [`history-market-prior-promote.md`](history-market-prior-promote.md) | History rating prior — **tune** (no PRODUCTION flip) |
| [`lineup-reopt-promote.md`](lineup-reopt-promote.md) | `lineup_reopt` — **keep** (already default) |
| [`prematch-1x2-prior-promote.md`](prematch-1x2-prior-promote.md) | Prematch 1X2 prior (FDB-216 / FDB-219) |
| [`price-dynamics-sell-on-promote.md`](price-dynamics-sell-on-promote.md) | As-of 50% sell-on — aborted (bug fix, not promote) |
| [`price-within-horizon-promote.md`](price-within-horizon-promote.md) | `mean_gw_delta` — **keep** (confirmed FDB-256) |
| [`price-forecast-confirm.md`](price-forecast-confirm.md) | Protocol confirmation of `mean_gw_delta` keep (FDB-256) |
| [`prune-n-equivalence-promote.md`](prune-n-equivalence-promote.md) | Prune-N equivalence — **keep** (already live) |

Plans that drove those boards: [`promote-plans/`](promote-plans/).

## Standing rules (agents)

- Start from [`AGENTS.md`](../AGENTS.md) + the design sections named by the issue.
- Prefer updating these files over inventing parallel docs under `src/` or Project store.
- Repo layout cheat-sheet: root [`README.md`](../README.md). Per-directory maps: `configs/`, `scripts/`, `src/fplt/`, `tests/`, `deploy/`, `external/`, `mc/` ([`mc/README.md`](../mc/README.md) — `fplt-mc` native kernel; agents must not compile).
