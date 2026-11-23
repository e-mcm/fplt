# Docs index

Authoritative architecture is **[`design.md`](design.md)**. Everything else here is a focused companion — do not fork a second design doc.

| Doc | Audience | Use when |
|---|---|---|
| [`design.md`](design.md) | Humans + agents | Contracts, PIT, schemas, models, optimiser, Dagster §8, storage layout |
| [`dagster-cadence-and-setup.md`](dagster-cadence-and-setup.md) | Ops / agents | GHA capture cadence, sensors, always-on setup |
| [`feature-eval-pipeline.md`](feature-eval-pipeline.md) | Modelling / agents | Promote-eval, bakeoffs, control vs treatment, keep/drop |
| [`evaluation-north-star.md`](evaluation-north-star.md) | Eval / agents | Forecast / decision / live aims; RMSE headline; points-at-rank decision-sim bars (2400 / 2235) |
| [`benchmark-archives.md`](benchmark-archives.md) | Eval / agents | `benchmark_projections` table (ep_next / OpenFPL / FPL Review) |
| [`openfpl.md`](openfpl.md) | Ops / agents | OpenFPL subprocess setup and season floor |
| [`fpl_review.md`](fpl_review.md) | Ops | Manual FPL Review CSV ingest (no scrape) |

## Standing rules (agents)

- Start from [`AGENTS.md`](../AGENTS.md) + the design sections named by the issue.
- Prefer updating these files over inventing parallel docs under `src/` or Project store.
- Repo layout cheat-sheet: root [`README.md`](../README.md). Per-directory maps: `configs/`, `scripts/`, `src/fplt/`, `tests/`, `deploy/`, `external/`.
