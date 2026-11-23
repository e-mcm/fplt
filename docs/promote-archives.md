# Promote / board archives (not in git)

Bulky board outputs — parquet ledgers, run logs, chip windows, rollups — live in the **Project Agent Store**, not this repository.

- **Store id:** `bc-8a102b3d-bfb7-40d0-ac50-e35a866b399f`
- **Path:** Project store `internal/<experiment>/` (e.g. `internal/defcon-rate-promote/`, `internal/price-within-horizon/`, `internal/stacked-prod-2400/`)

Promote write-ups under `docs/*-promote.md` may still say `internal/...`; that means the Project store, not a path in this clone.

## Companion eval notes still store-only

This repo commits the promote write-ups, plans, and decision summaries listed in [`docs/README.md`](README.md). Older one-off eval notes (e.g. `defensive-contribution-eval-2025-26.md`, `cs-only-market-lambda-eval.md`, grid/board scoreboards) may remain only in Project store `docs/` until someone copies them deliberately. Prefer linking those as “Project store `docs/<name>.md`” rather than inventing empty stubs here.
