# `fplt` package

Library-first FPL points prediction and team selection. Dagster / CLI / apps are thin callers — **all logic lives here**.

| Module | Role |
|---|---|
| `data/` | Clients, raw/staging, `DataStore`, **`PITView`** |
| `store/` | Write-once analytic runs (`RunWriter`); catalog / Store readers later |
| `schemas/` | `pandera.polars` contracts at module boundaries |
| `features/` | `@feature(...)` registry (versioned) |
| `scoring/` | Season `Rules` + chip tables from `configs/` |
| `models/` | `PointsPredictor`, components, simulation, specs |
| `backtest/` | Walk-forward, ledger, compare / promote-eval / bakeoff |
| `optimise/` | Squad MILP, chips, plans, apply |
| `orchestration/` | Thin Dagster assets / sensors / factories |
| `automation/` | Catch-up, pre-deadline, production ops, notify |
| `cli.py` | `uv run fplt …` entry |
| `settings.py` | Paths + env (`FPLT_*`); no hardcoded `./data` |

## Hard invariants (do not bypass)

1. Models/features read **only** through `PITView` (`fplt.data.pit`) — never `DataStore` / parquet paths / FPL API from model code.
2. Every staged table carries `available_at` (UTC).
3. New predictors/components pass `tests/contracts/` **including leakage**.
4. Polars everywhere (pandas only inside `models/external/` adapters).
5. Register models with `@register_model` + `configs/models/*.yaml`; bump feature versions on output changes.
6. Seeded `numpy.random.Generator` via `fplt.utils.rng.seed_for` — no global RNG.
7. Layered models compose via **simulation**, not chained point estimates.

## Agents

- Standing brief: [`AGENTS.md`](../../AGENTS.md).
- Architecture: [`docs/design.md`](../../docs/design.md) (start at the sections the issue names).
- Config map: [`configs/README.md`](../../configs/README.md).
- Before finishing: `make check`.
