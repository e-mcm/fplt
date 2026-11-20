# fplt

Fantasy Premier League points prediction and team selection.

`fplt` predicts points per player per gameweek with composable, versioned models, benchmarks them against each other and external models (e.g. OpenFPL) on point-in-time walk-forward backtests, and turns predictions into squad, captaincy and transfer recommendations. Dagster runs the whole cycle automatically after each gameweek and before each deadline.

- Architecture: [`docs/design.md`](docs/design.md)
- Rules for coding agents and contributors: [`AGENTS.md`](AGENTS.md)

## How it fits together

```
FPL API ─(GitHub Actions snapshots)─► raw ─► staged ─► features ─► models ─► predictions ─► optimiser ─► plan
                                                                     │
                                              backtests ──► ledger ──┴──► leaderboard app
```

- **Predictors** all implement one `PointsPredictor` interface: layered models (score model + minutes + involvement + other points, combined by simulation), all-in-one models, baselines, and external adapters.
- **Everything is point-in-time**: models only see data available at each gameweek's deadline, so backtests are honest.
- **Every model is a YAML `ModelSpec`**; its hash is its `model_id`, which keys artifacts, predictions and ledger rows.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync                      # install
cp .env.example .env         # set FPLT_DATA_DIR, FPLT_ENTRY_ID; for live squad also FPLT_FPL_SESSION
make check                   # lint, types, fast tests
make dagster-dev             # Dagster UI at http://localhost:3000
```

`FPLT_FPL_SESSION` is the `pl_profile` cookie from fantasy.premierleague.com. The live squad (selling prices, bank, free transfers) is loaded via `fplt.data.fpl_client.load_current_team` from authenticated `/api/my-team/{entry_id}/`. Without the cookie, use a saved picks JSON with `fplt plan --entry-json`, or public finished-gameweek picks (no selling prices). Never commit a real session cookie.

## Common tasks

| Task | Command |
|---|---|
| Lint, type-check, fast tests | `make check` |
| Full tests + coverage | `make test` |
| Dagster UI (dev) | `make dagster-dev` |
| Backtest a model | `uv run fplt backtest --model dc_layered_v1 --season 2025-26` |
| Predict next gameweeks | `uv run fplt predict --model dc_layered_v1 --horizon 5` |
| Recommend transfers | `uv run fplt plan --model dc_layered_v1` |
| Leaderboard app | `make leaderboard` |

## Adding things

- **A model**: implement the contract (or components), register it with `@register_model`, add a YAML spec to `configs/models/`. It appears as Dagster assets and on the leaderboard automatically after a backtest.
- **A feature**: add a registered function under `src/fplt/features/` with a version. Change its logic → bump the version.
- **A season**: add `configs/scoring/<season>.yaml` and the season partition.

## Repo layout

```
configs/         model specs, per-season scoring rules, squad state overrides
docs/            design document
scripts/         standalone snapshot capture (runs on GitHub Actions)
src/fplt/        library: data, schemas, features, scoring, models, backtest, optimise, orchestration
apps/            Streamlit leaderboard
external/        isolated environments for external models (OpenFPL)
deploy/          launchd service templates, docker-compose
tests/           unit, contracts (incl. leakage), integration, recorded fixtures
```

## Data

Data lives in `$FPLT_DATA_DIR` (default `./data`, gitignored). Raw FPL API snapshots are captured by `.github/workflows/snapshot.yml` and stored on the `snapshots` branch; local Dagster ingests them. Historical seasons come from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League).

## Running unattended

`deploy/launchd/` contains service definitions for `dagster-webserver` and `dagster-daemon` so sensors run whenever the machine is on. The post-gameweek sensor processes every missing gameweek, so downtime self-heals; the pre-deadline refresh can also be launched manually from the UI.
