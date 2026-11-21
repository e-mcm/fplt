# fplt

Fantasy Premier League points prediction and team selection.

`fplt` predicts points per player per gameweek with composable, versioned models, benchmarks them against each other and external models (e.g. OpenFPL) on point-in-time walk-forward backtests, and turns predictions into squad, captaincy and transfer recommendations. Dagster runs the whole cycle automatically after each gameweek and before each deadline.

- Architecture: [`docs/design.md`](docs/design.md)
- OpenFPL benchmark setup: [`docs/openfpl.md`](docs/openfpl.md)
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
cp .env.example .env         # set FPLT_DATA_DIR (+ entry / session / alerts as needed)
make check                   # lint, types, fast tests
make dagster-dev             # Dagster UI at http://localhost:3000
```

Useful env vars (all optional except where noted; see `.env.example`):

| Variable | Purpose |
|---|---|
| `FPLT_DATA_DIR` | Data root (default `./data`) |
| `FPLT_ENTRY_ID` | Manager entry for live-team plans (`fplt plan --from-api` / automation) |
| `FPLT_FPL_API_TOKEN` | Bearer token for authenticated `/api/my-team/{id}/` (selling prices, bank, FT) |
| `FPLT_FPL_SESSION` | Legacy `pl_profile` cookie; the live API uses `FPLT_FPL_API_TOKEN` |
| `FPLT_NTFY_URL` | ntfy topic for failure + recommendation alerts |
| `FPLT_SMTP_HOST` / `FPLT_SMTP_FROM` / `FPLT_SMTP_TO` | Recommendation email (all three required or email is a no-op) |
| `FPLT_SMTP_PORT` / `FPLT_SMTP_USER` / `FPLT_SMTP_PASSWORD` / `FPLT_SMTP_USE_TLS` | Optional SMTP details |

`FPLT_FPL_API_TOKEN` is the Bearer value from the `x-api-authorization` header on a logged-in `my-team` request (DevTools → Network). Paste the token only, not the `Bearer ` prefix. It expires about every 8 hours. Without it, use a saved picks JSON (`fplt plan --entry-json …`) or public finished-gameweek picks via `--from-api` / `FPLT_ENTRY_ID` (no selling prices). Never commit a real token. Missing ntfy/SMTP config is a no-op.

## Common tasks

| Task | Command |
|---|---|
| Lint, type-check, fast tests | `make check` |
| Full tests + coverage | `make test` |
| Dagster UI (dev) | `make dagster-dev` |
| Backtest models | `uv run fplt backtest --models dc_layered_v1 lgbm_direct_v1 --season 2025-26` |
| Compare models (walk-forward + fixed) | `make compare-models SEASON=2024-25` or `uv run fplt compare --season 2024-25` |
| Multi-model eval → ledger | `make eval-board SEASON=2024-25` |
| Predict next gameweeks | `uv run fplt predict --model dc_layered_v1 --horizon 5` |
| Recommend transfers | `make plan SEASON=2025-26 GW=10` (or `uv run fplt plan --model … --season … --gw …`) |
| Recommend transfers (saved picks) | `uv run fplt plan --model dc_layered_v1 --season 2025-26 --gw 10 --entry-json picks.json` |
| Leaderboard app | `make leaderboard` |

Set `FPLT_ENTRY_ID` and `FPLT_FPL_API_TOKEN` in `.env` so `make plan` / `fplt plan` load your live squad with selling prices. Without the token, public finished-gameweek picks have no selling prices. `--entry-json` still wins over the API. Default model for `make plan` is `rolling_mean_v1`; override with `MODEL=dc_layered_v1`.

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

Data lives in `$FPLT_DATA_DIR` (default `./data`, gitignored). Raw FPL API snapshots are captured by `.github/workflows/snapshot.yml` and stored on the `snapshots` branch; local Dagster ingests them.

Historical seasons come from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League). One command places season directories (`2021-22` …) where `import_available` expects them — directly under `$FPLT_DATA_DIR`, each with `gws/merged_gw.csv`:

```bash
make bootstrap-vaastav
# or: uv run python scripts/bootstrap_vaastav.py
```

That clones a sparse checkout into `$FPLT_DATA_DIR/.vaastav` (skipped when `--source` points at an existing vaastav `data/` tree) and copies seasons from `2021-22` onward. Use `--force` to refresh seasons already on disk. Dagster's staged ingest then picks them up via `import_available`.

## Running unattended

`deploy/launchd/` contains service definitions for `dagster-webserver` and `dagster-daemon` so sensors run whenever the machine is on. The post-gameweek sensor processes every missing gameweek, so downtime self-heals; the pre-deadline refresh can also be launched manually from the UI.

Wire live-squad and alert secrets (`FPLT_ENTRY_ID`, `FPLT_FPL_API_TOKEN`, `FPLT_NTFY_URL`, SMTP) via `~/.config/fplt/env` (launchd) or `deploy/.env` (compose) — see [`deploy/README.md`](deploy/README.md). Never commit those values.
