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
cp .env.example .env         # optional: FPLT_DATA_DIR only — never paste the API token here
uv run fplt auth from-browser --entry-id YOUR_ID   # pull token from local Chrome/Firefox
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

**Live team auth:** `FPLT_FPL_API_TOKEN` is the Bearer JWT FPL sends as `x-api-authorization` on `my-team` (typically ~8 hours). It lives in the browser’s **localStorage** (`oidc.user:*`), not in cookies — legacy `pl_profile` is not enough. **Never paste a token into the repo `.env`.** Prefer:

```bash
# Log into fantasy.premierleague.com in Chrome/Firefox, then (close the browser if the DB is locked):
uv run fplt auth from-browser --entry-id YOUR_ID
uv run fplt auth status                    # shows expiry; warns when expired
uv run fplt auth clear                     # drops the token
# Fallback if needed: uv run fplt auth set --entry-id YOUR_ID
```

Supported locally: Chrome / Chromium / Edge / Brave / Firefox on macOS and Linux. Safari is not supported. Settings load `~/.config/fplt/env` (or `$FPLT_ENV_FILE`) before the repo `.env`. Without a token, use `--entry-json` or public finished-GW picks (no selling prices). Missing ntfy/SMTP config is a no-op.

## Common tasks

| Task | Command |
|---|---|
| Lint, type-check, fast tests | `make check` |
| Full tests + coverage | `make test` |
| Dagster UI (dev) | `make dagster-dev` |
| Backtest models | `uv run fplt backtest --models dc_layered_v1 lgbm_direct_v1 --season 2025-26` |
| Compare models (walk-forward + fixed) | `make compare-models SEASON=2024-25` or `uv run fplt compare --season 2024-25` |
| Layered component metrics + `n_sims` sweep | `make eval-components SEASON=2024-25` or `uv run fplt eval-components --season 2024-25` |
| Multi-model eval → ledger | `make eval-board SEASON=2024-25` |
| Prediction championship (multi-arm bakeoff) | `make bakeoff BAKEOFF=example_prediction_championship` or `uv run fplt bakeoff --models … --control … --season …` |
| Decision bakeoff (season points) | `make decisions-board SEASON=2024-25` (after `fplt backtest … --steps` matching `--horizon`) |
| Predict next gameweeks | `uv run fplt predict --model dc_layered_v1 --horizon 5` |
| Recommend transfers | `make plan SEASON=2025-26 GW=10` (or `uv run fplt plan --model … --season … --gw …`) |
| Verbose CLI logs | `uv run fplt -v plan …` / `uv run fplt backtest -v …` (DEBUG; default is INFO) |
| Recommend transfers (saved picks) | `uv run fplt plan --model dc_layered_v1 --season 2025-26 --gw 10 --entry-json picks.json` |
| Draft from empty (today's budget) | `uv run fplt plan --model … --season … --gw … --rebuild` (bank + selling prices; needs `my-team` / `--entry-json`) |
| Apply plan to FPL (dry-run) | `make apply SEASON=2025-26 GW=10` or `uv run fplt apply --model … --season … --gw …` |
| Apply plan to FPL (POST) | `make apply SEASON=2025-26 GW=10 CONFIRM=1` (needs `FPLT_FPL_API_TOKEN`) |
| Store live-team token from browser | `uv run fplt auth from-browser --entry-id …` (reads localStorage → `~/.config/fplt/env`) |
| Store live-team token (manual paste) | `uv run fplt auth set --entry-id …` |
| Leaderboard app | `make leaderboard` |

Run `fplt auth set` (or set `FPLT_ENTRY_ID` / `FPLT_FPL_API_TOKEN` in `~/.config/fplt/env`) so `make plan` / `fplt plan` load your live squad with selling prices. Without the token, public finished-gameweek picks have no selling prices. `--entry-json` still wins over the API. `--rebuild` ignores current ownership and drafts a full fifteen with budget = bank + selling prices (same funds as a wildcard). Default model for `make plan` is `rolling_mean_v1`; override with `MODEL=dc_layered_v1`.

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

That clones a sparse checkout into `$FPLT_DATA_DIR/.vaastav` (skipped when `--source` points at an existing vaastav `data/` tree) and copies seasons from `2021-22` onward. Use `--force` to refresh seasons already on disk. Dagster's staged ingest then picks them up via `import_available`. Per-gameweek `value` is staged as `player_status.price` at that GW's deadline (news/chance stay unknown).

### Prediction championship (multi-arm bakeoff)

Thin wrapper on `compare` / promote-eval for championship-style ranking of several ModelSpecs on one shared window (content-addressed ledger cache, SPA/MCS). Descriptive — not a `keep`. Pairwise A/B with a recommendation stays on `fplt promote-eval`. See [`docs/feature-eval-pipeline.md`](docs/feature-eval-pipeline.md) §9 / bakeoff.

```bash
# YAML BakeoffSpec under configs/evals/
uv run fplt bakeoff --bakeoff example_prediction_championship          # dry-run plan
uv run fplt bakeoff --bakeoff example_prediction_championship --execute

# Or CLI models list (≥3 arms including --control)
uv run fplt bakeoff --models rolling_mean_v1 dc_layered_v2 lgbm_direct_v1 \
  --control rolling_mean_v1 --season 2024-25 --from-gw 1 --to-gw 20 --execute
```

### Historical decision bakeoff

Chips-off season sim from an optimal GW1 draft, ranked by realised FPL points (autosub + vice-on-blank; captain ×2; hits once; selling frozen at purchase):

```bash
# 1) Predictions + actuals into the ledger (horizon == decisions --horizon / --steps)
uv run fplt backtest \
  --models rolling_mean_v1 dc_layered_v1 lgbm_direct_v1 \
  --season 2024-25 --from-gw 1 --to-gw 38 --steps 3

# 2) Multi-model decision sim + season-total scoreboard
# Default planning horizon is 3 (match --steps above).
make decisions-board SEASON=2024-25
# or: uv run fplt decisions --season 2024-25 --horizon 3 \
#        --models rolling_mean_v1 dc_layered_v1 lgbm_direct_v1
```

Re-score an existing `ledger/decisions` without re-simulating: `uv run fplt decisions --season 2024-25 --summarise-only`. Single-model path remains `fplt plan --decisions --model …`.

## Running unattended

`deploy/launchd/` contains service definitions for `dagster-webserver` and `dagster-daemon` so sensors run whenever the machine is on. The post-gameweek sensor processes every missing gameweek, so downtime self-heals; the pre-deadline refresh can also be launched manually from the UI.

Wire live-squad and alert secrets (`FPLT_ENTRY_ID`, `FPLT_FPL_API_TOKEN`, `FPLT_NTFY_URL`, SMTP) via `~/.config/fplt/env` (launchd) or `deploy/.env` (compose) — see [`deploy/README.md`](deploy/README.md). Never commit those values.
