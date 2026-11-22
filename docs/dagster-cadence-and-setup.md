# Dagster cadence and setup

Answers for Eamon: which jobs run when, why Dagster (vs cron / Prefect / Airflow), and how to run it always-on. Based on `main` (`src/fplt/orchestration/`, `src/fplt/automation/cycle.py`, `deploy/`, `.github/workflows/`, `docs/design.md` §8).

---

## 1. Cadence map

Two layers: **remote capture** (GitHub Actions → `snapshots` branch) and **local/VPS orchestration** (Dagster sensors → ingest + plan). Remote capture never needs your machine. Sensors need a running `dagster-daemon`.

### Capture (no Dagster; runs on GitHub)

| Workflow | Cadence | What | Why that cadence |
|---|---|---|---|
| `.github/workflows/snapshot.yml` | Every **15 min** (`*/15 * * * *`) + manual | FPL API JSON → `snapshots` branch under `data/raw/fpl_api/...` | Poll often; `scripts/snapshot.py` **throttles writes**: ≤15m in the last 3h before a deadline, ≤1h in the 3–6h window, else when the last capture is >12h old. After `data_checked` also captures `event/{gw}/live` |
| `.github/workflows/api_football.yml` | Every **6h** (`15 */6 * * *`) + manual | API-Football `/injuries` JSON → `data/raw/api_football/...` | Free tier ~100 req/day; fetch also skips if &lt;4h old. Needs repo secret `API_FOOTBALL_KEY` (or `APISPORTS_KEY` / `FPLT_API_FOOTBALL_KEY`); skips quietly if unset |
| `.github/workflows/understat.yml` | `30 22 * * 0,1,6` (Sat/Sun/Mon 22:30 UTC) + manual | Understat match JSON → same `snapshots` branch | Match xG settles after weekend kickoffs; not needed continuously |

**Not on GHA:** football-data E0 odds (`make fetch-football-data`), vaastav bootstrap (`make bootstrap-vaastav`), Touchline predicted XI (opt-in `FPLT_TOUCHLINE_LINEUPS_ENABLED`), FPL Review CSV drop.

### Dagster sensors (need daemon; poll locally)

| Sensor | Min interval | Job / assets | Fires when | Why |
|---|---|---|---|---|
| `snapshots_sensor` | **10 min** | `refresh_staged` (`raw_snapshots` → `staged_ingest` → staged tables) | New files fetched from `snapshots` into `$FPLT_DATA_DIR/raw` | Pull remote captures soon after GHA publishes; skip if nothing new |
| `catchup_sensor` | **30 min** | `catch_up` → `recommendation_report` | A checked gameweek has no plan yet for `PRODUCTION_MODELS` @ horizon 3 | Self-heals missed deadlines / downtime; walk-forward fit+plan per missing GW |
| `pre_deadline_sensor` | **15 min** | `pre_deadline` → `pre_deadline_plan` + `benchmark_archive` | Next deadline is **within 3 hours** and plan and/or `benchmark_projections` still missing | Near-deadline plan + archive `ep_next`/OpenFPL |
| `notify_run_failure` | on failure | n/a | Any run fails and `FPLT_NTFY_URL` is set | Ops alert only |

Production live list today: `PRODUCTION_MODELS = ["dc_layered_v2"]`, horizon `DEFAULT_PLAN_HORIZON = 3` (both in `fplt.automation.cycle` / plan defaults — not env yet).

### Dagster jobs without sensors (manual UI / CLI)

| Job / asset | Purpose | Gap |
|---|---|---|
| `refresh_staged` | Full FPL raw → staged restage | Also triggered by `snapshots_sensor` |
| `understat_ingest` | Stage Understat from `$FPLT_DATA_DIR/raw/understat` + id maps | GHA captures; **no sensor** after pull |
| `api_football_ingest` | Stage injuries from `raw/api_football` + id maps | GHA captures; **no sensor** |
| `match_odds_ingest` | Stage football-data odds | **No GHA + no sensor** |
| `predicted_lineups_ingest` | Stage Touchline / AF lineups | Opt-in capture; **no sensor** |
| `catch_up` / `pre_deadline` | Plan missing checked GWs / imminent deadline | Also triggered by sensors |
| `eval_board` / `lineage_job` | Multi-model ledger / factory lineage | On-demand only |

### Training & planning (what “feeds + training” means)

| Cadence | What happens | Where |
|---|---|---|
| **Continuous (daemon up)** | Pull snapshots → restage FPL (+ vaastav dirs if present) | `snapshots_sensor` |
| **Post-GW (checked)** | `run_backtest` fit/score for production model(s) → `run_plan` (live squad if `FPLT_ENTRY_ID` set) → ntfy/SMTP recommendation | `catchup_sensor` → `catch_up` |
| **Pre-deadline (≤3h)** | Archive benchmarks; plan if missing; notify | `pre_deadline_sensor` |
| **Weekly-ish (manual)** | Understat after weekend; injuries every 6h on GHA but **stage by hand** | GHA + manual jobs |
| **On-demand** | `make eval-board`, `fplt compare`, bakeoffs, `fplt predict` / `fplt plan` | CLI — **no scheduled multi-model train** |

### One-shot / CLI (not sensors)

| Command | When |
|---|---|
| `make bootstrap-vaastav` | Once (or rare): historical seasons under `$FPLT_DATA_DIR` |
| `make fetch-football-data SEASON=…` then materialize `match_odds_ingest` | Per season / when CSV updates |
| `uv run fplt backtest` / `make eval-board` | On demand for train/eval |
| `uv run fplt predict` / `fplt plan` | On demand or until sensors cover your entry |

### Gaps to know

1. Understat / API-Football GHA write to the `snapshots` branch; `snapshots_sensor` fetches them into `raw/`, but **staging** those feeds is separate (`understat_ingest`, `api_football_ingest`) with **no sensor**. After weekend / AF captures, materialize those jobs in the UI (or add sensors later).
2. Odds and predicted lineups are fully manual (or opt-in) today.
3. Laptop sleep through the 3h pre-deadline window misses that tick until wake; catch-up still heals **checked** GWs after the fact.

### Daily / weekly mental model

```
GHA (cloud)                          Always-on host (laptop/VPS)
─────────────────                    ──────────────────────────
15m FPL snapshot ──┐                 dagster-daemon
6h  API-Football ──┼→ snapshots ──→  snapshots_sensor (10m) → refresh_staged
weekend Understat ─┘   branch        catchup_sensor (30m)   → fit+plan missing GWs
                                     pre_deadline_sensor    → archive+plan ≤3h
                                     [manual] understat / api_football / odds ingest
```

---

## 2. Why Dagster here (vs cron / Prefect / Airflow)

fplt already chose Dagster and wired assets + sensors around **PIT tables**, **asset checks**, and **event-ish triggers** (new snapshot files, “GW checked but no plan”, “deadline in 3h”). That is the load-bearing reason — not that other tools cannot schedule a shell script.

| Option | Fit for this repo | Gap vs current design |
|---|---|---|
| **cron / launchd timers alone** | Fine for “run `fplt plan` every hour” | No asset graph, no UI lineage over staged tables, no run history/checks; you reimplement skip logic (new files? deadline window?) in shell |
| **Prefect** | Good Python flows + deployments | Would rewrite sensors/assets; no win unless you already run Prefect |
| **Airflow** | Strong fixed DAGs / teams | Heavier always-on (scheduler + web + workers); overkill for a single-manager PIT store; partitions/DAG style fights “one parquet per table + `available_at`” |
| **Dagster (chosen)** | Assets = staged tables; sensors = pull/catch-up/pre-deadline; checks = pandera-style staged validation; UI for materialize + metadata (plan markdown) | You already depend on it in-tree (`fplt.orchestration.defs`) |

**Practical rule:** use **GHA cron** for unattended *capture* off your laptop; use **Dagster daemon** for *local ingest + deadline-aware planning* that must see `$FPLT_DATA_DIR` and secrets (`FPLT_ENTRY_ID`, `FPLT_FPL_API_TOKEN`). Pure cron can replace Dagster only if you accept losing the asset/check UI and rewrite the sensors as scripts.

You do **not** need Dagster to develop or run CLI paths (`backtest`, `predict`, `plan`).

---

## 3. Setup: yes, always-on for sensors

Sensors only tick while a **Dagster daemon** process is running (plus optional webserver for UI). Without the daemon, nothing auto-ingests or auto-plans. GHA keeps capturing either way; staged data and plans go stale until the daemon returns (then catch-up heals checked GWs).

### Dev (interactive)

```bash
export DAGSTER_HOME=…   # empty dir Dagster can write to
export FPLT_DATA_DIR=…  # your data root
make dagster-dev        # UI http://localhost:3000 — often runs webserver; enable daemon per Makefile/docs
```

Use the UI to materialize assets / launch jobs. For sensors in dev, ensure a daemon is also running (compose or second process).

### Mac always-on (repo-provided)

`deploy/launchd/`:

- `dev.fplt.daemon.plist` → `dagster-daemon run -m fplt.orchestration` (`KeepAlive`)
- `dev.fplt.webserver.plist` → `dagster-webserver -h 127.0.0.1 -p 3000 -m fplt.orchestration`

Both set `DAGSTER_HOME=/usr/local/var/fplt/dagster` and `FPLT_DATA_DIR=/usr/local/var/fplt/data`. Install/load those plists (adjust paths to your uv/venv `dagster*` binaries). Sensors then run whenever the machine is on; catch-up self-heals after downtime. Prefer secrets via `deploy/launchd/run-with-env.sh` + `~/.config/fplt/env` (mode `600`) — see `deploy/README.md`.

Set at least: `FPLT_ENTRY_ID`, `FPLT_FPL_API_TOKEN` (via `uv run fplt auth from-browser --entry-id …` / `fplt auth set`), optional `FPLT_NTFY_URL` / SMTP, and `FPLT_API_FOOTBALL_KEY` if you fetch injuries locally. `FPLT_FPL_SESSION` is legacy and unused by the current API.

### VPS / Docker

`deploy/docker-compose.yml`: **two services** — `webserver` and `daemon`, shared `DAGSTER_HOME` + data volumes, module `-m fplt.orchestration`. Build image `fplt:local` with the repo + deps first. Copy `deploy/.env.example` → `deploy/.env`. Confirm compose passes through `FPLT_FPL_API_TOKEN` (and optional `FPLT_API_FOOTBALL_KEY`) — if those env keys are missing from the compose file, live squad / local AF on VPS will not see them even when set in `.env`.

### Minimum mental model

1. GHA captures → `snapshots` branch (no local process): 15m FPL (throttled writes), 6h API-Football (if secret set), weekend Understat.  
2. Always-on **daemon** polls sensors → refresh staged / catch-up / pre-deadline.  
3. Optional **webserver** for UI.  
4. Manually run `understat_ingest` / `api_football_ingest` (or add sensors later) after those files land in `raw/`.
