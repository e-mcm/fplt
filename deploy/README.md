# Deploy — unattended Dagster + secrets

Always-on templates for `dagster-webserver` and `dagster-daemon`. Secrets for the
live squad and recommendation alerts are injected from the environment; they are
never committed.

**Agents:** alert → fix commands → [`docs/ops-runbook.md`](../docs/ops-runbook.md); cadence and sensor map → [`docs/dagster-cadence-and-setup.md`](../docs/dagster-cadence-and-setup.md); architecture §8 → [`docs/design.md`](../docs/design.md). Live models → `configs/ops/production.yaml` (or `FPLT_PRODUCTION_MODELS` / `FPLT_PLAN_HORIZON`). Do not put secrets in plists or compose committed files.

## What to set

| Variable | Purpose | Required? |
|---|---|---|
| `FPLT_ENTRY_ID` | FPL manager entry for live squad plans | For live team |
| `FPLT_FPL_API_TOKEN` | Bearer token from `x-api-authorization` (selling prices / FT) | For authenticated my-team |
| `FPLT_FPL_SESSION` | Legacy `pl_profile` cookie | Unused by the current API |
| `FPLT_API_FOOTBALL_KEY` | API-Sports key for injuries fetch/stage | For `make fetch-api-football` / GHA |
| `FPLT_PRODUCTION_MODELS` | Comma-separated live model specs (overrides `configs/ops/production.yaml`) | Optional |
| `FPLT_PLAN_HORIZON` | Automation plan horizon (overrides ops YAML; CLI still uses `DEFAULT_PLAN_HORIZON`) | Optional |
| `FPLT_NTFY_URL` | ntfy topic for failure + recommendation alerts | Optional |
| `FPLT_SMTP_HOST` | SMTP host for recommendation email | Optional\* |
| `FPLT_SMTP_PORT` | SMTP port (default `587`) | Optional |
| `FPLT_SMTP_USER` / `FPLT_SMTP_PASSWORD` | SMTP auth | Optional |
| `FPLT_SMTP_FROM` / `FPLT_SMTP_TO` | Envelope addresses | Optional\* |
| `FPLT_SMTP_USE_TLS` | STARTTLS before login (default `true`) | Optional |

\* Email sends only when `FPLT_SMTP_HOST`, `FPLT_SMTP_FROM`, and `FPLT_SMTP_TO`
are all set; otherwise that channel is a no-op (same as missing ntfy).

Canonical commented list: repo-root [`.env.example`](../.env.example). Never put
real cookie or password values in git.

## launchd (macOS always-on)

1. Create the secrets file (mode `600`). Prefer the CLI for the live-team token
   (never paste `FPLT_FPL_API_TOKEN` into the repo `.env`):

   ```bash
   uv run fplt auth from-browser --entry-id YOUR_ID   # local Chrome/Firefox localStorage
   # optional alerts still go in the same file:
   mkdir -p ~/.config/fplt
   chmod 600 ~/.config/fplt/env
   # edit: set FPLT_NTFY_URL and/or SMTP_* (token/entry already set by fplt auth)
   ```

   Or copy `.env.example` to `~/.config/fplt/env` and edit there. Override the
   path with `FPLT_ENV_FILE` if needed. `fplt` Settings also loads this file for
   local CLI use (not only launchd).

2. Copy the plists from `deploy/launchd/`, replace `__FPLT_REPO__` with the
   absolute path to this checkout, and ensure `dagster-webserver` /
   `dagster-daemon` are on `PATH` for the agent (or use absolute paths to the
   `uv`/`venv` binaries).

3. Load the agents:

   ```bash
   launchctl load ~/Library/LaunchAgents/dev.fplt.webserver.plist
   launchctl load ~/Library/LaunchAgents/dev.fplt.daemon.plist
   ```

`run-with-env.sh` sources the secrets file before exec. Paths
(`DAGSTER_HOME`, `FPLT_DATA_DIR`) stay in the plist; session cookie, ntfy, and
SMTP stay out of the plist.

After a successful catch-up or pre-deadline plan, recommendation `report.md` is
pushed when ntfy and/or SMTP are configured. Run failures still POST to
`FPLT_NTFY_URL` when set.

## Post-travel / downtime catch-up (no UI)

When the laptop slept through sensors or the daemon was off, drain the backlog
without opening Dagster:

```bash
make ops-catchup
# equivalent: uv run fplt ops-catchup
```

Order: fetch `snapshots` → restage FPL (+ vaastav if present) → Understat and
API-Football ingest → `catch_up` for every checked GW still missing a production
plan → print counts / pending GWs and `fplt auth status`. Refresh the live token
first if plans should use your squad (`uv run fplt auth status`).

Optional flags via `ARGS`, e.g. `make ops-catchup ARGS='--no-notify'`. Does not
replace a running daemon for ongoing pre-deadline windows; use this as a
one-shot recovery.

## GHA ops-tick / ops-catchup (foundation)

| Workflow | Trigger | Command |
|---|---|---|
| `.github/workflows/ops-tick.yml` | cron `7,22,37,52 * * * *` + `workflow_dispatch` | `fplt ops-tick` |
| `.github/workflows/ops-catchup.yml` | `workflow_dispatch` | `fplt ops-catchup` |

Both share `concurrency: group: ops` (no overlapping ticks). Capture dual-write
(branch + R2) is unchanged. `ops-tick` is deadline-aware (ops-02):
`due_checkpoints` drives T−24h / T−6h / T−90m / T−30m plan+refresh windows
(missed minutes catch up on the next tick; apply at T−90m is stubbed until
ops-05). The no-plan alarm retries at T−3h and sends an urgent ntfy at T−2h
when any production model still lacks a plan. Checkpoint completion state
lives under `$FPLT_DATA_DIR/ops/checkpoint_state.json`.

**Live gate:** repo variable `OPS_LIVE` must be `true` before a scheduled tick
plans on GitHub-hosted runners (avoids a cold cache re-planning every checked
GW). Manual dispatch can pass `force_live=true`. Locally, `make ops-tick` /
`make ops-catchup` always run against `$FPLT_DATA_DIR`.

**Secrets / vars (in addition to capture R2 secrets):**

| Name | Kind | Purpose |
|---|---|---|
| `OPS_LIVE` | variable | `true` to enable planning on GHA |
| `DATA_BUCKET` / `LAB_BUCKET` | variable | R2 bucket names |
| `FPLT_ENTRY_ID` / `FPLT_FPL_API_TOKEN` | secret | live squad (optional) |
| `FPLT_NTFY_URL` | secret | recommendation push from the tick |
| `HC_PING_URL_OPS_TICK` / `HC_PING_URL_OPS_CATCHUP` | secret | heartbeats (ops-03) |
| `HC_PING_URL_AUTH_CHECK` | secret | daily token monitor heartbeat (ops-01) |
| `NTFY_URL` | secret | failure alerts (same as capture) |

Daily token monitor: `auth-check.yml` runs `fplt auth check --allow-missing`
(`17 8 * * *`). Missing secret soft-skips; expired / warn / malformed exits
non-zero and notifies. Live `ops-tick` runs the same check before planning and
keeps producing plans with an `apply manually: token expired` banner when the
JWT is already dead.

```bash
gh workflow run ops-tick.yml -f force_live=true
gh workflow run ops-catchup.yml -f force_live=true -f no_notify=true
```

## docker compose (optional VPS)

1. Build or tag an image as `fplt:local` that contains this repo and its deps.
2. Copy secrets next to the compose file (gitignored):

   ```bash
   cp deploy/.env.example deploy/.env
   chmod 600 deploy/.env
   # edit values
   ```

   Or export the same `FPLT_*` variables in the shell. Compose substitutes them
   into both services, including `FPLT_FPL_API_TOKEN` (live my-team) and optional
   `FPLT_API_FOOTBALL_KEY` (local injuries fetch). Legacy `FPLT_FPL_SESSION` is
   still passed for older setups but unused by the current API.

3. Start:

   ```bash
   docker compose -f deploy/docker-compose.yml up -d
   ```

Container `FPLT_DATA_DIR` remains `/var/lib/fplt` (volume-backed). Do not point
compose at a host `.env` that sets `FPLT_DATA_DIR=./data` unless you also adjust
volumes.

## Heartbeats and failure alerts (ops-03)

Every scheduled workflow (`snapshot`, `understat`, `api_football`,
`football_data`, `ops-tick`, `raw_parity`, `secret-rotation-reminder`,
`auth-check`) plus dispatch-only `ops-catchup` pings a dead-man's switch on
success and POSTs a short ntfy alert on failure.

Missing `HC_PING_URL_*` / `NTFY_URL` secrets are a **soft skip** (log and exit
0). Jobs stay green until Eamon configures healthchecks.io + ntfy; do not treat
absent secrets as CI or schedule failures.

### Composite actions

| Action | When | What |
|---|---|---|
| `.github/actions/heartbeat` | `alerts` job when the main job succeeded | `curl` the check's ping URL |
| `.github/actions/notify-failure` | `alerts` job when the main job failed | ntfy POST with workflow, job, and run URL (no log excerpts) |

`alerts` is a separate job (`needs: <main>`, `if: always()`) with a fresh
checkout. Capture jobs may switch to the `snapshots` branch for publish, which
would otherwise drop `.github/actions/`.

### Repo secrets (never log these)

| Secret | Purpose |
|---|---|
| `NTFY_URL` | ntfy topic URL for failure alerts (same topic healthchecks.io should use) |
| `HC_PING_URL_SNAPSHOT` | healthchecks.io ping URL for `snapshot` |
| `HC_PING_URL_UNDERSTAT` | ping URL for `understat` |
| `HC_PING_URL_API_FOOTBALL` | ping URL for `api_football` |
| `HC_PING_URL_FOOTBALL_DATA` | ping URL for `football_data` |
| `HC_PING_URL_OPS_TICK` | ping URL for `ops-tick` |
| `HC_PING_URL_OPS_CATCHUP` | ping URL for `ops-catchup` |
| `HC_PING_URL_RAW_PARITY` | ping URL for `raw_parity` (dual-write; drop after A9) |
| `HC_PING_URL_SECRET_ROTATION` | ping URL for `secret-rotation-reminder` |
| `HC_PING_URL_AUTH_CHECK` | ping URL for `auth-check` (ops-01 daily token monitor) |

### healthchecks.io check table (configure later)

| Check name | Workflow | Period | Grace | Cron (UTC) |
|---|---|---|---|---|
| `fplt-snapshot` | `snapshot.yml` | 15 min | 45 min | `*/15 * * * *` |
| `fplt-understat` | `understat.yml` | 3 days | 1 day | `30 22 * * 0,1,6` |
| `fplt-api-football` | `api_football.yml` | 6 h | 3 h | `15 */6 * * *` |
| `fplt-football-data` | `football_data.yml` | 35 days | 7 days | `0 6 1 * *` |
| `fplt-ops-tick` | `ops-tick.yml` | 15 min | 45 min | `7,22,37,52 * * * *` |
| `fplt-raw-parity` | `raw_parity.yml` | 8 days | 2 days | `0 6 * * 1` |
| `fplt-secret-rotation` | `secret-rotation-reminder.yml` | 2 days | 1 day | `0 9 * * *` |
| `fplt-auth-check` | `auth-check.yml` | 1 day | 12 h | `17 8 * * *` |
| `fplt-ops-catchup` | `ops-catchup.yml` | n/a (dispatch) | — | manual |

Wire each check's "when down" integration to the same ntfy topic as `NTFY_URL`.
Paste each check's ping URL into the matching `HC_PING_URL_*` secret.

### Forced-failure smoke test

After `NTFY_URL` is set:

```bash
gh workflow run snapshot.yml -f simulate_failure=true
```

Expect exactly one ntfy alert with the Actions run URL. Heartbeat must not ping.

### 60-day scheduled-workflow disable (choice: runbook, not keepalive)

GitHub disables scheduled workflows on public repos after ~60 days without
repository activity. A monthly "re-enable" keepalive does not create that
activity and can itself be disabled, so **ops-03 relies on heartbeats** to
notice silence and documents the fix here (no keepalive workflow; also
[`docs/ops-runbook.md#ops-03-heartbeat-missing`](../docs/ops-runbook.md#ops-03-heartbeat-missing)):

1. healthchecks.io (or ntfy from a missed ping) alerts that a check is down.
2. Open the workflow in the Actions tab → **Enable workflow**, or
   `gh workflow enable <name>.yml`.
3. Optionally push a trivial docs commit so the 60-day inactivity clock resets.
4. Confirm the next scheduled run pings the heartbeat (or re-run via
   `workflow_dispatch`).

## Season rollover checklist (ops-07)

Run before (and weekly during) a new season. Full ops runbook (ops-13) will
absorb this; until then this is the checklist.

1. **Scoring + chips** — add `configs/scoring/<season>.yaml` and
   `configs/chips/<season>.yaml` from the official rules (copy the prior
   season and edit).
2. **Team id maps** — for every promoted / returning club, add entries to
   `configs/understat/id_maps.yaml`, `configs/api_football/id_maps.yaml`, and
   `configs/football_data/team_names.yaml` (FPL **codes**, not season-local
   ids). See those READMEs.
3. **Preflight** — after the first post-launch capture is staged:

   ```bash
   uv run fplt preflight --season 2027-28
   ```

   Checks scoring/chips, all 20 teams mapped per source, unmapped-player
   share, GW1 deadline, production ModelSpecs, and a dry-run plan probe.
4. **Player id maps** — suggest high-confidence joins (review PR; never
   auto-merged):

   ```bash
   uv run fplt idmap suggest --source understat --season 2027-28 \
     --out /tmp/understat-candidates.yaml
   uv run fplt idmap suggest --source api_football --season 2027-28 \
     --out /tmp/api-football-candidates.yaml
   ```

   Open a PR that merges only the `high_confidence_players` fragment into the
   matching `id_maps.yaml`. Re-run preflight until unmapped share is under
   the threshold in `configs/ops/preflight.yaml`.
5. **Weekly** — re-run `fplt preflight --season …` during the season to catch
   new signings and loans.

## Dependency updates (ops-12 / Renovate)

Weekly Renovate (`renovate.json`) groups updates and never auto-bumps
`fplt-mc` (that pin follows `mc-release` only):

| Group | Scope | Automerge | Labels |
|---|---|---|---|
| `dev tools (patch/minor)` | `dependency-groups.dev` patch/minor | yes, on green CI | `deps:dev` |
| `dev tools (major)` | dev majors | no | `deps:dev` |
| `runtime libraries` | `project.dependencies` | **no** | `deps:runtime`, `equivalence` |

Runtime PRs trigger `.github/workflows/deps-equivalence.yml`, which runs
`fplt deps-equivalence` against the fixed board
`configs/evals/runtime_deps_equivalence.yaml` (`kind: environment`, §7.4).
Merge requires CI green **and** this check; Renovate must not automerge
runtime PRs. Enable full promote-eval execute with repo variable
`DEPS_EQUIVALENCE_LIVE=true` (and R2 / snapshots access as for ops-tick).
Until then the workflow prepares the EvalSpec and soft-skips execute.

**Prod freeze hook (ops-04):** `fplt deps-equivalence --check-freeze-only`
calls `fplt.ops.freeze.in_prod_freeze` when that module exists and fails the
job while a deadline freeze is active. Until ops-04 lands the helper is
absent and the check **soft-skips** (logged). When ops-04 ships, wire the
same hook into Renovate automerge / branch protection so bot PRs never merge
inside the freeze window.

**Owner setup:** install the Renovate GitHub App on `e-mcm/fplt`, create
labels `deps:dev`, `deps:runtime`, and `equivalence` if missing, set
`DEPS_EQUIVALENCE_LIVE=true` when staged data is available on runners, and
require the `deps-equivalence` check on runtime PRs.

## Workflow timeouts (ops-11)

Every job under `.github/workflows/` sets `timeout-minutes` (no reliance on
GitHub's 6 h default). Sized at roughly 2–3× expected p95 wall time:

| Workflow / job | `timeout-minutes` | Rationale |
|---|---|---|
| `ops-tick` / `tick` | 90 | > 5× production HiGHS `time_limit_s` (600s) for chip-board solves |
| `ops-catchup` / `catchup` | 180 | Multi-GW catch-up; > 15× one solver budget |
| Capture `capture` jobs | 20–30 | Snapshot / fetch + R2 + branch publish |
| Capture / ops `alerts` | 5 | Heartbeat + ntfy only |
| `ci` lint / tests / builds | 5–90 | Lint 20; production-governance 10; linux test 60; macOS mc build 90 |
| `mc-release` | 15–90 | Same order as CI mc builds |
| `deps-equivalence` | 180 | Runtime equivalence board (ops-12) |
| `raw-parity` | 30 | Branch tip vs R2 one-way check |
| `secret-rotation-reminder` / `remind` | 10 | Issue create only |
| `secret-rotation-reminder` / `alerts` | 5 | Heartbeat + ntfy |

CI lint: `uv run python scripts/check_workflow_timeouts.py` (also in
`make check`). A missing or non-positive `timeout-minutes` fails the job.

## Secret inventory (ops-11)

Where each secret lives, who it is scoped to, and when to rotate. Calendar
dates live in `configs/ops/secret_rotation.yaml`; the
`secret-rotation-reminder` workflow opens a GitHub issue a week before each
`next_due`. Never log secret values.

### GitHub Actions (repo → Settings → Secrets and variables → Actions)

| Name | Kind | Scope / used by | Rotation |
|---|---|---|---|
| `R2_ACCOUNT_ID` | secret | Capture + ops + raw-parity (Cloudflare account id) | Yearly with R2 tokens |
| `R2_ACCESS_KEY_ID` | secret | Same; `fplt-gha-writer` Access Key ID | Yearly |
| `R2_SECRET_ACCESS_KEY` | secret | Same; `fplt-gha-writer` secret | Yearly |
| `API_FOOTBALL_KEY` | secret | `api_football.yml` (aliases below also accepted) | On API-Sports plan change / yearly review |
| `APISPORTS_KEY` | secret | Alias for API-Football key | Same |
| `FPLT_API_FOOTBALL_KEY` | secret | Alias matching local Settings name | Same |
| `FPLT_ENTRY_ID` | secret | `ops-tick` / `ops-catchup` live squad | When entry changes |
| `FPLT_FPL_API_TOKEN` | secret | Live my-team JWT (~8 h expiry) | Continuous (ops-01 / `fplt auth`); not on the yearly calendar |
| `FPLT_NTFY_URL` | secret | Recommendation push from ops tick | Yearly / if leaked |
| `NTFY_URL` | secret | Failure alerts from capture + ops `alerts` jobs | Yearly / if leaked |
| `HC_PING_URL_SNAPSHOT` | secret | healthchecks.io ping for `snapshot` | Yearly / if leaked |
| `HC_PING_URL_UNDERSTAT` | secret | Ping for `understat` | Yearly / if leaked |
| `HC_PING_URL_API_FOOTBALL` | secret | Ping for `api_football` | Yearly / if leaked |
| `HC_PING_URL_FOOTBALL_DATA` | secret | Ping for `football_data` | Yearly / if leaked |
| `HC_PING_URL_OPS_TICK` | secret | Ping for `ops-tick` | Yearly / if leaked |
| `HC_PING_URL_OPS_CATCHUP` | secret | Ping for `ops-catchup` | Yearly / if leaked |
| `HC_PING_URL_RAW_PARITY` | secret | Ping for `raw_parity` (dual-write; drop after A9) | Yearly / if leaked |
| `HC_PING_URL_SECRET_ROTATION` | secret | Ping for `secret-rotation-reminder` | Yearly / if leaked |
| `GITHUB_TOKEN` | auto | Default Actions token (`mc-release` Pages publish; reminder issues) | GitHub-managed |

### GitHub Actions variables (not secrets)

| Name | Purpose |
|---|---|
| `DATA_BUCKET` | R2 data bucket name (`fplt-data`) |
| `LAB_BUCKET` | R2 lab bucket name (`fplt-lab`) |
| `OPS_LIVE` | `true` enables planning on GHA ops runners |
| `FPLT_PRODUCTION_MODELS` | Optional live model override |
| `FPLT_PLAN_HORIZON` | Optional plan horizon override |
| `DEPS_EQUIVALENCE_LIVE` | `true` enables ops-12 equivalence execute on runners |

### Local / launchd / docker (`~/.config/fplt/env` or `deploy/.env`)

| Name | Scope | Rotation |
|---|---|---|
| `FPLT_ENTRY_ID` / `FPLT_FPL_API_TOKEN` | Owner Mac / VPS; mode `600` | Token ~8 h (prefer `fplt auth set`) |
| `FPLT_API_FOOTBALL_KEY` | Local injuries fetch | On plan change |
| `FPLT_R2_ACCOUNT_ID` / `FPLT_R2_ACCESS_KEY_ID` / `FPLT_R2_SECRET_ACCESS_KEY` | Owner token (`fplt-owner`) for both buckets | Yearly with R2 tokens |
| `FPLT_DATA_BUCKET` / `FPLT_LAB_BUCKET` | Bucket names | Rarely |
| `FPLT_NTFY_URL` / `FPLT_SMTP_*` | Alerts | Yearly / if leaked |

### Modal / cloud agents

| Name | Scope | Rotation |
|---|---|---|
| Modal Secret with `fplt-agent` R2 keys | Object Read on `fplt-data`; Read & Write on `fplt-lab` | Yearly with R2 tokens |
| Cloud-agent env (same agent token) | Same permissions; never write to `fplt-data` | Yearly |

Token layout: `docs/proposals/r2-data-bucket-plan.md` §2.3. After rotating R2,
bump `next_due` for `r2-tokens` (and usually `ntfy-and-heartbeats`) in
`configs/ops/secret_rotation.yaml`.

## Spend caps — owner steps (ops-11)

**Agents do not set these.** The owner should configure billing limits in each
provider UI so a runaway job cannot silently rack up cost:

1. **Modal workspace budget**
   - Open the Modal dashboard → Settings / Billing (workspace).
   - Set a workspace spending limit or budget alert at a level you are
     willing to lose on a bad week of experiment runs.
   - Confirm the workspace used by `fplt exp` / lab arms is the one capped.

2. **Cloudflare R2 billing notifications**
   - Cloudflare dashboard → Billing → Notifications (or R2 analytics).
   - Add an alert when R2 storage or Class A/B operations approach the free
     tier (or your paid plan threshold).
   - Buckets: `fplt-data` (immutable raw/curated) and `fplt-lab` (runs /
     experiments). No lifecycle deletes on locked prefixes — growth is
     expected; the alert is the safety net.

Record the chosen limits somewhere outside git if needed; this README only
documents *that* they must be set, not the dollar amounts.
