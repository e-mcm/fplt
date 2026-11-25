# Deploy — unattended Dagster + secrets

Always-on templates for `dagster-webserver` and `dagster-daemon`. Secrets for the
live squad and recommendation alerts are injected from the environment; they are
never committed.

**Agents:** cadence and sensor map → [`docs/dagster-cadence-and-setup.md`](../docs/dagster-cadence-and-setup.md); architecture §8 → [`docs/design.md`](../docs/design.md). Live models → `configs/ops/production.yaml` (or `FPLT_PRODUCTION_MODELS` / `FPLT_PLAN_HORIZON`). Do not put secrets in plists or compose committed files.

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
(branch + R2) is unchanged. Deadline checkpoints and the no-plan alarm are
ops-02 follow-ons; this foundation only runs the existing idempotent cycle.

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
| `NTFY_URL` | secret | failure alerts (same as capture) |

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

## Capture-job heartbeats and failure alerts (ops-03 / A7)

Scheduled capture workflows (`snapshot`, `understat`, `api_football`,
`football_data`) and the ops workflows (`ops-tick`, `ops-catchup`) ping a
dead-man's switch on success and POST a short ntfy alert on failure.

### Composite actions

| Action | When | What |
|---|---|---|
| `.github/actions/heartbeat` | `alerts` job when `capture` succeeded | `curl` the check's ping URL |
| `.github/actions/notify-failure` | `alerts` job when `capture` failed | ntfy POST with workflow, job, and run URL (no log excerpts) |

`alerts` is a separate job (`needs: capture`, `if: always()`) with a fresh
checkout. The capture job may switch to the `snapshots` branch for publish,
which would otherwise drop `.github/actions/`.

Missing ping / ntfy secrets are a no-op so the workflows stay green before A8
configures healthchecks.io.

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

### healthchecks.io check table (configure in A8)

| Check name | Workflow | Period | Grace | Cron (UTC) |
|---|---|---|---|---|
| `fplt-snapshot` | `snapshot.yml` | 15 min | 45 min | `*/15 * * * *` |
| `fplt-understat` | `understat.yml` | 3 days | 1 day | `30 22 * * 0,1,6` |
| `fplt-api-football` | `api_football.yml` | 6 h | 3 h | `15 */6 * * *` |
| `fplt-football-data` | `football_data.yml` | 35 days | 7 days | `0 6 1 * *` |
| `fplt-ops-tick` | `ops-tick.yml` | 15 min | 45 min | `7,22,37,52 * * * *` |
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
activity and can itself be disabled, so **A7 relies on heartbeats** to notice
silence and documents the fix here:

1. healthchecks.io (or ntfy from a missed ping) alerts that a capture check is
   down.
2. Open the workflow in the Actions tab → **Enable workflow**, or
   `gh workflow enable <name>.yml`.
3. Optionally push a trivial docs commit so the 60-day inactivity clock resets.
4. Confirm the next scheduled run pings the heartbeat (or re-run via
   `workflow_dispatch`).

Do not add a keepalive workflow in this stage. Full ops-03 may revisit this if
ops-tick needs a different policy.
