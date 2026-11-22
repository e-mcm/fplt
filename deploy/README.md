# Deploy — unattended Dagster + secrets

Always-on templates for `dagster-webserver` and `dagster-daemon`. Secrets for the
live squad and recommendation alerts are injected from the environment; they are
never committed.

## What to set

| Variable | Purpose | Required? |
|---|---|---|
| `FPLT_ENTRY_ID` | FPL manager entry for live squad plans | For live team |
| `FPLT_FPL_API_TOKEN` | Bearer token from `x-api-authorization` (selling prices / FT) | For authenticated my-team |
| `FPLT_FPL_SESSION` | Legacy `pl_profile` cookie | Unused by the current API |
| `FPLT_API_FOOTBALL_KEY` | API-Sports key for injuries fetch/stage | For `make fetch-api-football` / GHA |
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
