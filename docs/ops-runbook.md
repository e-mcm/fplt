# Ops runbook

Operator playbook for live FPL planning alerts. Each **ops-NN** section is the
anchor an ntfy / SMTP message should name (see `fplt.automation.runbook`).
Issue list: [`proposals/ops-hardening-issues.md`](proposals/ops-hardening-issues.md).
Cadence and secrets: [`../deploy/README.md`](../deploy/README.md).

Urgency is always relative to the **next** staged deadline
(`gameweeks.deadline_time`). When in doubt, treat anything inside T−3h as
urgent.

---

## Alerts

### ops-01: Token expiry

| | |
|---|---|
| **Means** | `FPLT_FPL_API_TOKEN` is missing, unparseable, expired, or expires before the next deadline (or within the warn window). Live squad / `fplt apply` needs a fresh JWT from the browser. |
| **Urgency** | **Urgent** if `exp` is before the next deadline or already past on a pre-deadline run. **Warning** if expiry is within ~3 days but after the deadline. |
| **Fix** | `uv run fplt auth from-browser --entry-id YOUR_ID` (or `uv run fplt auth set --token …`) |

Planning still runs when the token is dead; apply must be done by hand until
refreshed. Check: `uv run fplt auth status`.

### ops-02: No plan

| | |
|---|---|
| **Means** | Inside the pre-deadline window a production model still has no plan for this deadline (missed T−24h / T−6h checkpoint, or the T−3h catch-up retry failed). By T−2h this is the urgent alarm. |
| **Urgency** | **Urgent** at/after T−2h with no plan. **High** at T−3h (catch-up should already be firing). |
| **Fix** | `gh workflow run ops-tick.yml -f force_live=true` — or locally `make ops-catchup` / `uv run fplt ops-catchup` |

Confirm `OPS_LIVE=true` on the repo before expecting GHA to plan. See
[`../deploy/README.md`](../deploy/README.md) (ops-tick / ops-catchup).

### ops-03: Workflow failure

| | |
|---|---|
| **Means** | A scheduled or dispatched GitHub Actions job (or a Dagster run-failure sensor) failed. The alert carries the workflow, job, and run URL — no log excerpts (public repo). |
| **Urgency** | **Urgent** if the failing job is `ops-tick` / `ops-catchup` inside T−6h. Otherwise **high** until the next successful capture / tick. |
| **Fix** | Open the run URL in the alert, fix the cause, then `gh run rerun <run-id> --failed` (or re-dispatch the workflow). |

Composite action: `.github/actions/notify-failure`. Dagster path:
`notify_run_failure` → same runbook section.

### ops-03: Heartbeat missing

| | |
|---|---|
| **Means** | healthchecks.io (or equivalent) did not receive a success ping within period + grace. Usually a disabled workflow, a hung job, or the 60-day public-repo schedule silence. |
| **Urgency** | **Urgent** for `fplt-ops-tick` / `fplt-snapshot` inside T−6h. Otherwise **high**. |
| **Fix** | `gh workflow enable <name>.yml` (then `gh workflow run <name>.yml` once) |

Heartbeat table and secrets: [`../deploy/README.md`](../deploy/README.md#capture-job-heartbeats-and-failure-alerts-ops-03--a7).
Do not add a keepalive workflow; heartbeats are the dead-man's switch.

### ops-04: Bad release

| | |
|---|---|
| **Means** | Ops is running a bad `prod` tag (failed smoke plan, freeze-window violation caught late, or a regression after promote). Rollback restores the previous immutable tag. |
| **Urgency** | **Urgent** before the next deadline if live plans are wrong. After kickoff, roll back before the next tick. |
| **Fix** | `gh workflow run release.yml -f ref=<previous-prod-tag>` (or locally `make release ARGS='--ref prod-YYYYMMDD-<sha>'`) |

Immutable history tags look like `prod-YYYYMMDD-<shortsha>`. List with
`git tag -l 'prod-*' | sort | tail`. Confirm the sha in plan
`code_sha.txt` / `metadata.json` after the rollback lands. Do not
force-push `main` to roll back ops — move `prod` only. Promote a new sha
the same way (`ref=main`); refused during freeze, on red CI, or when smoke
fails.

### ops-05: Apply policy

| | |
|---|---|
| **Means** | Transfer approval / veto path: T−6h plan needs a human yes (hits, chips, >2 transfers, kill switch), apply was skipped, post-apply squad mismatch, or a veto was recorded. |
| **Urgency** | **Urgent** between T−6h and the deadline if transfers still need applying. After the deadline, do not apply. |
| **Fix** | `uv run fplt apply --confirm` (after `uv run fplt auth status` is ok) |

Global kill: repo variable `FPLT_APPLY_KILL=1`. Policy lives in
`configs/ops/production.yaml` once ops-05 lands. Recommendation pushes for a
normal auto plan also point here / [`#recommendation`](#recommendation).

### ops-06: Quarantine

| | |
|---|---|
| **Means** | A new raw capture failed a sanity gate (player count, null share, fixtures, monotone deadlines, `data_checked` flip). Staging skips it; planning uses the last good capture. |
| **Urgency** | **Urgent** if the quarantine blocks the only pre-deadline capture inside T−6h. Otherwise **warning** (one alert per new failure type). |
| **Fix** | `uv run fplt data gates --since <iso-utc>` |

Thresholds: `configs/ops/data_gates.yaml`. Raw stays write-once; do not delete
the capture. Clear or accept quarantine only after understanding the gate.

### ops-07: Season rollover

| | |
|---|---|
| **Means** | Preflight failed for a new season (missing scoring/chips YAML, unmapped teams/players, ModelSpec load error, or dry-run plan failure). Also the weekly in-season unmapped-player watch. |
| **Urgency** | **Blocking** before GW1. During the season, **warning** until id maps are patched. |
| **Fix** | `uv run fplt preflight --season YYYY-YY` |

Then `uv run fplt idmap suggest --source understat` (or `api_football`) and
review the candidate PR. Full checklist: [`#season-rollover`](#season-rollover).

### ops-08: Weekly report

| | |
|---|---|
| **Means** | Post-GW decision scoring finished (or failed). Summary is the weekly regret / captaincy / hit-payback screen. |
| **Urgency** | **Info** after `data_checked`. **Warning** only if report generation failed for a checked GW. |
| **Fix** | Re-run the post-GW report job (once wired) or regenerate `reports/gw=<n>/weekly.md` from the ledger |

Idempotent: a rerun must not double-notify.

### ops-09: Model drift

| | |
|---|---|
| **Means** | Live rolling RMSE / event Brier for the production model sat outside the backtest band for *k* consecutive checked gameweeks. Alert only — no auto model switch. |
| **Urgency** | **Warning** mid-season. Escalate to a promote-eval before the next `prod` move if drift persists. |
| **Fix** | Open the drift report / ledger window, then run a promote-eval before changing `configs/ops/production.yaml` |

Governance: production ModelSpecs are immutable; a change needs a new spec
name plus a linked promote-eval `run_id`.

### ops-10: Status

| | |
|---|---|
| **Means** | `fplt status` / `status.json` reports a hard problem (stale capture, missing plan, expired token, active quarantine, last error). Written each ops tick to R2. |
| **Urgency** | Follow the nested field (token → ops-01, quarantine → ops-06, …). |
| **Fix** | `uv run fplt status --json` |

Use the public/private field split; never put squad secrets in the public copy.

### ops-11: Secret rotation

| | |
|---|---|
| **Means** | Scheduled reminder that an ops secret (R2 token, API-Football key, …) is due for rotation within ~7 days, or a workflow job is missing `timeout-minutes`. |
| **Urgency** | **Warning** until rotated. Not deadline-coupled unless the secret is already failing live jobs. |
| **Fix** | Rotate the named secret in GitHub / Cloudflare / API-Sports, then update the repo secret |

Inventory and cadence: [`../deploy/README.md`](../deploy/README.md).

---

## Recommendation

Informational plan push from `ops-tick` / `ops-catchup` / Dagster
recommendation assets (`notify_recommendation`). Not a failure alert.

| | |
|---|---|
| **Means** | A new production plan `report.md` was written and pushed to ntfy/SMTP. |
| **Urgency** | **Info**. Escalate via [#ops-05-apply-policy](#ops-05-apply-policy) when the plan needs approval or apply. |
| **Fix** | Read the report under `$FPLT_DATA_DIR/plans/…/report.md` (no command required) |

---

## Season rollover

Pre-season checklist (ops-07). Run before the first live deadline of a new
season.

1. Confirm `configs/scoring/<season>.yaml` and `configs/chips/<season>.yaml`.
2. `uv run fplt preflight --season YYYY-YY` — fix every failure it prints.
3. Map all 20 teams in Understat and API-Football id maps; suggest players:
   `uv run fplt idmap suggest --source understat` then `--source api_football`.
4. Review the high-confidence id-map PR; merge only after a human check.
5. Load production `ModelSpec`s for the new season; dry-run plan on the first
   post-launch capture.
6. Confirm GW1 `deadline_time` in staged `gameweeks` and that `prod` points at
   a green release outside the freeze window.
7. Re-run preflight weekly during the season for signings / loans.

---

## Run a gameweek by hand

Total-outage fallback when GHA / Dagster / heartbeats are down. Same path as
[`next-gw-actions.md`](next-gw-actions.md) (live squad) or
[`blank-week-prod-plan.md`](blank-week-prod-plan.md) (empty draft).

```bash
# 0) Secrets + token (deadline morning)
uv run fplt auth from-browser --entry-id YOUR_ID
uv run fplt auth status

# 1) Pull latest raw / restage / side feeds / catch up plans
make ops-catchup
# or: uv run fplt ops-catchup

# 2) If catch-up cannot plan, resolve next deadline and plan explicitly
MODEL=$(uv run python -c "from fplt.automation.production import load_production_ops; print(load_production_ops().models[0])")
# set SEASON / GW from resolve_as_of (see next-gw-actions.md) — do not reuse README examples
uv run fplt predict --model "$MODEL" --season "$SEASON" --gw "$GW" --horizon 3
uv run fplt plan    --model "$MODEL" --season "$SEASON" --gw "$GW" --horizon 3

# 3) Read week-1 actions
# $FPLT_DATA_DIR/plans/<model_id>/as_of=<season>_gwNN_h03/report.md

# 4) Apply only when you intend to POST
make apply SEASON=$SEASON GW=$GW MODEL=$MODEL HORIZON=3           # dry-run
make apply SEASON=$SEASON GW=$GW MODEL=$MODEL HORIZON=3 CONFIRM=1 # POST
```

If only GHA is down but the laptop is fine, `make ops-tick` is the lighter
refresh inside the 3-hour window. Design §8.6 and [`../deploy/README.md`](../deploy/README.md)
cover always-on and compose setups.
