# Ops hardening issues: `e-mcm/fplt`

These issues are ordered by expected impact, and that order is also the suggested implementation order. Each `## ISSUE:` section becomes one GitHub issue. File them with `create_issues.py`, which creates the 13 issues in order, then a tracker issue that links them as sub-issues.

**Assumed foundations** (separate work): `ops-tick` / `ops-catchup` on GitHub Actions crons, and the R2 bucket (`r2-data-bucket-plan.md`).

---

## ISSUE: [ops-01] FPL API token expiry monitor
labels: ops, priority:p1

### Problem
`fplt apply --confirm` and live squad loading (`resolve_squad_state`) need `FPLT_FPL_API_TOKEN`, a JWT pasted from DevTools. It expires, and nothing notices until a deadline run fails or silently falls back to the YAML squad. It is the one dependency that can't be automated: scripting the FPL login is off the table.

### Proposal
- Add `fplt auth check --warn-days 3 --deadline-aware`, built on the existing `fplt.auth.inspect_token` / `status`, which already read `exp`. It exits non-zero and notifies (via `fplt.automation.notify`) when:
  - the token is missing or unparseable
  - `exp` is less than `warn_days` away (**warning**)
  - `exp` falls before the next deadline from staged `gameweeks.deadline_time` (**urgent**, with the deadline in the message)
- Run it daily, and at the start of every pre-deadline ops run.
- If the token is expired when a deadline run starts, keep planning. Send the plan with an "apply manually: token expired" banner instead of failing silently.
- The message includes the fix command: `fplt auth set --token …`.

### Acceptance criteria
- [ ] Unit tests cover valid, expiring-soon, expires-before-deadline, expired, missing and malformed tokens, with fixed `now` and a synthetic deadline.
- [ ] Notifications include the time to expiry and the next deadline, never the token itself.
- [ ] An expired token in a pre-deadline run still produces a plan, plus an urgent "manual apply" notification.

### Design refs
`docs/design.md` §8.5 (Automation). `src/fplt/auth.py`, `src/fplt/automation/notify.py`.

### PR split
1. `auth check` command and its tests. 2. Wire it into the daily and pre-deadline workflows, plus the expired-token fallback banner.

---

## ISSUE: [ops-02] Deadline-anchored ops runs and a "no plan by T−3h" alarm
labels: ops, priority:p1

### Problem
The pre-deadline refresh (`fplt.automation.cycle`, `late_refresh`) assumes something runs it every 15 minutes inside the 3-hour window. GitHub Actions cron is best-effort: runs are delayed under load and sometimes dropped, often around the top of the hour. A missed window means stale or missing transfers for a whole gameweek.

### Proposal
- Keep the cheap `*/15` tick. `ops-tick` decides what to do from the deadline in staged `gameweeks`. It stays idempotent, so extra ticks do no harm.
- Add **explicit deadline checkpoints**: T−24h (full plan), T−6h (plan and notify), T−90m (final refresh; apply, if ops-05 allows it), T−30m (last-chance refresh only).
- Add a **no-plan alarm**. At T−3h, if any production model has no plan for the deadline, run catch-up immediately (`workflow_dispatch` retry). If there is still no plan by T−2h, send an urgent notification.
- Use offset minutes (for example `7,22,37,52 * * * *`) to avoid top-of-hour congestion.
- Add `concurrency: ops` so two ticks never overlap.

### Acceptance criteria
- [ ] A pure function `due_checkpoints(now, deadline, done) -> list[Checkpoint]` has table-driven tests: before T−24h, between each checkpoint, a missed checkpoint caught up late, after the deadline, a blank gameweek, and a double gameweek with one deadline.
- [ ] A missed checkpoint runs on the next tick, and is never skipped because its exact minute passed.
- [ ] An integration test with a synthetic deadline and a missing plan triggers retry, then the alarm.

### Design refs
§8.5 (Automation), §10 (Optimiser). `src/fplt/automation/cycle.py`, `late_refresh.py`.

### PR split
1. `due_checkpoints` and its tests. 2. `ops-tick` uses it. 3. Workflow schedule, concurrency group and the alarm step.

---

## ISSUE: [ops-03] Heartbeats, failure alerts and scheduled-workflow keepalive
labels: ops, priority:p1

### Problem
Today run-failure alerts come from a Dagster run-failure sensor. On GitHub Actions, the dangerous failures are silent: a cron that never fires, a disabled workflow, a job that hangs. GitHub also disables scheduled workflows in public repos after 60 days without repository activity. Once snapshots stop being committed to the `snapshots` branch (the R2 cut-over), a quiet summer could trigger this.

### Proposal
- Add a **dead-man's switch** per scheduled job (snapshot, ops-tick, understat, api_football, football_data). Each pings a heartbeat check on success, for example healthchecks.io (free tier), with ping URLs stored as secrets. Expected periods: snapshot 15 min (grace 45), ops-tick 15 min, understat and api_football as scheduled, football_data monthly.
- On a missing ping, the heartbeat service pushes to the same ntfy topic. Configure that once in its UI.
- Add a **failure alert step** to every workflow: `if: failure()` sends an ntfy POST with the workflow, job and run URL. Keep it small, with no log excerpts, since the repo is public.
- Add a **keepalive**: a monthly workflow that calls the GitHub API to re-enable the scheduled workflows. Alternatively, rely on the heartbeat to notice and document the fix in the runbook. Pick one and record the choice.

### Acceptance criteria
- [ ] Every scheduled workflow has a heartbeat ping and an `if: failure()` alert.
- [ ] A forced failure on `workflow_dispatch` (a `simulate_failure` input) produces exactly one ntfy alert with the run URL.
- [ ] Ping URLs and ntfy topics live in secrets and never appear in logs.
- [ ] The heartbeat setup (check names, periods, grace times) is documented in `deploy/README.md`.

### Design refs
§8.5 (Automation). `.github/workflows/*.yml`, `src/fplt/automation/notify.py`.

### PR split
1. A composite action, `.github/actions/notify-failure` plus heartbeat ping, and its use in one workflow. 2. Roll it out to the remaining workflows. 3. Keepalive or a runbook entry.

---

## ISSUE: [ops-04] Run ops from a `prod` tag, with a pre-deadline freeze window
labels: ops, priority:p1

### Problem
Scheduled jobs check out `main` HEAD, so a merge on Friday night is live for Saturday's deadline with no soak time. A regression in the optimiser, PIT or staging code can cost a gameweek, and rolling back means reverting commits under time pressure.

### Proposal
- Ops workflows check out `ref: prod`, a moving tag, instead of the default branch. CI keeps running on `main`.
- Add `make release` / a `release.yml` `workflow_dispatch` that moves `prod` to a commit only if:
  - CI on that commit is green
  - the decision board / smoke plan for the current gameweek succeeds on it (a dry-run plan with the production model)
  - no deadline falls within the **freeze window**: 24h before a deadline, ending after kickoff
- **Rollback:** `release.yml` with `ref=<previous tag>`. Keep `prod-YYYYMMDD-<sha>` immutable tags for history.
- The ops job logs the `prod` sha into each plan's metadata, so every plan can be traced to its code.

### Acceptance criteria
- [ ] The freeze-window function has tests (inside, outside, at the boundary, double gameweek, blank gameweek).
- [ ] The release workflow refuses to move the tag during a freeze, on a red CI, or on a failed smoke plan, each with a clear message.
- [ ] Plans record the code sha; a test asserts the field is present.
- [ ] Runbook entry: "roll back prod".

### Design refs
§8.5, §11 (Repository layout), §13 (Decisions); the `fplt-mc` wheel pin (AGENTS.md workflow).

### PR split
1. Freeze-window function, tests and the sha in plan metadata. 2. `release.yml`. 3. Switch the ops workflows to `ref: prod`.

---

## ISSUE: [ops-05] Transfer approval policy: auto-apply with a veto window
labels: ops, priority:p1, needs-decision

### Problem
`fplt apply --confirm` exists, but nothing decides when it runs. Without a policy, the owner is the approval step every gameweek, which defeats unattended operation.

### Decision needed (owner)
- [ ] Auto-apply at all? (Proposed: yes, for free-transfer-only plans.)
- [ ] Veto window: notify at T−6h, apply at T−90m unless vetoed. (Proposed.)
- [ ] What always needs an explicit "yes": any hit, any chip, more than 2 transfers, captain change vs T−24h plan, plan EV gain below X? (Proposed: hits, chips, and more than 2 transfers.)
- [ ] Veto mechanism: a repo variable `FPLT_APPLY_KILL=1`, a `workflow_dispatch` "veto GW n", or a reply to ntfy via an action button. (Proposed: an ntfy action button that triggers a `workflow_dispatch`, with the repo variable as a global kill switch.)

### Proposal (once decided)
- `ApplyPolicy` (pydantic, in `configs/ops/production.yaml`) with a pure `decide(plan, previous_plan, policy) -> Auto | NeedsApproval(reasons) | Blocked(reasons)`.
- At T−6h, the notification shows the moves, expected-points gain, the reasons approval is needed, and veto/approve buttons.
- At T−90m, apply if the decision is `Auto` and there's no veto, or if it was approved. Otherwise notify "not applied: <reasons>". Never apply after the deadline.
- Afterwards, read back the squad via the API and verify it matches the plan; alert on a mismatch.

### Acceptance criteria
- [ ] `decide` has table-driven tests for every rule, including the kill switch and an approved-hit case.
- [ ] The apply path is idempotent: a second T−90m tick doesn't re-POST.
- [ ] Post-apply verification is tested with recorded API fixtures.

### Design refs
§10.3 (Optimiser output), §8.5. `src/fplt/optimise/apply.py`, `configs/ops/production.yaml`.

---

## ISSUE: [ops-06] Data sanity gates with quarantine and fallback
labels: ops, data, priority:p2

### Problem
pandera checks the schema, but not whether values are plausible. A truncated `bootstrap-static`, a mid-season FPL field rename, or an empty fixtures payload can stage fine and produce a nonsense plan, or crash the deadline run.

### Proposal
- Add `fplt.data.gates`: pure checks on each new capture compared with the previous good one:
  - player count within ±N
  - share of null `now_cost` / `status` / `team` below a threshold
  - next gameweek's fixtures present, with team count 20
  - deadlines monotone
  - `events[].data_checked` never flips from true to false
- On failure, write a **quarantine** record (`raw` stays write-once, so the bad capture isn't deleted). Staging skips quarantined captures, and planning uses the last good capture.
- Alert once per new failure type, not on every tick.
- Add `fplt data gates --since` to explain the current quarantine state.

### Acceptance criteria
- [ ] Every gate has pass and fail tests with synthetic builders, plus a recorded real capture that passes.
- [ ] A quarantined capture never reaches `staged/`; a test proves the PIT view excludes it.
- [ ] Gate thresholds live in config, not code.
- [ ] A new `available_at`-carrying quarantine table gets a schema first, per AGENTS.md.

### Design refs
§4.3 (Storage layout), §4.5 (Snapshot capture), §4.6 (Schemas), §8.4 (Asset checks).

### PR split
1. Quarantine schema. 2. The gates and their tests. 3. Staging skip and alerting.

---

## ISSUE: [ops-07] Season rollover preflight and automated id mapping
labels: ops, data, priority:p2

### Problem
Every summer, FPL element and team ids change; there are promoted teams, a new `configs/scoring/<season>.yaml`, chip rules, and new Understat and API-Football players that must be added to `configs/understat/id_maps.yaml` and `configs/api_football/id_maps.yaml` by hand. Unmapped players stage with a null `player_key`, which degrades features silently.

### Proposal
- `fplt preflight --season 2027-28` checks and reports:
  - scoring and chip rules present
  - all 20 teams mapped in every source
  - unmapped-player share by source, with the top unmapped players by minutes or price
  - the GW1 deadline parsed
  - production `ModelSpec`s load for the new season
  - a dry-run plan on the first post-launch capture succeeds
- `fplt idmap suggest --source understat|api_football` fuzzy-matches unmapped players (name, team, position, DOB where available) to `player_key`, writes candidates with scores, and opens a PR with the high-confidence ones. The owner reviews; nothing is auto-merged.
- During the season, run preflight weekly as well, to catch new signings and loans.

### Acceptance criteria
- [ ] Preflight has tests on a synthetic "new season" fixture with a promoted team, a missing scoring file and unmapped players.
- [ ] The matcher has tests for accents, nicknames, players with the same name at different clubs, and a below-threshold match that is never suggested.
- [ ] A runbook "season rollover" checklist links to preflight.

### Design refs
§4.2 (Identifiers), §5.5 / §5.5.1 (Scoring and chip rules), AGENTS.md §6 (ids change between seasons).

---

## ISSUE: [ops-08] Post-GW decision scoring and weekly report
labels: ops, eval, priority:p2

### Problem
The `data_checked` → ledger → refit path exists, but there's no automatic answer to "how did last week's decisions do?". Finding out is a manual investigation.

### Proposal
After each gameweek is checked:
- Append actuals to the (run-partitioned) ledger, as today.
- Score the **decisions**: realised points vs the plan's expected points; realised points vs the hindsight-best feasible plan under the same constraints (regret); captaincy regret; transfer hit payback to date; points-at-rank percentile (`live_percentile`).
- Write `reports/gw=<n>/weekly.md` and parquet, and send the summary via ntfy/SMTP: one screen with links.
- Keep a season-to-date running table of the same metrics.

### Acceptance criteria
- [ ] Decision-scoring functions are pure and tested on a synthetic gameweek with known optimum and regret.
- [ ] A double gameweek and a blank gameweek are handled; zero-minute players are scored correctly.
- [ ] The report is generated idempotently: a rerun produces the same file and no duplicate notification.

### Design refs
§7.2 (Ledger), §7.3 (Metrics), §7.8 (Live percentile), `docs/evaluation-north-star.md`.

---

## ISSUE: [ops-09] Production model governance and live drift monitor
labels: ops, eval, priority:p2

### Problem
`configs/ops/production.yaml` can be edited without an eval, and production `ModelSpec`s have been edited in place before (`dc_layered_v2`). Nothing watches whether live forecast quality is degrading relative to backtest.

### Proposal
- **Governance:** a CI check fails any PR that changes `configs/ops/production.yaml`, or a `ModelSpec` listed in it, unless the PR links a promote-eval result (ledger `run_id`) with a passing verdict. Production specs are immutable: changes mean a new spec name.
- **Drift monitor:** after each checked gameweek, compute rolling live RMSE and event Brier (minutes ≥ 60, goal, clean sheet) for the production model. Compare with the backtest distribution for the same horizon. Alert when the rolling window sits beyond a threshold for k consecutive gameweeks. Alert only; never switch models automatically.

### Acceptance criteria
- [ ] Governance check tests: unchanged config passes; a changed config without an eval link fails; with a link it passes; an in-place spec edit fails.
- [ ] Drift tests use synthetic ledgers: a stable series never alerts, and a step change alerts after k gameweeks.
- [ ] Thresholds are in config. No model-quality assertions in pytest (AGENTS.md §3); tests check the alerting logic only.

### Design refs
§5.2 (`ModelSpec`), §7.2–7.4 (Ledger, Metrics, Promote-eval), §8.5.

---

## ISSUE: [ops-10] `fplt status` → `status.json`
labels: ops, priority:p3

### Problem
Answering "is everything OK?" means reading Actions logs or SSHing somewhere.

### Proposal
- Add `fplt status --json` covering:
  - last capture time per source
  - last plan per production model and deadline
  - next deadline and its checkpoint state (ops-02)
  - token expiry (ops-01)
  - active quarantines (ops-06)
  - `prod` sha (ops-04)
  - last error
  - last heartbeat-relevant timestamps
- Every ops tick writes it to R2 at `fplt-data/status/status.json`. This is the only mutable object, overwritten each tick, so it is excluded from locks.
- A whitelisted copy can later feed the website or a phone widget. No squad or private data goes in the public copy.

### Acceptance criteria
- [ ] A schema for the status document; tests for each field from a synthetic data dir; missing sources reported as `null` with a reason.
- [ ] A public/private field split is tested.

### Design refs
§8.5. `r2-data-bucket-plan.md` (bucket layout).

---

## ISSUE: [ops-11] Workflow timeouts, spend caps and secret-rotation reminders
labels: ops, priority:p3

### Proposal
- `timeout-minutes` on every job, sized at 2–3× the p95 runtime from `fplt perf-report`. No job can hit GitHub's 6h default.
- Solver wall-clock limits are already in `highs_options`; check that the ops job timeout exceeds the total of the solver limits.
- Spend caps: set a Modal workspace budget or limit, and Cloudflare billing notifications for R2 usage above the free tier.
- Secret inventory in `deploy/README.md`: each secret, where it lives, who it is scoped to, and its rotation cadence (R2 tokens yearly, API-Football key on plan change). Add a scheduled reminder workflow that opens an issue a week before each rotation date.

### Acceptance criteria
- [ ] Every workflow job has `timeout-minutes`. Add a CI lint (a small script) that fails if any job lacks it.
- [ ] A test for the timeout-lint script.
- [ ] The secret inventory is documented.

### Design refs
§8.5; `deploy/README.md`.

---

## ISSUE: [ops-12] Dependency update bot, gated by the equivalence eval
labels: ops, priority:p3

### Proposal
- Renovate (preferred, because it understands `uv.lock`) or Dependabot, running weekly and grouped:
  - patch and minor dev tools
  - runtime libraries
  - `fplt-mc`, which is never auto-bumped and only follows its own release
- Runtime-library PRs carry a label that triggers the **equivalence** promote-eval lever (the `environment` arm) on a fixed board. Merge only with CI green **and** equivalence passing. No auto-merge for runtime libraries; dev tools may auto-merge on green.
- Respect the `prod` freeze window (ops-04): bot PRs never merge inside it.

### Acceptance criteria
- [ ] Bot config is committed, with schedule and groups as above.
- [ ] The label-triggered workflow runs the equivalence eval and posts its verdict on the PR.

### Design refs
§7.4 (Promote-eval), AGENTS.md workflow (`uv add`, never hand-edit `uv.lock`; `fplt-mc` wheel pin).

---

## ISSUE: [ops-13] Ops runbook
labels: ops, docs, priority:p3

### Proposal
Write `docs/ops-runbook.md`: one entry per alert the system can send (ops-01 to ops-11). Each entry gives what it means, how urgent it is relative to the next deadline, and the one command that fixes it. For example:
- token expiring → `fplt auth set`
- no plan by T−2h → `gh workflow run ops-tick.yml`, or locally `fplt ops-catchup`
- capture quarantined → `fplt data gates --since …`
- heartbeat missing → re-enable the workflow
- bad release → `release.yml ref=<previous prod tag>`

Also include a pre-season rollover checklist (ops-07) and "how to run a gameweek fully by hand" for a total-outage fallback.

### Acceptance criteria
- [ ] Every notification string in the code links (or names) its runbook section. A test greps the notify call sites for a runbook anchor.
- [ ] Linked from `docs/README.md` and `deploy/README.md`.

### Design refs
`docs/README.md` doc map; §8.6 (Running locally).
