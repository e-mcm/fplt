# Milestone: scheduled ingest → weekly official runs → R2 → laptop UI

**Status:** proposal · **Date:** 2026-11-27 · **Tracker:** see the tracker issue linked from the PR that adds this file

This is the shortest path to a working weekly loop. It deliberately leaves out experiments, `cutoff_at`, the YAML retirement and the old-path cut-over; none of them block this goal.

## 1. Goal (done when all four hold)

1. **Ingestion into R2 works on scheduled jobs.** Every capture job writes new raw files to `fplt-data/raw/` on its schedule, and fails loudly when it can't.
2. **The weekly prediction and planning workload runs on scheduled jobs.** `ops-tick` runs live and produces the official slate (`v4_h3` primary; `v4_h5` / `v4_h7` / `v4_h9` candidates) at the deadline checkpoints.
3. **Prediction results are written to R2.** Official predict and plan runs, with **sub-model tables** (§3.1), land in `fplt-data/runs/`, with pointers under `fplt-data/official/`.
4. **On the laptop:** `fplt data pull --official latest && fplt ui` opens a **This week** page showing this week's predictions and plan, with a per-player drill-down (minutes, xG/xA, clean sheet, points split, points distribution).

## 2. Where things stand (2026-11-28, after #281 / #291 PRs 1–2)

| Piece | State |
|---|---|
| Capture → R2 | `#278` landed (`environment: r2`); owner checklist M1–M4 still needed for live secrets/clock |
| Schedule | External clock (cron-job.org → `workflow_dispatch`) still owner step M2; GHA cron is backup |
| `ops-tick` | Timing session + `JOB_STARTED_AT` / step summary wired (#291 PR2); official slate finalize still owner-gated (`OPS_LIVE`) |
| Run contents | Layered predict runs write component tables (#279); plan runs write `plans/plan.parquet` |
| Per-player detail | This-week drill-down reads stored component tables + board/PMF (#281) |
| Laptop pull | `fplt data pull --official` (#280) |
| UI | Overview + **This week** (#281). **Run timing** panel (#291 PR3) reads `ops_timing.json` when present |

## 3. Decisions

### 3.1 Sub-model tables in every predict run (issue: "write component tables")

Every predict run of a **layered** model writes three extra tables, alongside `player_fixture.parquet`, at the paths `Store.load` already expects. All carry the L2 standard keys (`fplt-data-model.md` §3.4): `run_id, as_of_season, as_of_gw, target_season, target_gw, fixture_id`, plus `player_key` / `team_key` as noted.

| Table | Grain | Columns (beyond keys) | Source |
|---|---|---|---|
| `predictions/components/minutes.parquet` | player-fixture | `xmins`, `p_played`, `p_60` | the minutes component's PMF (`xmins_from_minutes_pmf`) |
| `predictions/components/scoreline.parquet` | team-fixture (two rows per fixture) | `team_key`, `opponent_team_key`, `is_home`, `team_eg`, `opp_eg`, `p_cs` | the score component (`team_eg` = λ for, `opp_eg` = λ against, `p_cs` = P(opponent scores 0)) |
| `predictions/components/involvement.parquet` | player-fixture | `team_key`, `goal_weight`, `assist_weight`, `exp_goals`, `exp_assists` | the involvement component, and `team_eg` |

- **Reuse the existing depth code.** `_player_depth_rows` / `layered_explain_parts` already compute `team_eg`, `opp_eg`, `goal_weight`, `assist_weight` and `xmins` for one player. Generalise them to all players and all target gameweeks in the horizon; don't write a parallel implementation. `fplt explain` then reads the stored tables when present, which is a behaviour-preserving refactor with a test.
- **`exp_goals` / `exp_assists`:** if the simulation exposes per-player means, use those (preferred: consistent with `xpts`). Otherwise use `team_eg × weight`, and add a column `exp_goals_method ∈ {sim_mean, team_eg_x_weight}` so the UI can label the approximation. Never chain point estimates into `xpts` (AGENTS.md §2); these columns are for display only.
- **Opaque models** (non-layered) write no component tables. Readers treat a missing table as "no detail available".
- **Schemas first:** a pandera schema per table in `src/fplt/schemas/` (reuse the L2 standard-key mixin), with `schema_version`.
- Bonus, saves and cards (`other.parquet`) are **out of scope** here.

### 3.2 `fplt data pull --official` (issue: "data pull")

```
fplt data pull --official latest            # latest gameweek with an official pointer
fplt data pull --official --gw 14 [--season 2026-27] [--checkpoint T-90m] [--line v4_h5]
```

- Reads `fplt-data` with the owner token (`Remote.from_settings`), and writes into the local `$FPLT_DATA_DIR` with `fplt.data.remote.pull` (`--immutable`, never overwrites local files).
- Pulls, in order:
  1. `official/season=<s>/gw=<gw>/` (all checkpoints and lines; small JSON);
  2. every run referenced by those pointers: `runs/run=<predict_run_id>/` and `runs/run=<plan_run_id>/`;
  3. each predict run's curated build manifest, `curated/builds/build=<build_id>.json`, and the tables the UI needs for labels: `players`, `teams`, `fixtures`, `gameweeks`, `player_status` (`curated/tables/<table>/sha=<sha>.parquet` as listed in that manifest).
- **`latest`** = the highest gameweek in the current season with a `checkpoint=T-90m` pointer for the **primary** line. If none exists yet, list what exists (gameweeks × checkpoints) and exit non-zero. `--checkpoint` lets you pull an earlier checkpoint for display.
- Idempotent: a second run downloads nothing. Prints what it pulled and the run ids.
- Missing credentials: a clear error naming the `FPLT_R2_*` settings (this is a laptop command, so no soft skip).

### 3.3 "This week" page (issue: "This-week UI page")

A new page in `fplt ui`, reading **only pulled local files** through `fplt.store.Store`. Logic lives in `src/fplt/analysis/this_week.py` (pure polars, tested); the page file only lays out widgets, and charts are Plotly (`fplt-ui-design.md`).

1. **Header:** season, gameweek, deadline (UTC and local), line, checkpoint, predict/plan run ids, `code_sha`, `build_id`, run finish time. A selector for gameweek / line / checkpoint among the pulled pointers (default: the applied line, else primary, at `T-90m`).
2. **Plan:** transfers out → in (with prices and Δ expected points), hits, chip, XI and bench order, captain and vice, and the plan's expected points for step 0, from `plans/plan.parquet`.
3. **Players table:** one row per player with name, team, position, price, status and news, P(played) next gameweek, expected points per gameweek over the horizon (one column per target gameweek, summed over double-gameweek fixtures), and the total over the next H (slider, 1…run horizon). Filters: position, team, price range, minimum P(played), "my squad only". Sortable; a CSV download.
4. **Player drill-down** (select a row):
   - per-fixture table for each gameweek in the horizon: opponent, H/A, `xmins`, `p_played`, `p_60`, `team_eg`, `opp_eg`, `p_cs`, `p_goal`, `p_assist`, `exp_goals`, `exp_assists`;
   - the **points split** per gameweek (appearance / goals / assists / clean sheet / residual), computed at read time with `build_forecast_board` from `player_fixture` and `Rules.for_season`;
   - the **points distribution** for the next gameweek from `points_pmf_gw` when present (official runs), as a bar chart;
   - when component tables are missing, show the board and the distribution only, plus a note "no sub-model detail for this run".
5. Player names and teams come from the curated `players` / `teams` tables of the run's `build_id`, never from the live API.
6. **Run timing** (issue #291): when `official/…/ops_timing.json` is present for the selected checkpoint, show a stacked horizontal bar of top-level phases, an expander with nested sub-phases (fit components, plan/MILP), R2 transfer rows (files / bytes / throughput from `kind=remote` timing artifacts), and — if several gameweeks are pulled — a small trend of `total_s` and top phases. Logic in `fplt.analysis.run_timing`; the page only renders. Missing timing files render a caption, not an error (older checkpoints).

## 4. Sequence

| # | Who | Step | Done when |
|---|---|---|---|
| 1 | agent | **#278**: `environment: r2` on every R2 job; fail loudly on `main`; heartbeat only after a real upload; CI lint | merged |
| 2 | owner | **M1–M4** below (secrets, clock, verification, re-copy) | new captures appear in `fplt-data/raw/` within 15 min of each tick |
| 3 | agent | **§3.1 component tables** | a predict run (local) contains the three component tables, validated |
| 4 | owner | **M5**: one manual live `ops-tick` (`force_live`) | official runs and pointers are in `fplt-data` |
| 5 | agent | **§3.2 `fplt data pull --official`** | pulls the runs from step 4 to the laptop |
| 6 | agent | **§3.3 This-week page** | the page renders from pulled files with the drill-down |
| 7 | owner | **M6**: set `OPS_LIVE=true` | the next deadline's checkpoints run unattended |

Steps 3 and 5 can run in parallel once #278 has merged. Step 6 needs 3 and 5.

## 5. Owner checklist (manual)

- [ ] **M1 Secrets and variables.** Environment `r2`: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`. Repository: `FPLT_ENTRY_ID`, `FPLT_FPL_API_TOKEN`, `HC_PING_URL_*` (one per scheduled workflow), and the ntfy URL as both `NTFY_URL` and `FPLT_NTFY_URL` (the workflows use both names; follow-up: standardise). Variables: `DATA_BUCKET=fplt-data`, `LAB_BUCKET=fplt-lab`.
- [ ] **M2 Clock.** cron-job.org jobs that POST `workflow_dispatch` for `snapshot.yml` and `ops-tick.yml` every 15 min at :07/:22/:37/:52, using a fine-grained GitHub token limited to this repo with Actions: write only. Keep the GitHub schedules as backup.
- [ ] **M3 Verify ingestion.** After #278 merges: `gh workflow run snapshot.yml`, then `rclone lsf fplt:fplt-data/raw/fpl_api/season=2026-27/ | tail` shows the new capture. Also trigger `understat.yml` once, as its fix hasn't run live yet.
- [ ] **M4 Re-copy the branch tip** into R2 for the captures from 24–27 Nov (`r2-data-bucket-plan.md` backfill step 8). Then add the `raw/` lock if it isn't on yet.
- [ ] **M5 First live run.** `gh workflow run ops-tick.yml -f force_live=true`, then check `rclone lsf fplt:fplt-data/official/ -R | head` and `rclone lsf fplt:fplt-data/runs/ | tail`.
- [ ] **M6 Go live.** Repository variable `OPS_LIVE=true`. Add the locks on `runs/` and `official/` once they hold real data.

## 6. Known risks (not blocking)

- `ops-tick` keeps working state between runs in the GitHub Actions cache, which can be evicted. Catch-up then rebuilds from raw (safe once raw is in R2), just slower. Moving that state to R2 is a follow-up.
- Official runs are defined by their checkpoint, but manifests don't yet record a data cutoff separate from the start time (`cutoff_at`). Follow-up issue.
