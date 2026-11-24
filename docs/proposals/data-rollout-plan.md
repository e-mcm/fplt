# Data rollout plan: bucket first, then the data model

**Goal of this plan:** get to a working R2 bucket with the agreed storage contract, the GitHub Actions ingestion jobs writing raw captures into it, and the data-model foundations in place for everything that will be written later. Experiments and the UI come afterwards (Stage C).

**Sources:** `r2-data-bucket-plan.md` (bucket ops, raw-ingest PRs), `fplt-data-model.md` (storage contract, migration §8), `ops-hardening-issues.md` (ops-03, ops-06).

Legend: **[you]** = a manual action by you · **[PR]** = code, one PR each, ≤ ~200 lines of non-test code, `make check` green, merged before the next starts (CLAUDE.md).

---

## Stage A: raw ingestion into R2 (the milestone)

The raw layer is unchanged by the data model: same paths, write-once, `captured_at`. So the ingestion jobs can move to the bucket **now**, before any of the run or curated machinery exists.

| # | Who | Step | Detail | Done when |
|---|---|---|---|---|
| A0 | [you] | **Set up R2** | Bucket plan §2.1–2.5: enable R2 (card needed); create `fplt-data` and `fplt-lab` (no public access, **no lifecycle rules**); add the **`raw/` lock only** (other locks come later); create the 3 tokens; add GitHub secrets `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` and the variable `DATA_BUCKET`; put the owner token in `~/.config/fplt/env`; install rclone + wrangler on the Mac | `rclone lsd fplt:fplt-data` works |
| A1 | [PR] | **Design: storage contract** | `docs/design.md`: decision 2 (R2 replaces the snapshots branch), §4.3 layout (three buckets, prefixes, every object written once, keep forever), §7.2 ledger-as-view (approved 2026-11-24). AGENTS.md: agents read `fplt-data`, write `fplt-lab`. Docs only. Combines bucket plan PR 0 and data model step 0 | merged |
| A2 | [PR] | **`fplt.data.remote`** | Settings fields (`r2_*`, `data_bucket`, `lab_bucket`), rclone wrapper with the per-bucket `PUSHABLE` whitelist, tests against real rclone on local dirs; `rclone` installed in `ci.yml` (bucket plan PR 1) | merged |
| A3 | [PR] | **Dual-write the capture workflows** | `.github/actions/push-raw` composite action; add it to `snapshot.yml`, `understat.yml`, `api_football.yml`, `football_data.yml` **before** the branch publish, with both steps `if: always()` (bucket plan PR 3). Check each workflow's real output dir | merged |
| A4 | [you] | **Backfill + verify** | Bucket plan §2.6: copy the Mac's `raw/` and the branch tip with `rclone copy --immutable`; `rclone check` shows 0 differences; try overwriting one object (must be refused); watch the first few scheduled runs land in R2 | **Milestone: every GHA ingestion job writes to R2** |
| A5 | [PR] | **Read raw from the bucket** | `fetch_raw` (bucket when configured, branch otherwise); ops-catchup and sensors call it; `--source auto/bucket/branch` (bucket plan PR 2) | merged |
| A6 | [PR] | **Parity check** | `fplt data parity` (branch tip vs bucket) plus a weekly GHA job during dual-write (bucket plan PR 5) | first weekly run green |
| A7 | [PR] | **Heartbeats and failure alerts for capture jobs** | The capture-workflow part of ops-03: a healthchecks ping on success, an ntfy alert `if: failure()`, plus a keepalive or runbook note for the 60-day schedule disable. **Must land before A9** | a forced failure gives one alert |
| A8 | [you] | **Approve cut-over** | ~2 weeks of clean parity; heartbeat configured in healthchecks.io with the ntfy integration | go/no-go |
| A9 | [PR] | **Cut-over** | Remove the branch-publish steps; `fetch_raw` defaults to the bucket; the `snapshots` branch is kept as a read-only archive, never deleted (bucket plan PR 6) | merged; the first post-cut-over captures are only in R2 |
| A10 | [PR] ×3 | **Data sanity gates** (optional here, otherwise Stage B) | ops-06 gates run in the capture workflows right after capture; quarantine records; staging reads a skip list until curated builds (B6) carry it in the build manifest | gates green on recorded fixtures |

**After A4 the goal is met** (ingestion writes to R2). A5–A9 make the bucket the single source of truth.

---

## Stage B: data-model foundations (data side only)

Order: fix the live bug, then schemas, then the writers that new data will flow through. Each lock is added by you **when its prefix first receives real data**.

| # | Who | Step | Detail | Depends on |
|---|---|---|---|---|
| B1 | [PR] | **`model_id` collision fix** | A regression test (two code shas, same spec, must not reuse predictions), then an interim fix: `write_predictions_or_existing` stores the code sha in a sidecar and refuses reuse on mismatch. The proper fix arrives with runs (B4) | none; can run in parallel with Stage A |
| B2 | [PR] | **Schemas** | pydantic / pandera: `RunManifest`, `BuildManifest`, `OfficialPointer`, `AppliedRecord`, `InterventionRecord`, the outcomes tables, `squad_state`, the L2 standard-key mixin, `schema_version` on every table (data model step 2). Experiment and batch schemas wait for Stage C | A1 |
| B3 | [PR] ×n | **Typed component params** | One pydantic params model per component kind, `extra="forbid"`; a test that the resolved defaults equal today's `.get` defaults (data model step 2b). One PR per component kind | none |
| B4 | [PR] ×2 | **Run writer** | `RunWriter` (sortable `run_id`, manifest first, `_SUCCESS` last, refuses an existing run); `predict` / `plan` / backtest write through it locally, old paths still written in parallel. Second PR: push finished runs (official → `fplt-data`, everything else → `fplt-lab`). Until curated builds exist, manifests record `raw_manifest_hash` plus the staged-table SHA-256 digests the eval code already computes | B2; `spec_hash` uses the existing `resolved_spec_sha256` until B8 |
| B5 | [you] | **Locks for runs** | `runs/` on both buckets, `specs/` on both (bucket plan §2.2) | B4 merged and its first runs pushed |
| B6 | [PR] ×2 | **Curated builds** | Content-addressed tables + build manifests + the `staging_version` golden-hash test; then `PITView(build_id, deadline)`, runs record `build_id`, builds pushed to `fplt-data/curated/`. If staging moves into GHA with ingestion, this is where it plugs in | B2, A5 |
| B7 | [you] | **Lock curated** | `curated/` lock | B6 pushing |
| B8 | [PR] ✓ | **Spec identity** | `spec_hash` over the resolved spec (defaults included, name excluded), `registry.lock.yaml` with a CI hash-pin check, `specs/sha=` store (data model step 2c, minus grids) | B3 complete |
| B9 | [PR] ×2 | **Outcomes + ledger view** | Stage `player_fixture_outcomes` / `fixture_outcomes`; `Store.ledger()` view; switch the metric readers (`compare.py`, `run.py`, decision pool); then remove the `fill_actuals` / `append_predictions` mutation. Equivalence test: the view returns exactly the old ledger's rows | B4, B6 |
| B10 | [you] → [PR] | **Legacy import** | [you] back up `$FPLT_DATA_DIR` (external disk), then [PR] `fplt legacy import` wraps existing `predictions/`, `plans/`, `ledger/`, `artifacts/`, eval outputs as `kind=legacy` runs; [you] run it from the Mac into `fplt-lab`, check the counts | B4, B9 |
| B11 | [PR] ✓ | **`PipelineSpec`** | data / model / planner / policy / env sections, hashed; production expressed as PipelineSpecs; a test that today's ops behaviour is reproduced exactly (experiments doc §8 step 1) | B8 |
| B12 | [PR] ×3 | **Official lines** | `lines:` in `production.yaml` (`v4_h3` primary; `v4_h5` / `v4_h7` / `v4_h9`); predict once per spec, plan per line; T−90m pointers plus `applied` / `submitted` / intervention records; virtual squad chains; `points_pmf` + `points_pmf_gw` for official runs (data model steps 6, 6b) | B4, B11, and ops-02 (checkpoints) + ops-04 (`prod` tag) on the ops track |
| B13 | [you] | **Lock official** | `official/` lock; choose the gameweek at which the virtual lines start from your real squad | B12 live |

**End of Stage B:** ingestion → raw → curated builds → official and ad hoc runs → official pointers, all write-once in R2, with the ledger as a view and every run reproducible from its manifest.

---

## Stage C: later

- **Experiments:** experiment and batch manifests, events, arm reuse, `results/` tables, `registry.yaml` `experiment_id`; the generic patch engine and grids; stage-tagged measures; the decision engine; the standing robustness run; YAML retirement (data model step 7, experiments doc §9). Then [you] add the `experiments/` and `batches/` locks on `fplt-lab`.
- **Catalog + read API:** `catalog/`, `fplt.store`, `fplt sql` (data model step 8).
- **UI:** Streamlit dashboard on the Mac (UI doc §7).
- **Website:** [you] create `fplt-public` and its custom domain; publish step from official pointers.
- **Cut-over of old paths:** stop writing `predictions/` / `plans/` / `ledger/` locally (data model step 10).

## Parallel ops track (not data, but gates B12)

ops-01 (token monitor), ops-02 (deadline checkpoints), ops-03 (the rest of heartbeats), ops-04 (`prod` tag), ops-05 (apply policy). They can run alongside Stage A/B; B12 needs ops-02 and ops-04.

## Your manual actions, in order

1. **A0:** R2 account, two buckets, the `raw/` lock, three tokens, GitHub secrets, local env, rclone + wrangler.
2. **A4:** raw backfill, `rclone check`, overwrite test, watch the first runs.
3. **A8:** healthchecks.io checks + ntfy; approve cut-over after ~2 clean weeks.
4. **B5:** locks on `runs/` and `specs/`.
5. **B7:** lock on `curated/`.
6. **B10:** back up the Mac data dir; run the legacy import; check counts.
7. **B13:** lock on `official/`; pick the virtual-line start gameweek.
