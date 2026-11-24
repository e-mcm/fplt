# fplt data model: layers, runs, cadences, keep-forever retrieval

**Status:** proposal · **Repo:** `e-mcm/fplt` · **Date:** 2026-11-24
**Supersedes:** §1 / §1.1 of `r2-data-bucket-plan.md` (bucket layout), and the 90-day scratch expiry there.

## 0. Requirements

| Dimension | Values |
|---|---|
| Layer | raw ingest · curated (post-processed) · analytic outputs (sub-model predictions, main-model predictions, actions over horizons) |
| Cadence | **official** weekly (the best the system can produce) · **ad hoc** experimentation · **A/B** tests |
| Retention | everything kept forever |
| Retrieval | easy: "official predictions for GW12", "every run of spec X", "both arms of experiment Y", "what did we know at the GW12 deadline" |

## 1. What's wrong today

Found by reading the code at `f6dda83`:

1. **`model_id` doesn't capture code changes (bug).** `ModelSpec.model_id` hashes the spec plus `version("fplt")`, and the package version is a static `0.1.0`. Two runs of the same YAML on different code get the same `model_id`, so they write to the same `predictions/<model_id>/as_of=…` path. `write_predictions_or_existing` (orchestration) then **returns the existing file**, so a refit after a code change can silently reuse predictions made by older code. `write_plan` raises `FileExistsError` instead, which blocks the rerun outright.
2. **The ledger mutates.** `append_predictions` reads the whole `part.parquet`, concatenates and rewrites it. `fill_actuals` rewrites prediction rows in place to add `actual_*`. `ledger/decisions` replaces rows when a model and gameweek are rerun. History is overwritten, and two writers lose rows.
3. **Curated data is overwritten.** `staged/` is rebuilt in place. Once staging code changes, there's no way to recover the exact inputs an old prediction used.
4. **There's no notion of a run.** Outputs are keyed by `(model_id, as_of)` only. Nothing records which invocation, code sha, machine, inputs, purpose (official, ad hoc, A/B) or seed produced them.
5. **Official isn't distinguishable.** A live production plan and an ad hoc local plan for the same spec and gameweek land in the same place.
6. **Model identity is muddled** (see §3.7):
   - The hash includes the human `name`, so two identical specs under different names get different ids.
   - The hash excludes code defaults: there are 94 `spec.params.get("…", default)` calls in `src/fplt/models`, and changing one changes behaviour without changing any hash.
   - Every parameter variant needs a hand-written YAML; there are already ~50 in `configs/models/`.
7. **Experiment outputs are keyed by eval name** (`_default_output_dir(data_dir, spec.name)`), so a rerun of an EvalSpec writes over the previous one. The committed `configs/evals/results/registry.yaml` (verdict, window, data digests) is good, and this design keeps it.

Raw is already done well (write-once, `captured_at`, lossless JSON) and stays as it is.

## 2. The model in one paragraph

**Everything that produces analytic output is a run.** A run has an immutable id, a manifest saying exactly what went in (code sha, spec, curated build, seeds, deadline) and why (official, ad hoc, experiment arm), and a folder of output tables. Runs are never modified. **"Official" isn't a storage location but a pointer:** a small write-once record saying "for GW12 at checkpoint T−90m, the official run is `r_…`". **An A/B test is an experiment record** that names its arm runs and its verdict. **Actuals live in their own curated table** and are joined to predictions at read time; they are never written into prediction rows. **Curated data is built in content-addressed snapshots**, so any run can name the exact inputs it saw. Every object is written once, so the whole bucket can sit under an indefinite retention lock.

## 3. Layers

```
L0 raw/         external payloads as received          write-once, keyed by source + captured_at
L1 curated/     typed tables derived from raw          content-addressed builds, never overwritten
                + outcomes (actuals)                   (staged/ and features/ move here)
L2 runs/        analytic outputs of one invocation     write-once per run_id
                 ├ component predictions (scoreline, minutes, involvement, other)
                 ├ player-fixture predictions (main model)
                 ├ plans / actions over the horizon
                 └ metrics, logs
L3 index        official pointers, experiments, catalog   pointers and experiments write-once; catalog derived
```

This is the standard lakehouse "bronze / silver / gold" split, plus the run concept from experiment trackers (an MLflow run has the same shape as the manifest below). Section 7 covers why no table format (Iceberg, Delta) is needed yet.

### 3.1 L0 raw: unchanged
`raw/<source>/season=…/…/captured_at=<UTC>/<payload>`, gzip JSON or CSV exactly as published. New sources (fplcache, odds capture) follow the same convention.

### 3.2 L1 curated: content-addressed builds

```
curated/
  tables/<table>/sha=<sha256 of file>.parquet        # immutable, deduplicated
  builds/build=<build_id>.json                        # manifest: table -> sha, plus provenance
  features/<name>/v=<ver>/build=<build_id>/season=…/gw=…/part.parquet
```

- **`build_id`** = `hash(raw manifest, staging_version, schema versions)`. The raw manifest is the sorted list of `(path, size)` for every raw file included. Raw is write-once, so path plus size identifies it.
- **`staging_version`** is a constant bumped whenever any staging transform changes output, with the same discipline as feature versions (AGENTS.md §2).
- **Dedup:** a table whose content didn't change between builds points at the same `sha=` file. Keeping every build forever therefore costs about one copy of each *distinct* table version, which is megabytes per gameweek.
- A **build manifest** records `build_id, created_at, staging_version, code_sha, raw_manifest_hash, raw_cutoff (max captured_at), tables{name: sha, rows, schema_hash}, quarantined_captures[]` (from ops-06).
- **Outcomes** (`player_fixture_outcomes`, `fixture_outcomes`) are ordinary curated tables built from `event_live` and fixtures once `data_checked` is true. They carry `available_at`, and corrections create a new build rather than a rewrite.
- `PITView` is constructed **from a build**: `PITView(build_id, deadline)`. Model and feature code still only see a `PITView` (AGENTS.md §2). Locally the build is materialised under `$FPLT_DATA_DIR/curated/` as today, just addressed by id.

This makes the system **bitemporal** for free. `available_at` says when a fact became knowable in the world. `build_id` says what our pipeline believed at the time. "Reproduce the GW12 official plan exactly" means: the same build, the same code sha, the same spec, the same seed.

### 3.3 L2 runs

```
runs/run=<run_id>/
  manifest.json
  predictions/player_fixture.parquet      # main model, player-fixture grain
  predictions/components/scoreline.parquet
  predictions/components/minutes.parquet  # PMF buckets
  predictions/components/involvement.parquet
  predictions/components/other.parquet
  predictions/points_pmf.parquet          # official runs only: P(points = k), k = -5..30+, player-fixture grain
  predictions/points_pmf_gw.parquet       # official runs only: same, player-gameweek grain, from the sims (correct for doubles)
  plans/plan.parquet                      # actions per horizon step
  plans/decisions.parquet                 # XI, captain, transfers, chips for step 0
  metrics/*.parquet                       # for backtests: per-row and summary scores
  logs/timing.json
  _SUCCESS                                # written last; a run without it is incomplete
```

**`run_id`**: time-sortable and globally unique across machines, e.g. `r_20261124T125601Z_7f3a2c` (UTC timestamp + 6 random hex), or a ULID. Sortable ids make "latest" a listing, not a lookup.

**`manifest.json`** (pydantic `RunManifest`, a schema in `src/fplt/schemas/`):

```json
{
  "run_id": "r_20261124T125601Z_7f3a2c",
  "kind": "official | adhoc | experiment | benchmark",
  "mode": "live | backtest",
  "purpose": "free text, e.g. 'GW12 T-90m' or 'try xg_blend 0.5'",
  "experiment_id": null,
  "arm": null,
  "run_type": "predict | plan | eval",
  "spec_hash": "sha256 of the fully resolved spec, defaults included, name excluded (§3.7)",
  "spec_label": "dc_layered_v4_native, or base[path=value,…] for unnamed variants",
  "model_id": "existing id, kept for compatibility",
  "config_hashes": {"scoring": "…", "id_maps": "…", "ops": "…", "eval_spec": "…"},
  "line": "v4_h5 | null (official lines only, §4.1)",
  "plan_horizon": 5,
  "squad_mode": "real | virtual | null",
  "prev_run_id": "previous week's run of the same virtual line, or null",
  "code_sha": "git sha of the checkout",
  "code_dirty": false,
  "prod_tag": "prod-20261120-ab12cd3 or null",
  "fplt_mc_version": "0.3.1",
  "lockfile_hash": "sha256 of uv.lock",
  "build_id": "curated build the PITView was made from",
  "as_of": [{"season": "2026-27", "gw": 12, "deadline_utc": "…"}],
  "horizon": 6,
  "checkpoint": "T-90m | null",
  "seeds": {"base": 0},
  "host": "gha | mac | vm-<name> | modal | agent-<id>",
  "started_at": "…", "finished_at": "…",
  "status": "succeeded | failed",
  "parent_run_id": null,
  "tags": {}
}
```

The manifest identity is **(spec_hash, code_sha, lockfile_hash, build_id, seeds)**. That fixes problem 1 without changing `model_id` semantics. `model_id` stays as a label, and nothing keys storage on it any more.

**Predict runs and plan runs are separate.** A predict run simulates one spec out to the longest horizon any consumer needs. A plan run reads its parent predict run (`parent_run_id`) and solves the optimiser for one `plan_horizon` and solver setting. Several plan horizons therefore share one simulation, and their differences come purely from planning.

**A live run covers one deadline. A backtest run covers a window:** `as_of` lists every deadline it predicted, and its tables carry `as_of_*` columns.

**Sub-models vs main model.** Component outputs are stored per run under `predictions/components/`, with the same keys as the main table. "What did the minutes model say for Salah at the GW12 deadline?" is a filter on one table, not a re-fit.

**Monte Carlo draws** aren't stored: 10k sims × ~700 players × fixtures is too big to keep forever per run. Store summaries (mean, sd, quantiles, event probabilities) plus, for **official runs only**, the points distributions: 36 bins from −5 to 30+ (FPL points can be negative: red −3, own goal −2, missed penalty −2). They are stored at player-fixture grain and at player-gameweek grain, taken directly from the simulations, because a double gameweek's total isn't the convolution of its fixtures. Stored in long format without zero-probability bins. Backtest and ad hoc runs keep summaries only (mean, sd, quantiles, event probabilities); a run that needs distributions can regenerate them from its seeds. A run's draws can be regenerated exactly from the manifest because the seeds are recorded.

Plan runs in `virtual` mode also write `plans/squad_state_in.parquet` and `plans/squad_state_out.parquet` (squad, bank, free transfers, chips used) so each line's season is an auditable chain (§4.1).

### 3.4 Standard keys (every L2 table)

| Column | Notes |
|---|---|
| `run_id` | always present, even though it is in the path; files stay self-describing when copied |
| `as_of_season`, `as_of_gw`, `deadline_utc` | the deadline the prediction was made for |
| `target_season`, `target_gw`, `horizon` | `horizon = target_gw − as_of_gw` for the same season |
| `fixture_id` | player-fixture grain; a double gameweek gives two rows, a blank gameweek none (AGENTS.md §6) |
| `player_key`, `team_key` | cross-season keys, never FPL element ids |
| `step` | plans only: horizon step 0…H−1 |

Each table has its own pandera schema, added before its producer (AGENTS.md §2).

### 3.5 L3 index

```
specs/sha=<spec_hash>.json                                   # resolved spec, content-addressed, forever (§3.7)
official/season=<s>/gw=<gw>/checkpoint=<cp>/line=<line>.json  # {"predict_run_id", "plan_run_id", "role", "squad_mode", "written_at"}
official/season=<s>/gw=<gw>/checkpoint=<cp>/applied.json      # {"line": …, "plan_run_id": …, "submitted_at": …}
experiments/exp=<experiment_id>/manifest.json   # EvalSpec name + hash, arms → run_ids, windows, CRN seeds, data build ids
experiments/exp=<experiment_id>/results/…       # paired comparison, DM/bootstrap outputs, verdict computed by code
catalog/runs.parquet                            # derived: one row per run manifest; rebuildable at any time
catalog/experiments.parquet                     # derived
```

- **Official pointers** are written once per line per checkpoint, by the ops job only (owner or GHA token), and only after the run's `_SUCCESS`. The **official answer for a gameweek is the fixed T−90m checkpoint** of each line, whether or not that is what was applied. The applied plan is recorded separately (`applied.json`, `submitted.json`, interventions), so the record keeps both the system's official output and what actually happened. If the T−90m run fails, the gameweek's official pointer for that line is missing, never back-filled from another checkpoint. The weekly report flags it and falls back to the latest earlier checkpoint for display only, marked as such. The website, weekly report and live percentile read official runs only.
- **Experiments**: `fplt promote-eval` mints an `experiment_id`, runs each arm as a run (`kind=experiment`, `arm=control|treatment|…`) and writes the experiment manifest and results. The human-written keep/drop decision stays in the committed `configs/evals/results/registry.yaml`, which gains an `experiment_id` field. The registry is the decision; the experiment folder is the evidence.
- **Catalog** files are the only objects that are rewritten. They are pure derivations of the manifests (`fplt catalog rebuild` lists `runs/*/manifest.json`), so losing one costs nothing. For many concurrent writers, each run also writes a one-row `catalog/inbox/run=<id>.parquet`, and readers union inbox with the compacted file.

### 3.7 Model identity: hashes for machines, names for humans

A name isn't a sound identifier: it can be edited in place, it can't express a variant without a new file, and today it changes the hash even when the model doesn't. Identity is split into three parts, each with one job:

| Part | Answers | Derived from |
|---|---|---|
| `spec_hash` | **which model**: components, parameters, features, windows | sha256 of the *fully resolved* spec: `extends` applied, `overrides` merged, **every default filled in**, `name` and comments excluded, canonical JSON |
| implementation | **which code ran it** | `code_sha`, `lockfile_hash`, `fplt_mc_version` in the run manifest |
| `spec_label` | **what to call it** | a registered name, or an auto-label for an unnamed variant |

**Typed parameters, so defaults are part of the hash.** Replace the 94 `spec.params.get("x", default)` calls with one pydantic params model per component kind (`class InvolvementXgParams(BaseModel): window: int = 10; …; model_config = ConfigDict(extra="forbid")`). The resolved spec is then built by validating each component's params, so defaults are explicit and hashed. A typo'd parameter name fails loudly instead of silently using the default. Changing a default now changes `spec_hash` for every spec that relied on it, which is correct: a different model gets a different id. A test pins the `spec_hash` of every registered name to catch this, and a deliberate default change updates the pins in the same PR. This also enforces AGENTS.md's "no hyperparameters hardcoded in source" mechanically.

**Names are an immutable registry, not files per variant.** `configs/models/` keeps YAML only for models you refer to by name: production, released candidates, and frozen parents that forks extend. A committed `configs/models/registry.lock.yaml` maps `name → spec_hash`. CI fails if a released name's hash changes. That is the mechanical form of "never edit a released spec in place": to change the model you must pick a new name. The `_v1, _v2…` suffix convention then only marks released lineage, not every experiment.

**Variants are generated, not written out.** An experiment or ad hoc run states a base spec plus overrides, as a grid or a list, using dotted paths:

```yaml
# configs/evals/xg_blend_sweep.yaml
base: dc_layered_v4
grid:
  components.involvement.params.xg_blend: [0.3, 0.5, 0.7, 0.9]
  components.involvement.params.shrinkage: [3, 5, 8]
```

That's 12 arms from one file. Each arm's resolved spec is written to `specs/sha=<hash>.json` and gets the auto-label `dc_layered_v4[involvement.xg_blend=0.5,involvement.shrinkage=8]`. The label is derived from the diff against the base, so it is always truthful. Ad hoc runs do the same on the command line: `fplt predict --spec dc_layered_v4 --set components.involvement.params.xg_blend=0.5`. Optuna sweeps use the same mechanism: each trial is an override set and therefore a spec hash.

**Promotion names a winner.** When a variant wins, `fplt spec freeze <spec_hash> --name dc_layered_v5` writes the YAML (as `extends: dc_layered_v4` + `overrides:` from the diff) and the registry entry. Production lines and the website refer only to registered names.

**Two specs with the same hash are the same model.** If two names resolve to one hash, the registry flags it, catalog queries group by hash, and results stay comparable across renames.

### 3.8 Experiments

The general definition (stages, patches on a `PipelineSpec`, stage-tagged measures, decision types, the 21-kind catalogue, component experiments) is in **`fplt-experiments.md`**. This section covers storage only.

An experiment is **any multi-arm question with pinned inputs**. There are two flavours, sharing one model:

| Flavour | Example | Outcomes | Verdict |
|---|---|---|---|
| `comparison` | "H=10 predicted points, spec A vs B, as of now" | none (future) | none: you look at the differences |
| `ab` | "does `xg_blend=0.5` beat production on 2023-24 to 2025-26?" | backtest windows (or accumulated live lines) | rule-based from the EvalSpec, then a human decision |

Pinning is what makes arms comparable: every arm uses the **same curated `build_id`**, the same deadlines and the same CRN seeds. They differ only in their spec, or their environment for equivalence levers.

```
experiments/exp=<experiment_id>/
  manifest.json          # written at submit; immutable
  events/<ts>_<event>.json   # submitted, arm_started, arm_succeeded, arm_failed, analysed, aborted
  results/arm_metrics.parquet     # arm × window × metric → value, se, n
  results/paired.parquet          # treatment − control per metric × window → diff, ci_lo, ci_hi, p, method
  results/per_unit.parquet        # per gameweek (and per player where relevant) diffs for drill-down
  results/verdict.json            # each gate pass/fail with its threshold, overall computed verdict
  results/report.md
batches/batch=<batch_id>/manifest.json   # experiments submitted together
batches/batch=<batch_id>/events/…
```

**`ExperimentManifest`:**
- `experiment_id`, `flavour`, `question` (one line, required), `batch_id`
- `eval_spec` name and hash (for `ab`)
- `arms`: a list of {`arm`, `role` (control | treatment | baseline), `spec_hash`, `spec_label`, `env`}
- `build_id`, `as_of` deadlines or windows (primary / confirm), `horizon`, `seeds`
- `metrics` and `gates` declared up front, so the verdict rule is fixed before results exist
- `code_sha`, `submitted_by`, `host`

**Status is derived from events.** Progress appends small event files rather than updating a status field, so everything stays write-once. `Store.experiment(id).status` folds the events: planned → running (k/n arms) → analysed | failed | aborted.

**Arm reuse.** An arm run is fully identified by `(spec_hash, code_sha, lockfile_hash, build_id, deadlines, horizon, seeds)`. Before running an arm, look that key up in the catalog. If a succeeded run exists, the experiment just references it. Ten overnight experiments against the same production control then run the control **once**, not ten times. This is the ledger cache from the experiment-framework design, now falling out of the data model.

**Verdict vs decision.** `results/verdict.json` is what the code computes from the declared gates. Keep, drop or tune stays a human decision in the committed `configs/evals/results/registry.yaml`, which references the `experiment_id`. An `ab` experiment can never flip production by itself.

**Batches.** `fplt exp submit configs/evals/a.yaml configs/evals/b.yaml … --batch nightly` mints a `batch_id`, writes each experiment manifest, and schedules arms on a backend (local, VM or Modal), deduplicating shared arms across the whole batch. When the last experiment is analysed, the batch writes `report.md` and sends one ntfy notification with the summary table.

**Comparisons are experiments too.** `fplt compare --spec dc_layered_v4 --spec dc_layered_v5 --horizon 10` pins the latest build and current deadline, creates a `comparison` experiment with two predict arms, and opens the UI on it. Same machinery, no verdict.

### 3.6 Actuals and the "ledger"

The ledger stops being a stored table and becomes a **view**:

```
ledger = runs/*/predictions/player_fixture  ⋈  curated outcomes (latest build)
         on (target_season, fixture_id, player_key)
```

- `append_predictions`, `fill_actuals` and the decisions "replace rows" go away. A backtest run *is* its ledger contribution.
- Scoring uses the latest outcomes build by default. Pass `outcomes_build=` to pin one, e.g. to reproduce an audit.
- This also fixes the concurrency problem from the bucket plan: there's no shared appended file.

## 4. Cadences mapped onto the model

| | Official weekly | Ad hoc | A/B |
|---|---|---|---|
| Who runs it | ops job from the `prod` tag (ops-04) | you or an agent, any machine | `fplt promote-eval` |
| `kind` / `mode` | `official` / `live` | `adhoc` / `live` or `backtest` | `experiment` / `backtest` (arms), optionally `live` shadow arms |
| Inputs | latest curated build at the checkpoint | any build (default latest) | **one pinned build for all arms**, CRN seeds |
| Extra records | `official/…/checkpoint=….json` | none | `experiments/exp=…/` + registry entry |
| Bucket | `fplt-data` | `fplt-lab` | `fplt-lab` |
| Retention | forever, locked | forever, locked | forever, locked |

### 4.1 Official lines: several models and horizons per gameweek

A gameweek's official output is a **slate of lines**, defined in `configs/ops/production.yaml`:

```yaml
lines:
  - {line: v4_h3, spec: dc_layered_v4_native, horizon: 3, role: primary}
  - {line: v4_h5, spec: dc_layered_v4_native, horizon: 5, role: candidate, squad_mode: virtual}
  - {line: v4_h7, spec: dc_layered_v4_native, horizon: 7, role: candidate, squad_mode: virtual}
  - {line: v4_h9, spec: dc_layered_v4_native, horizon: 9, role: candidate, squad_mode: virtual}
```

- **Exactly one `primary` line** is applied (ops-05 rules). Candidates are recorded, never acted on. This keeps to the existing "never silently multi-arm live" rule: the arms are explicit and only one acts.
- **One predict run per distinct spec** per checkpoint, at the longest horizon its lines need. **One plan run per line.** The four v4 lines above (the initial slate) cost one simulation, out to horizon 9, and four solves.
- **Squad modes.** Only the applied line's transfers actually happen:
  - `real`: plan from the real squad each week. Fair for "what would it do differently this week", and for advisory views.
  - `virtual`: the line carries its own squad, bank, free transfers and chips forward from its own previous decisions, like a separate FPL team. This is the only fair season-long comparison between, say, horizon 3 and horizon 9, because long horizons pay off over several weeks. The chain is linked by `prev_run_id`, and the starting state is recorded when the line is created. It reuses the ownership-advancing logic already in `optimise/decisions.py` and `ensemble_sensitivity.py`.
  - The primary line is always `real`. Each candidate gets `virtual` by default and can also run `real` if you want both views.
- **Changing a line** (new spec or horizon) means a new line name. A line's history is one spec at one horizon, so its season record means something.
- **Retrieval:** `s.official(gw=12)` gives the applied line; `s.official(gw=12, line="v4_h9")` any other; `s.line_season("v4_h9")` returns the virtual-team season with realised points per week.
- This replaces the separate "shadow runs" idea: shadows are candidate lines. They accumulate genuinely live, out-of-sample results that no backtest can fake. At season end, "which model and horizon should have been primary?" is a query, and the weekly report (ops-08) shows the lines side by side.

### 4.2 Other things the record needs

- **What was actually submitted.** The plan says what we intended. After applying, read the squad back from the FPL API and store `official/…/submitted.json` plus a `submitted_squad` table, so the record holds both "planned" and "what FPL has". This is the ops-05 verification, stored.
- **Manual interventions.** If you veto, edit transfers by hand, or override a chip, write an `intervention` record (who, when, what changed, reason) next to `applied.json`. Otherwise the official record claims the model made decisions it didn't.
- **Every config that changes results is hashed into the manifest**, not just the spec: scoring rules, id maps, ops config, EvalSpec (`config_hashes`).
- **Failed runs are kept**, with `status=failed` and their logs. They are invisible to `Store` reads by default but queryable, so a failure pattern can be investigated later.
- **Schema evolution.** Data is kept forever, so every L1/L2 table carries `schema_version`. Readers upgrade old versions on read through registered upgrade functions, each tested against a stored sample of the old version. Never rewrite old files.
- **Benchmarks** (FPL `ep_next`, OpenFPL, the projections in `benchmark_projections/`) are runs with `kind=benchmark`, keyed the same way, so they sit in the same comparisons.
- **UTC everywhere.** `deadline_utc`, `available_at`, `captured_at`, `run_id` timestamps.

**Live official predictions are the only irreplaceable analytic output.** They are what was actually known and decided at the deadline. Backtests can be regenerated from the manifest; official live runs can't. That's why they get the strictest bucket and the lock.

## 5. Buckets and retention (revises the bucket plan)

| Bucket | Prefixes | Writers | Lock |
|---|---|---|---|
| `fplt-data` | `raw/`, `curated/`, `runs/` (official only), `official/`, `catalog/` | GHA capture (raw), ops job and owner token | indefinite on everything except `catalog/` and `status/` |
| `fplt-lab` | `runs/` (adhoc, experiment, benchmark), `experiments/`, `catalog/` | owner, agents, Modal | indefinite on `runs/` and `experiments/` |
| `fplt-public` | whitelisted JSON for the website | publish step only | none (regenerable from official runs) |

- **The 90-day scratch expiry is dropped.** Ad hoc and A/B runs are kept forever as you asked. The lab bucket is separate only so agent tokens can never touch official data.
- A run's own scratch (timing traces, solver logs) lives inside the run folder and is kept with it.
- **Size check:** a live run is about 1–3 MB of parquet (components + main + plan, zstd). A full multi-season backtest run is roughly 50–300 MB. A heavy year of 200 experiments × 3 arms × 150 MB comes to about 90 GB, which costs ~$1.20 a month on R2 Standard at $0.015/GB-month after the 10 GB free. "Forever" is affordable. Revisit only if sweeps reach thousands of arms; then trim summaries on backtest arms first.

## 6. Retrieval

One read API, `fplt.store`, used by analysis, eval, reports and publish. Model and feature code never use it; they only see a `PITView`.

```python
from fplt.store import Store
s = Store.from_settings()                       # local mirror, pulls missing objects from R2 on demand

s.official(season="2026-27", gw=12)              # -> RunManifest of the applied official run
s.load("player_fixture", run=s.official(gw=12))  # -> pl.LazyFrame
s.load("minutes", run="r_2026…")                 # sub-model output
s.runs(spec_name="dc_layered_v5", kind="adhoc")  # catalog query -> pl.DataFrame
s.experiment("exp_20261124_xgblend").arms        # {"control": run_id, "treatment": run_id}
s.ledger(runs=[...], outcomes_build=None)        # predictions ⋈ outcomes view
s.pit(build_id=..., deadline=...)                # PITView for reproducing a run
```

- Local first: `Store` reads `$FPLT_DATA_DIR`, and on a miss pulls the specific objects with `rclone copy --immutable`. Immutable objects mean a local copy is never stale.
- **SQL for ad hoc questions:** DuckDB over the same parquet, locally or straight from R2 via its S3 support. `fplt sql` opens a DuckDB shell with views pre-registered (`runs`, `official`, `player_fixture`, `minutes`, `plans`, `outcomes`, `ledger`).
- CLI: `fplt runs ls --kind official --season 2026-27`, `fplt runs show <run_id>`, `fplt runs diff <a> <b>` (manifest and output deltas), `fplt official show --gw 12`.

## 7. Why not Iceberg / Delta now

Table formats give atomic appends, time travel and schema evolution on top of parquet. Cloudflare offers a managed Iceberg catalog on R2 (R2 Data Catalog). The design here gets the same guarantees from immutability: no appends, versions are builds and runs, and time travel means "pick an older id". It stays plain parquet + JSON, readable by polars, DuckDB or anything else, with no catalog service to depend on. The upgrade path stays open: if the catalog outgrows parquet files (tens of thousands of runs) or you want multi-writer tables, register the run tables in Iceberg without moving files.

## 8. Migration (small PRs, each ≤ ~200 lines of non-test code)

0. **Design PR (docs only).** Rewrite design §4.3 and §7.2 with this model. The PR notes that it changes the ledger format and the storage contract. The owner approved this on 2026-11-24 (AGENTS.md §7); link the approval in the PR.
1. **Regression test for the `model_id` collision (bug).** Two code shas with the same spec must not reuse each other's predictions. The test fails today through `write_predictions_or_existing`. The fix lands in PR 3.
2. **Schemas first:** `RunManifest`, `BuildManifest`, `OfficialPointer`, `AppliedRecord`, `InterventionRecord`, `ExperimentManifest`, the outcomes and `squad_state` tables, and the standard-key mixin for L2 tables.
2b. **Typed component params (§3.7).** One pydantic params model per component kind, `extra="forbid"`. Migrate component by component: the first PR adds the models and a test that each component's resolved defaults equal today's `.get` defaults (output provably unchanged). Then `spec_hash` over the resolved spec, with `name` excluded.
2c. **Spec registry and generated variants.** `registry.lock.yaml` with a CI hash-pin check; `specs/sha=` store; `--set path=value` overrides; grid expansion in EvalSpecs; auto-labels; `fplt spec freeze`. Tests: grid order doesn't change hashes, an unknown override path raises, two names with one hash are flagged.
3. **Run writer.** `fplt.store.runs.RunWriter`: a context manager that mints `run_id`, writes the manifest at start (`status=running`), writes tables, writes `_SUCCESS` and the final manifest last, and refuses to write into an existing run. `predict`, `plan` and backtest entry points write through it. Old paths keep being written in parallel during migration.
4. **Outcomes table + ledger view.** Stage `player_fixture_outcomes`. Add `Store.ledger()`. Switch metrics readers to the view. Remove `fill_actuals` mutation once readers have moved.
5. **Curated builds.** Content-addressed table writer, build manifests, `PITView(build_id, …)`. `staging_version` constant with a test that fails if staging output changes without a bump (golden hash over a recorded fixture set).
6. **Official lines + ops wiring.**
   - `lines:` in `production.yaml`, validated: exactly one primary, unique line names.
   - The ops job runs one predict run per distinct spec, then a plan run per line, then writes the per-line pointers, `applied.json`, `submitted.json` and any intervention records.
   - The publish step (website) reads only official pointers.
6b. **Virtual squads.** `squad_state` in/out tables and the `prev_run_id` chain. Tests: the chain carries bank, free transfers and chips correctly; a chip used on a virtual line is unavailable to it later; a double gameweek; a blank gameweek; and initialising a line mid-season from the real squad.
7. **Experiments (§3.8).**
   - `ExperimentManifest`, `BatchManifest` and event schemas.
   - promote-eval mints `experiment_id`, runs arms as runs with catalog arm reuse, and writes the standard `results/` tables. `registry.yaml` gains `experiment_id`.
   - `fplt exp submit --batch`, `fplt compare`.
   - Tests:
     - status folding from events, including out-of-order and duplicate events
     - arm reuse hits only on an exact key match
     - a batch deduplicates a shared control
     - gates are frozen at submit (editing the EvalSpec afterwards doesn't change a submitted experiment's verdict)
8. **Store read API + catalog + `fplt sql`.**
9. **Legacy import.** Wrap existing `predictions/`, `plans/`, `ledger/` and eval outputs as `kind=legacy` runs. Best-effort manifests (`code_sha` unknown) and `run_id = legacy-<model_id>-<as_of>`, so old results stay retrievable alongside new ones.
10. **Cut-over.** Stop writing the old paths; old directories stay as read-only archives.

Tests throughout: write-once refusal, `_SUCCESS` semantics (an incomplete run is invisible to `Store`), sortable `run_id` uniqueness under parallel minting, build dedup (an unchanged table reuses its sha), pointer write-once, ledger-view equivalence (the view after migration returns exactly the rows the old ledger had), and double and blank gameweeks in every keyed table.

## 9. Decisions for you

- [x] Approve the ledger-as-view change and the run-partitioned storage contract (AGENTS.md §7). **Approved by owner, 2026-11-24.**
- [x] Official checkpoint: **fixed T−90m** for every line, independent of what was applied (§3.5).
- [x] Points distributions: **both tables (player-fixture and player-gameweek), official runs only**, bins −5…30+ (§3.3).
- [x] Initial slate: primary `v4_h3`, plus candidates **`v4_h5`, `v4_h7`, `v4_h9`** (§4.1).
- [x] Virtual squads for **every** candidate line (§4.1).
- [x] **Retire** most model YAMLs into registry entries plus EvalSpec grids; keep production, frozen parents and named forks used by EvalSpecs; old names stay in `registry.lock.yaml` (§3.7).

Decisions recorded 2026-11-24.
