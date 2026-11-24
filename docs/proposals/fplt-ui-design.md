# fplt results UI: local Streamlit dashboard

**Status:** proposal · **Depends on:** `fplt-data-model.md` (runs, experiments, `fplt.store`)

## 1. Principles

1. **The UI never computes model output.** It reads runs and experiments through `fplt.store`. Anything that needs fitting or simulating is a run or experiment started from the CLI (or from a UI button that just shells out to the CLI) and shows up when it's done.
2. **Logic lives in a library, not in the app.** Every table and chart is built by pure polars functions in `src/fplt/analysis/`, which are unit-tested like everything else (AGENTS.md §3). The Streamlit files only lay out widgets, the same "thin wrapper" rule as Dagster.
3. **Immutable data means caching is free.** Runs never change, so `st.cache_data` keyed by `run_id` / `experiment_id` never needs invalidating. Only the catalog and experiment status are re-read, every 30–60 s on pages that show live progress.
4. **The same helpers feed the website later.** The public site uses `fplt.analysis` plus the publish step. Streamlit stays private and local.

## 2. Layout

```
src/fplt/analysis/          # pure functions, tested; no Streamlit imports
  compare.py                # horizon-summed xPts per player for N arms/runs, diffs, rank changes
  experiments.py            # batch table, normalised forest data, stage panels, decision-band data
  sweeps.py                 # response curves, heatmaps, holdout-vs-selection
  reliability.py            # binned calibration for any probability output
  lines.py                  # official slate, virtual-team season curves
  players.py                # one player's predictions across runs vs actuals, component breakdown
  plans.py                  # plan diffs: transfers, XI, captain, chips across runs
src/fplt/ui/
  app.py                    # st.navigation entry point
  pages/overview.py  experiments.py  experiment.py  lines.py  player.py  runs.py
  renderers.py              # view name -> renderer registry (§3.2)
```

- `fplt ui` → `streamlit run src/fplt/ui/app.py`, reading `$FPLT_DATA_DIR`. It pulls missing immutable objects from R2 on demand through `Store`, on the **Mac only**. Not exposed on VMs or any network interface; it binds to `localhost`.
- Dependencies go in an optional group (`uv add --optional ui streamlit plotly`), so CI and agents don't install them unless needed. Justify them in the PR (AGENTS.md workflow). Charts use **Plotly** (zoom and hover matter for scatter, forest and reliability views); renderers return Plotly figures shown with `st.plotly_chart`.

## 3. Pages

### Overview
Next deadline and checkpoint state, the applied line for the current gameweek, `status.json` (token expiry, last capture, quarantines), and running batches with a progress bar per experiment.

### Experiments: the batch view
Experiments are now general (`fplt-experiments.md`): any number of arms, several measured stages, four decision types, oracle arms and tags. The batch page therefore can't assume "one metric, treatment vs control":

- **Table:** one row per experiment with question, tags, intervention stage → measured stages (e.g. `minutes → component, season`), status (k/n arms, short-circuited stages), decision type, primary metric and stage, and the verdict. Filter by tag, intervention stage or verdict; failed experiments sort first.
- **Normalised forest plot.** Different experiments have different primary metrics, so plot each experiment's best non-oracle arm as **Δ / margin** (or Δ / noise-floor SD when there is no margin), with its CI.
  - superiority: does the CI clear 0?
  - non-inferiority: does it clear −1 (the margin)?
  - equivalence: is it inside ±1?

  Hover for the raw Δ and units. Ten experiments, one axis.
- Arm and component reuse across the batch ("control shared by 10; Dixon–Coles outputs reused by 7").

### Experiment page: stage by stage
1. **Header:** question, the pipeline diagram with the **intervention node highlighted** and the **measured nodes ringed**, the decision rule in words, the verdict, and the context (build, windows, seeds, population).
2. **Arms table:** each arm's patch exactly as declared (the harness has already checked it equals the resolved diff), its config hash, run ids, and whether outputs were reused from cache. Oracle arms are labelled "reference ceiling".
3. **One panel per measured stage, in pipeline order.** A short-circuited stage shows greyed out, with the gate that stopped it. Each panel shows:
   - per-arm metrics by window, and paired Δ vs control with CI;
   - a forest plot of **all arms** for that stage's primary metric, with the **noise-floor band** shaded (from the experiment's seed replicates, or the latest robustness run of the same base) and the oracle arm drawn as a vertical ceiling line;
   - views specific to the stage (§3.1).
4. **One global slice selector** (position, flag at deadline, horizon step, season phase, double/blank gameweek, price band) re-filters every panel, so segment analysis isn't a separate tool.
5. **Decision panel**, drawn according to the decision type:
   - superiority: CI against 0;
   - non-inferiority: CI against −margin;
   - equivalence: CI inside the ±margin band, with both one-sided tests shown;
   - none: no verdict, just the descriptive panels.
6. **Holdout check** for sweeps: the metric on the selection window vs the holdout per arm. A winner that collapses on holdout is flagged as overfit.
7. Links: manifests, `report.md`, and "copy registry entry".

### 3.1 Stage-specific views

| Stage | Views |
|---|---|
| `component:minutes` | reliability diagrams for P(played) and P(60+) per arm; bucket-PMF calibration; team XI consistency (Σ P(start), GK count); metric by horizon step; biggest misses table |
| `component:score` | RPS / log loss by arm; clean-sheet reliability; calibration of total goals; Δ by team |
| `component:involvement`, `other` | per-channel scorecard (eval-01) in both modes (model's own minutes / actual minutes played); reliability for P(goal), P(assist), bonus, cards |
| `prediction` | RMSE and haul Brier per arm; for N arms, pick any two for the A-vs-B scatter and diff table (the old Compare page); per-player component breakdown explaining a gap |
| `plan` | plan diff across arms per step (transfers, XI, captain, chips); share of deadlines where the arms' decisions differ |
| `season` | cumulative Δ season points by gameweek with CI ribbon; attainable share; regret; virtual-team curves for live-forward experiments |
| `env` (equivalence) | bitwise-match rate, distribution of absolute per-row differences, the max difference with the row that produced it |
| sweeps (any stage) | **response curve** (metric vs parameter value, CI bars) for one-parameter grids; **heatmap** for two-parameter grids; holdout overlaid |
| live forward | metric accruing by week, with CI narrowing and an estimate of the weeks until the CI excludes the margin |

### Snapshot comparisons
"H=10 points for every player, A vs B, as of now" is a `mode: snapshot` experiment, so it opens on the same experiment page with only `prediction` (and optionally `plan`) panels. The horizon slider, scatter, diff table and per-player breakdown are the `prediction` stage views above. There's no separate Compare page; a "New comparison" shortcut runs `fplt compare …` and opens the result.

### 3.2 Renderer registry: new measures need no UI code
Each `Measure` (`fplt-experiments.md` §8) declares which views its output supports, e.g. `views=["reliability", "by_slice", "forest"]`. The UI maps view names to renderers through a small registry in `fplt/ui/renderers.py`:
- `@renderer("reliability")` consumes any table with `pred_bin, mean_pred, observed, arm`;
- `@renderer("forest")` consumes `arm, diff, ci_lo, ci_hi`.

A new scorer that writes those standard columns appears in the UI automatically. Renderers are thin; the data shaping lives in `fplt.analysis` and is tested there.

### Official lines
Per gameweek, the slate of lines with predicted vs realised points. Season curves of each line's virtual team (cumulative points, rank-equivalent). Which line was applied each week, and any intervention records.

### Player
One player across time: predictions from the official line (and optionally any runs) at each deadline vs actual points, minutes and goals. Component history: how P(start) evolved as the flags changed. Useful for "why did we keep benching him?".

### Runs explorer
A catalog table filtered by kind, spec label, spec hash, code sha, host, date and status. Open a run to see its manifest and the first rows of each table. Diff two run manifests.

## 4. Starting long runs from the UI (later: after the read-only pages)

A "Submit" form on the Experiments page that picks EvalSpecs, a backend and a batch name, and runs `fplt exp submit …` as a subprocess (or a Modal call). The page then follows progress from experiment events. It never runs arms inside the Streamlit process. Keep this behind a local-only flag: the UI has no authentication and must never be exposed publicly.

## 5. Testing

- `fplt.analysis` functions: unit tests on synthetic runs and experiments from `tests/builders.py`. Cover:
  - horizon summing with double and blank gameweeks;
  - players present in only one arm (outer join, null Δ);
  - rank ties;
  - normalised Δ/margin for each decision type, including the no-margin case falling back to the noise floor;
  - response curves with a failed arm;
  - holdout-vs-selection alignment;
  - a short-circuited stage producing an empty panel with its reason.
- Renderer registry: each renderer is tested against a minimal table with its standard columns. A measure declaring an unknown view fails at registration.
- App smoke tests with `streamlit.testing.v1.AppTest` against a tiny fixture store containing one experiment of each decision type and one snapshot comparison: each page renders without exceptions and shows the expected panels. Mark slow if over ~2 s.
- No network: the fixture store is local.

## 6. Alternatives considered

- **marimo / Jupyter:** better for one-off digging, and they can import the same `fplt.analysis` helpers. Keep them for exploration; the dashboard is for repeated views.
- **Evidence.dev / Observable Framework:** good for the public static site later, since they build from SQL or JSON. Not needed locally.
- **Dash / Panel:** more control, more code. Streamlit is enough for a single-user local tool.

## 7. PR sequence

1. `fplt.analysis.compare` and `reliability` + tests; `fplt compare` CLI (needs data-model steps 3 and 7).
2. Streamlit skeleton, `fplt ui`, the renderer registry with `forest`, `reliability`, `scatter_diff`, the experiment page for snapshot experiments (prediction stage only), and an AppTest smoke test.
3. Stage panels for `component:*` and `prediction`, the global slice selector, and the decision panel for all four types.
4. Batch page with the normalised forest plot; `season` and `plan` panels.
5. Sweeps (response curve, heatmap, holdout), `env` equivalence views, live-forward views.
6. Overview, Runs explorer, Official lines, Player pages.
7. Optional: the submit form.
