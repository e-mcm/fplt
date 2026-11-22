# Benchmark projection archives (N1)

Canonical table: **`benchmark_projections`**.

- Schema: `fplt.schemas.benchmark_projections.BenchmarkProjectionSchema`
- Library: `fplt.data.benchmark_projections`
- On disk: `$FPLT_DATA_DIR/benchmark_projections/as_of=<season>_gwNN/part.parquet`

This is the single archive for paired Tier-A/B forecast boards. **Do not add a
parallel table.** FDB-118 (FPL Review) should append `source=fpl_review` rows
here after a terms check — manual CSV / licensed API only, no scrape.

## Columns

| Column | Notes |
|---|---|
| `source` | `ep_next` \| `openfpl` \| `fpl_review` |
| `as_of_season`, `as_of_gw` | Deadline the archive belongs to |
| `target_season`, `target_gw` | Predicted gameweek (horizon 1 = same GW) |
| `player_key`, `fixture_id` | Player–fixture grain |
| `xpts` | Expected points |
| `available_at` | Archive stamp, **strictly before** the deadline |
| `quality_flag` | Nullable; OpenFPL uses `openfpl_degraded` when Deep/PPDA is missing |

## What this PR archives

| Source | How | Notes |
|---|---|---|
| `ep_next` | `ep_next_v1` from PIT snapshots | Required |
| `openfpl` | `openfpl_v1` (repo adapter) | Required; labelled `openfpl_degraded` without `understat_team_matches` |
| `fpl_review` | Stub only | `fpl_review_projections` raises until terms allow ingest |

Missing required sources fail the job (`BenchmarkArchiveError`).

## Run before a deadline (live)

Automation already hooks this:

1. Dagster `pre_deadline` sensor (every 15 min) fires when a deadline is inside
   three hours and the plan **or** `benchmark_projections` archive is missing.
2. Job materialises `benchmark_archive` and `pre_deadline_plan`.
3. `pre_deadline()` also calls `archive_if_missing` before planning.

Manual:

```bash
# Due deadline only (inside the 3h window)
uv run fplt archive-benchmarks

# Or via Dagster
make dagster-dev
# materialise asset: benchmark_archive
```

Files land at:

```text
$FPLT_DATA_DIR/benchmark_projections/as_of=2025-26_gw12/part.parquet
```

## Backfill a past gameweek

Reconstruct from staged PIT data at that deadline. `available_at` is stamped
one second before the deadline when the clock is already past it.

```bash
uv run fplt archive-benchmarks --season 2024-25 --gw 10
# Optional local-only: --skip-openfpl  (not for production archives)
```

OpenFPL needs a usable `model_dir` (see [`docs/openfpl.md`](openfpl.md)).
Without Understat team Deep/PPDA, rows are still written with
`quality_flag=openfpl_degraded`.

## FPL Review (manual ingest)

FPL Review is a paid service. **Check its terms before any real data lands**
(summary in [`docs/fpl_review.md`](fpl_review.md)).

- Schema accepts `source=fpl_review`.
- Automated fetch stub `fpl_review_projections(...)` still raises (no scrape).
- Manual path: `fplt ingest-review path/to/drop.csv` →
  `fplt.data.fpl_review` merges into the same deadline parquet via
  `merge_benchmark_source` (keeps existing `ep_next` / `openfpl` rows).

```bash
fplt ingest-review ./exports/gw12.csv \
  --available-at 2025-11-21T17:00:00Z \
  --copy-raw
```

## PIT reads

```python
from datetime import datetime
from fplt.data.benchmark_projections import visible_benchmarks

# Rows with available_at <= cutoff
frame = visible_benchmarks(data_dir, cutoff)
```

Archives are invisible before their `available_at` stamp.

## Paired compare (N2)

Once archives exist and the candidate model is on the prediction ledger:

```bash
make archive-compare SEASON=2024-25 MODEL=dc_layered_v2
# or
uv run fplt archive-compare --season 2024-25 --model dc_layered_v2 \
  --from-gw 1 --to-gw 38 --sources ep_next openfpl
```

Library: `fplt.backtest.archive_compare.compare_to_archives`.

- Joins on player–fixture (+ as-of / target keys).
- Drops any row missing a pre-deadline snapshot on any compared arm.
- Board: RMSE (headline), MAE, Spearman by position, zeros/blanks/tickers/haulers.
- Intervals: existing HAC / stationary bootstrap vs each archive source.
- Writes `$FPLT_DATA_DIR/archive_compare/<stamp>/` (JSON, parquet, `board.md`).
- `fpl_review` is accepted as a `--sources` value when those rows exist; there is still no scrape path.
