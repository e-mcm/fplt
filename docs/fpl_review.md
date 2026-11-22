# FPL Review — terms summary and manual ingest

**No scraping.** This repo only accepts a human-exported CSV/JSON drop into the
canonical `benchmark_projections` archive (see
[`docs/benchmark-archives.md`](benchmark-archives.md)). Automated fetch of
Review pages or APIs is out of scope until Eamon explicitly approves a licensed
path.

Schema: `fplt.schemas.benchmark_projections.BenchmarkProjectionSchema`  
Loader: `fplt.data.fpl_review` / `fplt ingest-review`

## Terms summary (public pages, fetched 2026-11-22)

Sources:

- [Terms of Service](https://fplreview.com/tos) (last updated 01/03/2026)
- [Premium features](https://docs.fplreview.com/getting-started/premium-features/) (includes **Data Export: Export projections in CSV format**)
- [Export Projections](https://docs.fplreview.com/the-model/planner-interface/export_projections/)
- [Patreon sign-up](https://docs.fplreview.com/the-site/patreon-sub/)

| Topic | Reading |
|---|---|
| Personal use | Outputs may be used for **personal, non-commercial purposes only** (§5). |
| Scraping / bots | **Forbidden**: automated tools, scripts, bots, or scraping to access or extract data (§3). |
| Redistribution / competition | **Forbidden**: copy, aggregate, redistribute, republish, or reuse projections in a way that substitutes for or competes with the Service, including derivative tools, datasets, or automated systems (§5). |
| Premium export | Premium Patreon tier documents a **CSV export** of projections for analysis in spreadsheets or other tools — the intended manual path. |
| Access control | Do not bypass Patreon / subscription checks (§2–3). |

**Repo interpretation (not legal advice):** a subscribed user exporting CSV via
the product UI and keeping a private, local archive for personal model
evaluation is the intended path. Publishing Review projections, bundling them
into a redistributed dataset, or scraping the site is not.

## Recommended legal path before real Review data lands

1. **Eamon** takes (or confirms) a Patreon Premium subscription and accepts the current ToS.
2. Prefer the built-in **CSV export**; do not scrape HTML/JSON from the planner.
3. Keep archives **private** under `$FPLT_DATA_DIR` (gitignored). Do not commit Review rows to the repo or push them to public branches.
4. If unsure whether personal archival + offline evaluation against Review is acceptable under §5, **ask `formdork@fplreview.com`** before ingesting real data — especially if results will be published.
5. Until that approval (subscription + personal comfort / any needed clarification), Tier B remains blocked on Review archives; continue archiving `ep_next` and OpenFPL only.

## How to drop a file

Canonical columns (aliases in parentheses):

| Column | Notes |
|---|---|
| `as_of_season` | (`season`) deadline season, e.g. `2025-26` |
| `as_of_gw` | (`gw`, `gameweek`) deadline GW 1–38 |
| `target_season` / `target_gw` | Predicted GW (defaults to as_of when omitted) |
| `player_key` | FPL element **code** (stable). Map from Review’s season `ID` / FPL id via staged `players.element_id` → `player_key`. |
| `fixture_id` | FPL fixture id for that player–GW (DGWs → one row per fixture). |
| `xpts` | (`EV`, `GW_Pts`) expected points |
| `available_at` | UTC ISO save time, **strictly before** the GW deadline |
| optional | `quality_flag` |

`source` is always forced to `fpl_review` by the loader. Rows merge into
`$FPLT_DATA_DIR/benchmark_projections/as_of=<season>_gwNN/part.parquet`,
preserving any existing `ep_next` / `openfpl` rows for that deadline.

```bash
# Optional: stamp if the file omits available_at
fplt ingest-review path/to/projections.csv \
  --available-at 2025-08-15T10:00:00Z \
  --copy-raw
```

`--copy-raw` writes write-once under `data/raw/fpl_review/season=…/gw=…/captured_at=…/`.

Tiny synthetic fixtures (not real Review data): `tests/fixtures/fpl_review/`.

## What Tier B still needs

From the evaluation north star, Tier B needs a full season of paired,
pre-deadline Review projections. After this ingest path exists, still required:

1. Eamon’s subscription / terms sign-off (above).
2. Per-deadline manual (or later licensed) drops for every GW — automation that scrapes is not allowed.
3. Deadline archives for `ep_next` / OpenFPL on the same schema (already on main).
4. Paired prospective compare (N2) joining our forecasts to the archive on identical player–fixtures.
5. Event scores / calibration (N3) where Review supplies probabilities (optional; often blank from CSV export).
