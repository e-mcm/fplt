# football-data.co.uk team name map

`team_names.yaml` maps E0 CSV `HomeTeam` / `AwayTeam` strings onto FPL
`team_key` (stable team `code`).

## When to update

1. **Promoted club** — add the football-data spelling → FPL code under `teams`.
2. **Renamed spelling** — football-data spellings are stable (`Man United`,
   `Nott'm Forest`, `Wolves`); if a new alias appears, add it alongside the old
   one (same `team_key`).

## Cadence

| Layer | Cadence | Path |
|---|---|---|
| GHA `.github/workflows/football_data.yml` | 1st of month 06:00 UTC + `workflow_dispatch` | `snapshots` → `raw/football_data/season=…/captured_at=YYYY-MM-DD/E0.csv` |
| Local `make fetch-football-data` | On demand | Legacy `season=…/E0.csv`, or `CAPTURED_AT=YYYY-MM-DD` for a refresh |
| Dagster `match_odds_sensor` | every 30 min (daemon) | Stages when a new `E0.csv` appears under raw |

Raw is write-once: never overwrite an existing path. Monthly GHA uses a new
`captured_at=` day so the growing season CSV can land again. Later captures win
on `match_odds` unique keys when restaged.

## Fetch → stage

```bash
# Write-once E0 CSV under raw/football_data/season=YYYY-YY/E0.csv
make fetch-football-data SEASON=2024-25
# Monthly-style local refresh:
make fetch-football-data SEASON=2024-25 CAPTURED_AT=2025-11-01
# equivalent: python scripts/fetch_football_data.py --out "$FPLT_DATA_DIR/raw" \
#   --season 2024-25 --captured-at 2025-11-01

# Stage match_odds (Dagster match_odds_ingest job / match_odds_sensor).
make dagster-dev
```

Each E0 match stages up to two `match_odds` rows:

| `odds_kind` | Columns | `available_at` |
|---|---|---|
| `closing_proxy` | Closing averages (`AvgCH`/`AvgCD`/`AvgCA`, O/U `AvgC>2.5`…), falling back to opening when close is missing | kickoff |
| `prematch_scheduled` | Non-closing only (`AvgH`/`AvgD`/`AvgA`, `B365H`…, O/U `Avg>2.5`…). Closing columns are **not** relabelled | Conservative collection: **Friday 17:00 London** for Fri–Sun kickoffs; **Tuesday 17:00 London** for Mon–Thu |

Closing ≠ FPL deadline — do not claim deadline PIT. Prematch is usually closer to deadline-legal but still needs
`check_prematch_deadline_coverage` / `require_prematch_deadline_coverage`: any fixture whose FPL deadline falls before the estimated collection time drops out of prematch boards.

Offline tests use `tests/fixtures/football_data/E0.csv`.

## Re-ingest / verify

```bash
make fetch-football-data SEASON=2024-25   # write-once; skips if raw exists
# Stage (Dagster job or library):
python -c "
from pathlib import Path
from fplt.data.football_data import default_team_map_path, ingest_football_data, load_team_names
from fplt.data.store import DataStore
from fplt.settings import get_settings
s = get_settings()
ingest_football_data(DataStore(s.data_dir), s.data_dir / 'raw', load_team_names(default_team_map_path()))
"
# Audit kinds:
python -c "
from fplt.data.store import DataStore
from fplt.settings import get_settings
print(DataStore(get_settings().data_dir).read('match_odds')['odds_kind'].value_counts())
"
```

Raw files are never overwritten. Maps live in git so reviews catch bad joins.
