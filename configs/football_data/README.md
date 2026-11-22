# football-data.co.uk team name map

`team_names.yaml` maps E0 CSV `HomeTeam` / `AwayTeam` strings onto FPL
`team_key` (stable team `code`).

## When to update

1. **Promoted club** — add the football-data spelling → FPL code under `teams`.
2. **Renamed spelling** — football-data spellings are stable (`Man United`,
   `Nott'm Forest`, `Wolves`); if a new alias appears, add it alongside the old
   one (same `team_key`).

## Fetch → stage

```bash
# Write-once E0 CSV under raw/football_data/season=YYYY-YY/E0.csv
make fetch-football-data SEASON=2024-25
# equivalent: python scripts/fetch_football_data.py --out "$FPLT_DATA_DIR/raw" --season 2024-25

# Stage match_odds (Dagster match_odds_ingest job).
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
