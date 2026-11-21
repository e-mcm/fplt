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

Closing 1X2 averages (`AvgCH`/`AvgCD`/`AvgCA`) and O/U when present stage with
`odds_kind=closing_proxy` and `available_at=kickoff`. Closing ≠ FPL deadline —
do not claim deadline PIT. Offline tests use `tests/fixtures/football_data/E0.csv`.

Raw files are never overwritten. Maps live in git so reviews catch bad joins.
