# Understat id maps

`id_maps.yaml` maps Understat integer ids onto FPL codes used elsewhere in
`fplt`:

| Map key | Understat field | FPL field |
|---|---|---|
| `teams` | `tmpl.h` / `tmpl.a` (and roster `team_id`) | `team_key` (team `code`) |
| `players` | roster `player_id` | `player_key` (element `code`) |

## When to update

1. **New club in the EPL** — Understat assigns a stable team id. Look it up on
   a finished match (`getMatchData` → `tmpl.h` / `tmpl.a`, or a roster
   `team_id`) and add `understat_team_id: fpl_team_code` under `teams`.
2. **Unmapped players** — staged rows keep `player_key` null. After you decide
   the FPL element `code`, add `understat_player_id: player_key` under
   `players` and re-run the `understat_ingest` Dagster job (writes collapse on
   the table keys, so remapped rows replace the nulls).

FPL element and team `id`s change every season; **codes do not**. Never map to
season-local `element_id` / team `id`.

## Fetch → stage

```bash
# Write-once raw match JSON under $FPLT_DATA_DIR/raw/understat/...
python scripts/fetch_understat.py --out "$FPLT_DATA_DIR/raw" --season 2024-25

# Or materialise the understat_ingest asset / understat_ingest job in Dagster.
make dagster-dev
```

Raw files are never overwritten. Maps live in git so reviews catch bad joins.
