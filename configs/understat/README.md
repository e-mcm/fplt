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
   the table keys, so remapped rows replace the nulls). For candidates:

   ```bash
   uv run fplt idmap suggest --source understat --season 2024-25 --out /tmp/us.yaml
   ```

   Review the high-confidence fragment in a PR; nothing is auto-merged.
   Season checklist: [`deploy/README.md`](../../deploy/README.md) § Season
   rollover.

FPL element and team `id`s change every season; **codes do not**. Never map to
season-local `element_id` / team `id`.

## Fetch → stage

```bash
# Write-once raw match JSON + league teamsData (Deep/PPDA) under raw/understat/...
make fetch-understat SEASON=2024-25
# equivalent: python scripts/fetch_understat.py --out "$FPLT_DATA_DIR/raw" --season 2024-25

# Stage player matches + understat_team_matches (Dagster understat_ingest job).
make dagster-dev
```

`teams/captured_at=.../teams.json.gz` holds league-page `teamsData` (Deep,
PPDA, team xG history). Ingest joins `match_id` from the dates snapshot.
Default delay between match requests is 0.2s — raise `--delay` if Understat
rate-limits. Offline tests use fixtures under `tests/fixtures/understat/`.

Raw files are never overwritten. Maps live in git so reviews catch bad joins.
