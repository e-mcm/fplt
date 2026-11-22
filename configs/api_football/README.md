# API-Football id maps

`id_maps.yaml` maps API-Football (API-Sports) integer ids onto FPL codes used
elsewhere in `fplt`:

| Map key | API-Football field | FPL field |
|---|---|---|
| `teams` | `team.id` on `/injuries` | `team_key` (team `code`) |
| `players` | `player.id` on `/injuries` | `player_key` (element `code`) |

## Mapping gaps

- **Players start empty.** Unmapped players still stage into `player_injuries`
  with a null `player_key`. Ingest logs a warning with a sample of unmapped
  API ids. Until mapped, the minutes injury overlay cannot see them.
- **Teams are seeded for recurring EPL clubs.** Promoted sides and id churn
  need a manual check against `GET /teams?league=39&season=YYYY` each summer.
  Unmapped teams stage with a null `team_key` (rows are kept).
- **FPL fixture ids** are not mapped in Package B (`fixture_id` stays null;
  `api_fixture_id` is stored). Add a fixture map later if fixture-grain
  overrides are needed.

FPL element and team `id`s change every season; **codes do not**. Never map to
season-local `element_id` / team `id`.

## Secrets

Live fetch needs an API-Sports key. Set **one** of these env vars (never argv,
never commit):

| Env var | Where |
|---|---|
| `FPLT_API_FOOTBALL_KEY` | Preferred — `~/.config/fplt/env` or repo `.env` (gitignored) |
| `API_FOOTBALL_KEY` | Also accepted by `scripts/fetch_api_football.py` |
| `APISPORTS_KEY` | Also accepted by the fetch script |

GitHub Actions: add repository secret `API_FOOTBALL_KEY` (workflow maps it into
the job env). Unit tests use recorded fixtures under `tests/fixtures/api_football/`
and never call the network.

## Fetch → stage

```bash
# Write-once raw injuries JSON under raw/api_football/...
# Skips when a capture newer than --min-interval-hours (default 4) exists.
make fetch-api-football SEASON=2024-25
# equivalent:
#   export FPLT_API_FOOTBALL_KEY=...   # or API_FOOTBALL_KEY / APISPORTS_KEY
#   python scripts/fetch_api_football.py --out "$FPLT_DATA_DIR/raw" --season 2024-25

# Stage player_injuries (Dagster api_football_ingest job).
make dagster-dev
```

Free tier is ~100 requests/day. One league+season `/injuries` call per capture
is enough; the 4h min-interval matches upstream refresh cadence.

## Minutes overlay

`empirical_minutes_v2` (and v1) accept `use_injury_overlay: true`. Default is
`false` so production `dc_layered_v2` stays FPL-tags-only. Enable via a YAML
fork for A/B, e.g. `configs/models/empirical_minutes_v2_injuries.yaml`.
