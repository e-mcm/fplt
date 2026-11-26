# scripts/

Standalone capture / bootstrap helpers. Prefer **`make …`** or **`uv run fplt …`** when a library CLI exists; use these when GHA or a thin stdlib job must stay Dagster-free.

| Script | Purpose | Typical invoke |
|---|---|---|
| `snapshot.py` | FPL API → write-once gzipped JSON (throttled) | `.github/workflows/snapshot.yml` |
| `fetch_understat.py` | Understat match + league teamsData | `make fetch-understat SEASON=…` |
| `fetch_api_football.py` | API-Sports `/injuries` | `make fetch-api-football SEASON=…` |
| `fetch_football_data.py` | football-data.co.uk E0 CSV | `make fetch-football-data SEASON=…` |
| `fetch_overall_rank.py` | Overall-league points-at-rank | `make fetch-overall-rank` |
| `bootstrap_vaastav.py` | Historical seasons from vaastav | `make bootstrap-vaastav` |
| `eval_board.py` | Multi-model ledger helper | `make eval-board SEASON=…` |
| `perf_report.py` | Rank `$FPLT_DATA_DIR/timing` hotspots | `make perf-report` |
| `mc_bench.py` | Per-step MC micro-bench (§1.1) | `make mc-bench` |
| `check_workflow_timeouts.py` | Fail if any GHA job lacks `timeout-minutes` (ops-11) | `make check` / CI lint |
| `secret_rotation_reminders.py` | List due secret-rotation reminder issues (ops-11) | `.github/workflows/secret-rotation-reminder.yml` |

## Invariants

- **Raw is write-once.** Never overwrite an existing capture path.
- Snapshot / fetch scripts must stay network-thin and fixture-testable; unit tests use `tests/fixtures/`, never live HTTP.
- Secrets (`FPLT_API_FOOTBALL_KEY`, etc.) come from env / `~/.config/fplt/env` — never argv, never commit.
- Cadence for GHA vs local: [`docs/dagster-cadence-and-setup.md`](../docs/dagster-cadence-and-setup.md).

## Agents

- Do not put business logic here that belongs in `src/fplt/` — Dagster and CLI must call the library.
- Lint scope includes `scripts/` (`make check`).
- Runtime dependency equivalence (ops-12) lives in `fplt.automation.deps_equivalence`
  / `fplt deps-equivalence` + `.github/workflows/deps-equivalence.yml`, not here.
