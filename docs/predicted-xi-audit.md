# Predicted XI audit + late re-predict (FDB-253 / PR8–9)

Plumbing for the minutes plan’s XI-audit / late-refresh slice. **No keep A/B**, **no `PRODUCTION_MODELS` flip**, **no LLM press parse**, **no odds-rotation model**.

## Predicted XI audit

CLI:

```bash
make predicted-xi-audit SEASON=2025-26 ARGS='--from-gw 1 --to-gw 19'
# or
uv run fplt predicted-xi-audit --season 2025-26 --from-gw 1 --to-gw 19
```

Writes `$FPLT_DATA_DIR/predicted_xi_audit/<season>_gw…/{summary,detail}.*` plus `report.md`.

Reports (on staged `predicted_lineups` joined to official `player_fixture_stats.starts`):

1. **Start precision** overall and by `source`, hours-to-KO bucket (`0-6` / `6-24` / `24-72` / `72+`), confidence bin.
2. **Start recall** — share of official starters that appear as a predicted `start`.
3. **Bench→start rate** — predicted `bench` players who actually started (sub / XI miss).
4. **Full-XI coverage** — share of team-fixtures with ≥11 predicted starters.

Exit code `1` when overall start precision is below `--min-accuracy` (default `0.55`) — use as the promote-smoke abort before `predicted_xi_ab`.

## Late re-predict hook

Inside the 3 h pre-deadline window, `pre_deadline(..., refresh_if_stale=True)` (default) compares each plan parquet’s mtime to the newest `available_at` on `player_status` / `predicted_lineups` / `player_injuries` at or before the deadline. When a signal is newer, the plan directory is moved under `plans/<model_id>/superseded/` and the deadline is re-predicted / re-optimised.

Library: `fplt.automation.late_refresh`.

## Deferred (not in this PR)

- LLM / RSS press → chance features
- Odds-based rotation prior
- Separate early-sub histogram model beyond existing minutes v2 `p_sub` + predicted-XI bench targets
