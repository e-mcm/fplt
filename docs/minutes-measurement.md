# Minutes measurement split (FDB-246)

Diagnostic board for the minutes / availability lever. **Measurement only** — no keep A/B and no production flip.

## What it reports

On snapshot-backed rows (`player_fixture_stats.source = fpl_api`):

1. **Error regimes** (priority partition) with Brier + log score on played / ≥60 / start:
   - `availability` — FPL availability gate &lt; 1 at the deadline
   - `early_sub` — fully available and started (`starts ≥ 1`); ≥60 is the hinge
   - `rotation` — fully available and did not start
2. Further slices: FPL flag (`a` / `d` / `unavailable` / `chance_lt_100`), position, returning-from-absence (prior appearance was a blank).
3. **xPts-weighted Brier** on the decision-relevant pool (weights = PIT `ep_next`, else `P(≥60)`).
4. **Weekly misses** ranked by `|ΔP(start)| × xPts` (`P(start)` operationalised as `P(≥60)` from the minutes PMF).

## CLI

```bash
make minutes-measure SEASON=2025-26 ARGS='--from-gw 1 --to-gw 19'
# or
uv run fplt minutes-measure --season 2025-26 --from-gw 1 --to-gw 19
```

Writes `$FPLT_DATA_DIR/minutes_measurement/<season>_gw…/{summary,misses,detail}.*` plus `misses.md`.

## Planned windows (eval protocol)

Per Project store `eval-protocol.md` defaults — declare before any later keep A/B (blend / GK XI / overlays). This measurement PR does **not** run those A/Bs.

| Season | Development | Confirmation | Sealed |
|---|---|---|---|
| `2024-25` | GW1–20 | GW21–38 | — |
| `2025-26` | GW1–19 | GW20–38 (finished / `data_checked` only) | Unfinished GWs |
| `2026-27+` | First half of finished GWs | Second half of finished GWs | Next unfinished stretch |

Primary read for structural bugs: development window on the season with the densest `fpl_api` status snapshots. Confirmation stays untouched until a keep candidate exists.
