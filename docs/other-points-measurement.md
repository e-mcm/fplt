# Other-points / haul measurement (FDB-240)

Diagnostic board for Rank-3 match-conditioned other points. **Measurement only** — no keep A/B and no production flip.

Today the simulator adds bonus / saves / cards / defcon / residual other from the minutes bucket alone (`sample_other` only adds mean-preserving noise). Those channels are **independent of simulated goals and clean sheets**. This board quantifies haul calibration and the residual structure a later match-conditioned other model would target.

## What it reports

On horizon-1 walk-forward rows from a layered points model (default `dc_layered_v4`):

1. **Haul Brier + log score** overall, on the decision-relevant pool, by position, and by realised minutes bucket.
2. **Top xPts-decile haul calibration** — `haul_mean_p`, `haul_mean_y`, `haul_cal_gap`, plus haul Brier inside the top decile.
3. **Match-context other residual** — predicted other (component at the realised minutes bucket) vs actual other (bonus+saves+cards+defcon+residual), sliced by team goals band (`0` / `1` / `2+`), clean sheet, and whether the player scored; plus Pearson corr of residual vs team goals on 60+ rows.
4. **Captain regret proxy** — unconstrained decision-pool max-`xPts` vs max-`actual_pts` per GW (not the squad-constrained `fplt regret-board`).

## CLI

```bash
make other-measure SEASON=2025-26 ARGS='--from-gw 1 --to-gw 19'
# or
uv run fplt other-measure --season 2025-26 --from-gw 1 --to-gw 19
# faster smoke:
uv run fplt other-measure --season 2025-26 --from-gw 1 --to-gw 5 --n-sims 200
```

Writes `$FPLT_DATA_DIR/other_points_measurement/<season>_gw…/{summary,captain_regret,detail}.*` plus `captain_regret.md`.

## Planned windows (eval protocol)

Per Project store `eval-protocol.md` defaults — declare before any later match-conditioned keep A/B. This measurement PR does **not** run those A/Bs and does **not** flip `PRODUCTION_MODELS`.

| Season | Development | Confirmation | Sealed |
|---|---|---|---|
| `2024-25` | GW1–20 | GW21–38 | — |
| `2025-26` | GW1–19 | GW20–38 (finished / `data_checked` only) | Unfinished GWs |
| `2026-27+` | First half of finished GWs | Second half of finished GWs | Next unfinished stretch |

Primary read for structural residuals: development window. Confirmation stays untouched until a keep candidate exists. Clearest keep metrics remain haul Brier, top-decile cal, and (squad) captain regret.
