# Promote plan: availability overlays (Package A/B/C)

**Eval status:** executed Package A; **blocked on PIT signal** for A/B/C keep claims — see [availability-overlays-promote.md](../availability-overlays-promote.md).  
**Path:** ModelSpec · primary metric **RMSE** (decision pool) · minutes Brier secondary.  
**Out of scope:** Deep/PPDA / Understat (FDB-55).  
**Inventory:** [unpromoted-improvements.md](../unpromoted-improvements.md) · sources [availability-news-sources.md](../availability-news-sources.md)

## Control vs treatment

| Lever | Control | Treatment | Production today |
|---|---|---|---|
| **A. FPL status/chance/news gate** (#116) | `dc_layered_noavail_v1` (`use_availability_gate: false`) | `dc_layered_v2` (gate on) | **Already on** — no flip; board measures impact |
| **B. Injury overlay** (#118) | `dc_layered_v2` | `dc_layered_injuries_v1` (`use_injury_overlay: true`) | **Off** |
| **C. Predicted XI** (#119) | `dc_layered_v2` | `dc_layered_predicted_xi_v1` (`predicted_xi: true`) | **Off** |

Hard-starters / soft-squad stay on in both Package A arms (one lever = FPL gate only).

## Artifact / runner

```bash
uv run fplt promote-eval --eval fpl_availability_gate_ab --execute
uv run fplt promote-eval --eval minutes_injury_overlay_ab --execute   # when injuries staged
uv run fplt promote-eval --eval predicted_xi_ab --execute             # when lineups on finished GWs
```

## Windows / signal

| Lever | Primary window | Signal requirement |
|---|---|---|
| A | Prefer finished GWs with PIT `source=fpl_api` | Snapshots captured **before** each deadline (archive today starts ~2026-11-19 → first signal **GW11**, unfinished) |
| B | Same once `player_injuries` mapped | Non-empty PIT injuries with `player_key` |
| C | Finished GWs with pre-deadline `predicted_lineups` | Touchline/AF lineups before KO; `score_predicted_starts` smoke |

**Vaastav:** chance/news null, status=`a` → Package A arms **identical**. Injury/XI with empty tables → **identical**. Do not claim keep on vaastav-only boards.

**2026-11-23 finding:** GW1–10 promote-eval ran but fpl_api rows are not PIT-visible before GW11 → arms effectively tied (MC noise only).

## Gates

- Prediction: treatment RMSE ≤ control (MAE continuity).
- Components: minutes Brier played / ≥60 not worse on `fpl_api` rows.
- B/C: abort keep if overlay tables empty or XI start-accuracy is noise.
- No `PRODUCTION_MODELS` edit until keep + confirmation (A is already production).

## Ordered steps

1. Restage FPL snapshots → confirm `player_status` `source=fpl_api` share on window.
2. Land forks + EvalSpecs (this PR).
3. Execute Package A primary window → written call.
4. B/C: stage injuries (API key + id maps) / archive lineups on finished GWs; else document **blocked**.
5. Flip PR only on keep for B/C; A already live.

## Flip PR surface

- **A:** none (already production); call documents keep/drop of *retaining* the gate.
- **B:** turn overlay on in production minutes **or** adopt `dc_layered_injuries_v1`.
- **C:** merge `predicted_xi` into `dc_layered_v2` **or** ops model swap after accuracy audit.
