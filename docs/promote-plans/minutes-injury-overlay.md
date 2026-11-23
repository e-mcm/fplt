# Promote plan: minutes injury overlay

**Eval status:** **in flight** — FDB-208 · agent `bc-be9d4b44-bb4c-5547-9b90-78724e58b4e6` (availability overlays). **Do not duplicate.**  
**Path:** ModelSpec · primary metric **RMSE** (decision pool) · minutes Brier secondary.  
**Inventory:** [unpromoted-improvements.md](../unpromoted-improvements.md)

| | Control | Treatment |
|---|---|---|
| Spec | `dc_layered_v2` | `dc_layered_injuries_v1` (extends v2; `components.minutes.params.use_injury_overlay: true`) |
| Minutes | `empirical_minutes_v2` overlay **off** | same + staged `player_injuries` |

### Artifact / runner
- EvalSpec `configs/evals/minutes_injury_overlay_ab.yaml` (to land with layered fork).
- `uv run fplt promote-eval --eval minutes_injury_overlay_ab --execute`
- Window: **2025-26** snapshot-backed GWs preferred (vaastav has no injuries); confirm on a second window if keep.

### Gates
- Staging: non-empty `player_injuries` with `available_at` under PIT.
- Prediction: treatment RMSE ≤ control (prefer RMSE; MAE continuity).
- Components: minutes Brier played / ≥60 not worse on snapshot `source=fpl_api` rows.
- No `PRODUCTION_MODELS` edit until keep + confirmation.

### Flip PR surface
- Turn overlay **on** in production minutes params **or** adopt `dc_layered_injuries_v1` into ops `models:` / `PRODUCTION_MODELS`.
- Prefs / design note: injury overlay promoted.

### Cost / risk
- Low compute (one promote-eval). Risk: bad id-map → false zeros; overlay must never raise an FPL zero (already coded).

### Ordered steps
1. Confirm `API_FOOTBALL_KEY` + staged `player_injuries` for window.
2. Land layered fork YAML + EvalSpec (small PR if missing).
3. Primary window execute → written call.
4. Confirmation window (or snapshot-only slice).
5. Flip PR only on keep.
