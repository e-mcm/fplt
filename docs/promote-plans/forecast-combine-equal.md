# Promote plan: forecast combination (`forecast_combine_equal_v1`)

**Eval status:** **evaluated — drop** (FDB-217).  
**Result:** [forecast-combine-promote.md](../forecast-combine-promote.md) · Archive: `internal/forecast-combine-promote/`  
**Path:** ModelSpec first (RMSE) → decision board (season pts) before any `PRODUCTION_MODELS` flip.  
**Inventory:** [unpromoted-improvements.md](../unpromoted-improvements.md) · [forecast-combination-bakeoff.md](../forecast-combination-bakeoff.md)

| | Control | Treatment |
|---|---|---|
| Spec | `dc_layered_v2` | `forecast_combine_equal_v1` (DC v2 + LGBM equal blend) |

### Artifact / runner
- Bakeoff: `configs/evals/forecast_combination.yaml` (descriptive).
- Promote EvalSpec: `configs/evals/forecast_combine_equal_ab.yaml` (#157).
- Decisions: `fplt decisions --chip-board` at **H=3** optimised.
- Solver: H=3 auto-prune (#156) — hang fixed; not sufficient for keep.

### Windows
- Forecast: 2024-25 GW1–20 (cite bakeoff) + **2025-26 GW1–20 confirmation** (ran).
- Decisions: 2025-26 + 2024-25 H=3 optimised (ran).

### Gates (outcomes)
- Forecast: RMSE ≤ control on confirmation → **FAIL** (Δ −0.024).
- Decisions: season pts ≥ DC at H=3 both seasons → **FAIL** (2025-26 +194; 2024-25 −82).
- SPA inconclusive alone ≠ keep.

### Flip PR surface
- **Not opened.** `PRODUCTION_MODELS` stays `dc_layered_v2`.

### Ordered steps
1. Fix H=3 hang — **done** (#156).
2. Confirmation RMSE — **done** (fail).
3. Chip-board H=3 both seasons — **done** (fail both-seasons gate).
4. Written call → **drop** (no flip PR).
