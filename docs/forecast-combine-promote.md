# Forecast combine equal — promote call (**drop**)

**Date:** 2026-11-23 · **Task:** FDB-217 (after FDB-210 / #156, FDB-214 / #157)  
**Control:** `dc_layered_v2` (unchanged; still production)  
**Treatment:** `forecast_combine_equal_v1` (equal blend of DC v2 + `lgbm_direct_v1`)  
**Plan:** [promote-plans/forecast-combine-equal.md](./promote-plans/forecast-combine-equal.md)  
**Prior bakeoff:** [forecast-combination-bakeoff.md](./forecast-combination-bakeoff.md)  
**Archive:** `internal/forecast-combine-promote/`

---

## Call

**Drop — do not flip `PRODUCTION_MODELS`.**

Confirmation RMSE fails the plan gate (equal worse than DC on 2025-26). H=3 optimised season points also fail confirmation (equal −82 vs DC on 2024-25) even though equal wins 2025-26. Solver hang is fixed (#156); the promote still fails both written gates.

| Gate | Result |
|---|---|
| Forecast: RMSE ≤ control on confirmation | **FAIL** — 2025-26 ΔRMSE (ctrl−treat, treat better >0) = **−0.024** |
| Decisions: season pts ≥ DC at H=3 both seasons | **FAIL** — 2025-26 **+194**; 2024-25 **−82** |
| SPA alone | Not a keep (bakeoff SPA already inconclusive) |

---

## What ran

| Piece | Status |
|---|---|
| H=3 auto-prune (`PRUNE_AUTO_MIN_HORIZON` 5→3) | Merged [#156](https://github.com/e-mcm/fplt/pull/156) |
| EvalSpec `forecast_combine_equal_ab` | Merged [#157](https://github.com/e-mcm/fplt/pull/157) |
| Promote-eval execute 2025-26 GW1–20 | Done — `internal/forecast-combine-promote/confirm-2025-26/` |
| H=3 chip-board optimised, both seasons | Done — GW16 completes in seconds post-prune |

Tip: `main` @ `e4c2f23` (+ #156). Live defaults unchanged: H=3, force-play on, optimised chips, model `dc_layered_v2`.

---

## Forecast board

### Cite — 2024-25 GW1–20 (descriptive bakeoff)

From [forecast-combination-bakeoff.md](./forecast-combination-bakeoff.md); n=13,242; SPA `p_consistent`≈0.31.

| Model | MAE ↓ | RMSE ↓ | Spearman ↑ |
|---|---:|---:|---:|
| `forecast_combine_equal_v1` | 1.0620 | **1.981** | 0.687 |
| `dc_layered_v2` | 1.0645 | 2.006 | 0.670 |
| `lgbm_direct_v1` | 1.0743 | 1.982 | **0.690** |

Equal nudges RMSE/MAE vs DC; LGBM still leads Spearman. Not a keep by itself.

### Confirmation — 2025-26 GW1–20 (`forecast_combine_equal_ab --execute`)

Decision-relevant pool; deltas are **control − treatment** (treatment better when >0 on lower-is-better).

| Model | RMSE ↓ | MAE ↓ | Spearman ↑ | top10 ↑ | n |
|---|---:|---:|---:|---:|---:|
| `dc_layered_v2` | **2.763** | **1.807** | **0.625** | 0.074 | 19105* |
| `forecast_combine_equal_v1` | 2.788 | 1.837 | 0.593 | **0.105** | 6409 |

\*DC row inflated by prior multi-horizon cache rows in the compare summary; paired Δ below is the gate.

| Metric | Δ (ctrl − treat) | Read |
|---|---:|---|
| RMSE (headline) | **−0.024** | treatment **worse** |
| MAE | −0.030 | treatment worse |
| Spearman | −0.032 | treatment worse |
| top10 | +0.031 | treatment better |

Runner verdict stays `inconclusive` (FDB-73 thresholds); plan gate still requires RMSE ≤ control on confirmation → **fail**.

---

## Decision board — H=3 optimised (live horizon)

Post-#156 prune: full-season continuous sims complete (equal GW16 ≈0.7–3s). Chips: optimised, force-play on. LGBM noted as ranking companion only.

### 2025-26 GW1–38

| Rank | Model | Season pts | Mean/GW |
|---:|---|---:|---:|
| 1 | **`forecast_combine_equal_v1`** | **2127** | 56.0 |
| 2 | `lgbm_direct_v1` | 2084 | 54.8 |
| 3 | `dc_layered_v2` | 1933 | 50.9 |

Equal **+194** vs DC — passes this season alone.

### 2024-25 GW1–38

| Rank | Model | Season pts | Mean/GW |
|---:|---|---:|---:|
| 1 | **`dc_layered_v2`** | **2259** | 59.4 |
| 2 | `forecast_combine_equal_v1` | 2177 | 57.3 |
| 3 | `lgbm_direct_v1` | 1918 | 50.5 |

Equal **−82** vs DC — fails “both seasons” gate.

---

## Production

| Knob | Action |
|---|---|
| `PRODUCTION_MODELS` / ops `models` | **No change** — remain `[dc_layered_v2]` |
| `forecast_combine_*` | Stay opt-in bakeoff / compare arms |
| `lgbm_direct_v1` | Ranking companion only (unchanged) |
| H=3 prune | Keep (#156) — unblocks live-horizon MILPs |

---

## Reading

1. **Primary-window RMSE wins do not generalise.** 2024-25 bakeoff favours the blend; 2025-26 confirmation reverses it.
2. **Decision wins are season-unstable at live H.** Equal leads 2025-26 H=3 optimised and loses 2024-25 under the same policy.
3. **Solver fix ≠ promote.** #156 removed the H=3 hang; the ModelSpec still fails the written gates.
4. **Do not flip from H≤2 chips-off or single-season decision wins** (plan + inventory already said this).

---

## Evidence paths

- Promote-eval: `internal/forecast-combine-promote/confirm-2025-26/` (`recommendation.md`, `summary.json`, compare/)
- Decision logs: `internal/forecast-combine-promote/fplt-fc-decisions-2025-26.log`, `…-2024-25.log`
- Backtest log: `internal/forecast-combine-promote/fplt-fc-backtest.log`
- Plan / inventory: [promote-plans/forecast-combine-equal.md](./promote-plans/forecast-combine-equal.md) · [unpromoted-improvements.md](./unpromoted-improvements.md)
