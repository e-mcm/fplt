# Stacked production season-points vs points-at-rank bars

**Call: measurement only — no PRODUCTION flip.** Absolute score vs the [points-at-rank north star](./evaluation-north-star.md) on the live decision stack.  
**Task:** FDB-224 (board) · FDB-231 (bar conversion) · **Date:** 2026-11-23  
**Tip:** `35b9ac5` (`main`) · model_id `12152f0e53162d2f` (`dc_layered_v2`)  
**Artifacts:** [`internal/stacked-prod-2400/`](../internal/stacked-prod-2400/)

| Lever | Live default used |
|---|---|
| Horizon | H=3 |
| Force-play | **on** |
| Chips | **optimised** (scheduler → MILP) |
| Lineup reopt | **True** (`bench_weight=0`) |
| Within-horizon price | **`mean_gw_delta`** (`--price-forecast mean_gw_delta`) |
| Sell-on | **on** (as-of 50%; always) |
| Model | production **`dc_layered_v2`** (defcon rate already in; **no** forecast-combine equal blend) |

---

## Season totals vs per-season bars

Anchor: **2400 @ 2024-25** ≡ OR **≈573k** (top ~5%). Per-season bar = points at that rank.

| Season | Points | Season bar | Δ vs bar | Δ vs old 2400 | Mean PPG | Wall |
|---|---:|---:|---:|---:|---:|---:|
| **2025-26** (primary) | **2055** | **2235** | **−180** | −345 | 54.1 | 136s |
| **2024-25** (confirm) | **2239** | **2400** | **−161** | −161 | 58.9 | 99s |

Mean-PPG sanity ≤ 90: **pass** both. Neither season hits its **per-season** bar.

**Mean bar** (dashboard only) = mean(2400, 2235) = **2318**. Do **not** score either season against 2318 or against a fixed 2400 — see [evaluation-north-star.md](./evaluation-north-star.md) §2.

### Chip timelines

| Season | Chips |
|---|---|
| 2025-26 | TC@2, FH@3, WC@4, BB@16, TC@26, WC@31, BB@33, FH@34 |
| 2024-25 | WC@14, BB@25, WC@28, FH@29, TC@33 |

---

## Predictions

| Season | Source |
|---|---|
| 2025-26 | Remapped `h8-shared-data` ledger (`b12aad…` → tip hash `12152f0e…`); horizons 1–8 available, planner H=3 |
| 2024-25 | Fresh `fplt backtest --steps 3` on tip `dc_layered_v2` (`12152f0e…`); wall 485s |

---

## Call

**Gap to the per-season bars remains.** Primary live-rules year is **2055 (−180 vs 2235)**; confirmation **2239 (−161 vs 2400)**. Under the old fixed-2400 read the 2025-26 gap looked −345; points-at-rank shrinks that gap because 2025-26 was a lower-scoring season at the same overall rank. Stack already matches live defaults (force-play, lineup_reopt, `mean_gw_delta`, sell-on, optimised chips, `dc_layered_v2`). Forecast-combine equal stays **dropped** ([forecast-combine-promote.md](./forecast-combine-promote.md)). **Do not** flip `PRODUCTION_MODELS` or planner defaults from this board.

---

## Commands

```bash
# 2025-26 (after remapping shared preds to tip model_id)
uv run fplt decisions --chip-board --season 2025-26 --horizon 3 \
  --models dc_layered_v2 --from-gw 1 --to-gw 38 --chip-arms optimised \
  --price-forecast mean_gw_delta

# 2024-25
uv run fplt backtest --season 2024-25 --models dc_layered_v2 \
  --from-gw 1 --to-gw 38 --steps 3
uv run fplt decisions --chip-board --season 2024-25 --horizon 3 \
  --models dc_layered_v2 --from-gw 1 --to-gw 38 --chip-arms optimised \
  --price-forecast mean_gw_delta
```
