# Promote result: within-horizon list-price forecast (`mean_gw_delta`)

**Call: keep** — treatment ≥ control on both seasons.  
**Status:** **Confirmed** under eval protocol §2.3 (FDB-256 — [price-forecast-confirm.md](./price-forecast-confirm.md)).  
**Plan:** [promote-plans/price-within-horizon.md](./promote-plans/price-within-horizon.md) · **Design:** [price-within-horizon-design.md](./price-within-horizon-design.md)  
**Artifacts:** Project store `internal/price-within-horizon/` (see [promote-archives.md](./promote-archives.md)) · tip `ee9c980` (+ `#163` board flag)  
**FDB-205:** canceled (as-of sell-on = bug fix #124, not a gate).

| | Control | Treatment |
|---|---|---|
| Policy | As-of sell-on; **flat** within-horizon (`price_forecast=None`) | `mean_gw_delta` PIT list-price forecast → `plan_squad` |
| CLI | omit `--price-forecast` | `--price-forecast mean_gw_delta` |
| Chips / H / force | optimised · H=3 · force_play **on** | same |
| Model | `dc_layered_v2` (shared preds) | same |

**PIT:** forecasts use only `available_at ≤ deadline` via `pit_deadline_prices` / `PITView`; step 0 locked to as-of candidate prices.

---

## Boards (season points)

| Season | Treatment | Control | Δ | Mean-PPG (T / C) | Wall (T / C) |
|---|---:|---:|---:|---|---|
| **2025-26** (primary) | **2002** | 1973 | **+29** | 52.7 / 51.9 | 128s / 114s |
| **2024-25** (confirm) | **2253** | 2244 | **+9** | 59.3 / 59.1 | 95s / 99s |

Mean-PPG sanity ≤ 90: **pass** both arms.

2025-26 preds: remapped `h8-shared-data` ledger (`b12aad…` → tip hash `12152f0e…` after defcon YAML change); same frame both arms.  
2024-25 preds: fresh `fplt backtest --steps 3` on tip `dc_layered_v2` (`12152f0e…`).

### Chip timelines

| Season | Arm | Chips |
|---|---|---|
| 2025-26 | control | WC@2, FH@3, BB@16, TC@19, TC@26, WC@30, FH@31, BB@33 |
| 2025-26 | treatment | WC@2, FH@3, TC@13, BB@19, TC@26, WC@31, BB@33, FH@34 |
| 2024-25 | control | WC@14, TC@24, BB@25, WC@28, FH@29 |
| 2024-25 | treatment | WC@14, BB@25, WC@28, TC@33, FH@34 |

Forecast changes chip packing and transfer timing; both seasons still clear the season-points bar.

---

## Call

**Keep (confirmed FDB-256).** Treatment never loses and wins +29 / +9. Flip live `run_plan` (and design §7/§10) to build `mean_gw_delta` forecasts by default; leave board ablation via omitting `--price-forecast` / `flat`. Flip PR: [#165](https://github.com/e-mcm/fplt/pull/165) merged at `35b9ac5`. Protocol confirmation: [price-forecast-confirm.md](./price-forecast-confirm.md).

**Out of scope / not claimed:** transfer-volume threshold model; mid-horizon buy→sell sell-on for non-owned; N6 sim-SE (not needed — deltas not tiny).

---

## Commands

```bash
# Control
uv run fplt decisions --chip-board --season $SEASON --horizon 3 \
  --models dc_layered_v2 --from-gw 1 --to-gw 38 --chip-arms optimised

# Treatment
uv run fplt decisions --chip-board --season $SEASON --horizon 3 \
  --models dc_layered_v2 --from-gw 1 --to-gw 38 --chip-arms optimised \
  --price-forecast mean_gw_delta
```
