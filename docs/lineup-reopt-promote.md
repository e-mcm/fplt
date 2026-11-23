# Promote result: `lineup_reopt` (Scenario XI / EV)

**Call: keep** — treatment ≥ control on both seasons; **default already `True`** ([#144](https://github.com/e-mcm/fplt/pull/144)); **no further flip PR**.  
**Plan:** [promote-plans/lineup-reopt.md](./promote-plans/lineup-reopt.md) · **Pipeline:** [promote-pipeline-unified.md](./promote-pipeline-unified.md)  
**Artifacts:** Project store `internal/lineup-reopt-promote/` (see [promote-archives.md](./promote-archives.md)) · board tip `af1d32b` · tooling merge [`91173d7`](https://github.com/e-mcm/fplt/commit/91173d7b5f197c90c3b98dd1a1c0eb1a64df73e8) ([#162](https://github.com/e-mcm/fplt/pull/162))

| | Control | Treatment (live default) |
|---|---|---|
| Policy | `lineup_reopt=False` (raw MILP XI/captain) | `lineup_reopt=True` (EV reopt) |
| CLI | `--no-lineup-reopt` | omit |
| Chips / H / force | optimised · H=3 · force_play **on** | same |
| Model | `dc_layered_v2` | same |

**Live cycle already on treatment:** `run_plan` → `plan_squad` default `True`. Prior force×H boards logged `lineup_reopt=True`.

---

## Boards (season points)

| Season | Treatment | Control | Δ | Mean-PPG (T / C) | Wall (T / C) |
|---|---:|---:|---:|---|---|
| **2025-26** (primary) | **2019** | 1889 | **+130** | 53.1 / 49.7 | 562s / 322s |
| **2024-25** (confirm) | **2283** | 2283 | **0** | 60.1 / 60.1 | 238s / 234s |

Mean-PPG sanity ≤ 90: **pass** both arms. Same forecasts both arms within each season (2025-26: shared `h8-shared-data` preds; 2024-25: fresh backtest `--steps 3`).

### Chip timelines

| Season | Arm | Chips |
|---|---|---|
| 2025-26 | treatment | TC@2, WC@3, FH@6, BB@19, BB@26, WC@33, FH@34, TC@38 |
| 2025-26 | control | FH@2, WC@3, BB@16, TC@19, BB@26, WC@30, FH@34, TC@38 |
| 2024-25 | both | WC@14, TC@24, BB@25, WC@28, FH@34 |

2025-26: reopt changes early chip packing and every GW’s squad/XI path (+130).  
2024-25: chip timeline identical; per-GW points identical; XI/squad/cap differed only GW **25** and **29** (no season-points delta).

---

## Call

**Keep.** Treatment is never worse and wins +130 on the live-rules year. Leave `plan_squad(..., lineup_reopt=True)` and `bench_weight: 0` as production defaults. Document only — **do not** open a default-flip PR.

**Out of scope / not claimed:** SAA (`scenarios=`) vs EV; multi-hour N6 SE (not needed for this keep; smoke GW1–2 only verified the A/B flag).

**Tooling landed for the board:** `fplt decisions --chip-board --no-lineup-reopt` (ablation mirror of `--no-force-play`).

---

## Commands

```bash
# Treatment
uv run fplt decisions --chip-board --season $SEASON --horizon 3 \
  --models dc_layered_v2 --from-gw 1 --to-gw 38 --chip-arms optimised

# Control
uv run fplt decisions --chip-board --season $SEASON --horizon 3 \
  --models dc_layered_v2 --from-gw 1 --to-gw 38 --chip-arms optimised \
  --no-lineup-reopt
```
