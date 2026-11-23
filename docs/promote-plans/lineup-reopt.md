# Promote plan: `lineup_reopt` (Scenario XI / EV)

**Task:** FDB-206 · **Path:** Decision-policy · **Pipeline:** [promote-pipeline-unified.md](../promote-pipeline-unified.md)  
**Context:** [scenario-xi-bench-d1.md](../scenario-xi-bench-d1.md) · inventory [unpromoted-improvements.md](../unpromoted-improvements.md)  
**Supersedes draft:** [lineup-reopt-season-points.md](./lineup-reopt-season-points.md) (same lever; this is the executable plan).

Default already flipped in [#144](https://github.com/e-mcm/fplt/pull/144) (`lineup_reopt=True`, `bench_weight=0`) **without** a season-points promote gate. This plan boards treatment vs MILP-only control, then writes keep / drop / tune. Flip PR only after keep (or document that default already matches keep).

---

## 1. Control vs treatment

| | **Control** | **Treatment (live default)** |
|---|---|---|
| Lever | `plan_squad(..., lineup_reopt=False)` | `lineup_reopt=True` (EV reopt; optional `scenarios=` SAA) |
| CLI | `fplt decisions --chip-board … --no-lineup-reopt` | omit flag (default) |
| Forecasts | `dc_layered_v2` | same |
| Chips / H / force | optimised · **H=3** · force_play **on** | same |
| `bench_weight` | 0 (YAML) | 0 |

**Live cycle already uses treatment:** `run_plan` → `plan_squad` without overriding `lineup_reopt` → default `True`. Prior force×H boards at tip `28916d7+` logged `lineup_reopt=True`. This promote is the missing season-points gate, not a new default flip.

One lever only — do not co-vary horizon, force_play, chip arm, or model.

---

## 2. Artifact / runner

Decision board (not EvalSpec / `promote-eval`):

```bash
# Treatment (default reopt)
uv run fplt decisions --chip-board --season $SEASON --horizon 3 \
  --models dc_layered_v2 --from-gw 1 --to-gw 38 --chip-arms optimised

# Control (raw MILP XI / captain)
uv run fplt decisions --chip-board --season $SEASON --horizon 3 \
  --models dc_layered_v2 --from-gw 1 --to-gw 38 --chip-arms optimised \
  --no-lineup-reopt
```

Separate `$FPLT_DATA_DIR` per arm (both write `::chip_optimised`). Archive ledgers under `internal/lineup-reopt-promote/`. Optional SAA arm (`scenarios=`) is out of scope for this gate — EV path only.

---

## 3. Windows

| # | Window | Role |
|---|---|---|
| **A** | **2025-26** GW1–38 H=3 opt force_play=on | Primary |
| **B** | **2024-25** GW1–38 same shape | Confirmation |

Predictions: shared staged package + season preds covering H≥3 (`--steps 3` sufficient; reuse `h8-shared-data` preds for 2025-26).

---

## 4. Gates / call

- Primary metric: **season points** (north star §2).
- Mean-PPG sanity ≤ 90.
- **Keep** if treatment ≥ control on **both** seasons (ties OK — default already on).
- **Drop** if treatment loses either season → flip PR sets `lineup_reopt=False` + prefs/design.
- **Tune** only if mixed seasons with a clear EV vs SAA / captaincy follow-up (do not flip).
- **No multi-hour N6** unless needed; optional cheap `sim-season` smoke only if claiming uncertainty.

---

## 5. Flip PR surface

| Call | Action |
|---|---|
| **Keep** | Document in `docs/lineup-reopt-promote.md`; **no code flip** (default already True). Update inventory eval status. |
| **Drop** | PR: `plan_squad` / live default `lineup_reopt=False`; prefs + design + `scenario-xi-bench-d1.md`. |
| **Tune** | Written follow-up; leave default; no flip. |

---

## 6. Ordered steps

1. Confirm control/treatment + live already on treatment (this doc §1).
2. Wire `--no-lineup-reopt` through `simulate_decisions` → `run_decisions` → `run_chip_decision_board` → CLI (tests).
3. Run **2025-26** H2H → archive under `internal/lineup-reopt-promote/2025-26/`.
4. Run **2024-25** confirmation → archive.
5. Write `docs/lineup-reopt-promote.md` with keep/drop/tune + flip note.
6. Flip PR only on drop; on keep, document no further flip needed.
