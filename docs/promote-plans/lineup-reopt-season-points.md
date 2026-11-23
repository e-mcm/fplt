# Promote plan: lineup_reopt (EV / SAA) season points

**Superseded by:** [lineup-reopt.md](./lineup-reopt.md) · **Result:** [lineup-reopt-promote.md](../lineup-reopt-promote.md) (**keep**).

**Eval status:** **evaluated — keep** (FDB-206). Default already `True` in `#144`; no further flip.  
**Path:** Decision-policy · primary **season points**.  
**Inventory:** [unpromoted-improvements.md](../unpromoted-improvements.md) · [scenario-xi-bench-d1.md](../scenario-xi-bench-d1.md)

| | Control | Treatment |
|---|---|---|
| Policy | `plan_squad(..., lineup_reopt=False)` (MILP XI/captain only) | `lineup_reopt=True` (live default; EV; optional `scenarios=` SAA) |
| Forecasts | `dc_layered_v2` | same |
| Chips / H | optimised · H=3 · force_play on | same |

### Artifact / runner
- Decision board: `--chip-board` ± `--no-lineup-reopt` (wired in FDB-206).
- Optional second arm: EV reopt vs SAA (`scenarios=`) — still open.

### Windows / gates
- Primary 2025-26 + confirmation 2024-25; treatment ≥ control both → **pass** (+130 / 0).

### Flip PR surface
- Keep → **document only** (default already on) — done.
