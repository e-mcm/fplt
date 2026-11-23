# Promote plan: defensive-contribution rate / P(award) model

**Eval status:** **evaluated — keep (provisional)** (FDB-211); **protocol confirm fail** (FDB-255).  
**Path:** ModelSpec · primary **RMSE** (2025-26+).  
**Inventory:** [unpromoted-improvements.md](../unpromoted-improvements.md) · [defensive-contribution-eval-2025-26.md](../defensive-contribution-eval-2025-26.md)  
**Report:** [defcon-rate-promote.md](../defcon-rate-promote.md) · [defcon-rate-confirm.md](../defcon-rate-confirm.md)

| | Control | Treatment |
|---|---|---|
| Spec (at eval) | `dc_layered_v2` mean wire | `dc_layered_defcon_rate_v1` |
| After keep / FDB-226 | `dc_layered_defcon_mean_v1` (opt-out) | production `dc_layered_v3` (rate) |

### Artifact / runner
- Component: `defcon_mode: rate` on `other_player` (Poisson actions-per-90 → P(award)).
- `fplt promote-eval --eval defcon_rate_ab --execute` on 2025-26.
- Opt-in at eval time; flipped to production; EvalSpec now opt-out vs mean fork.

### Windows
- FDB-211 primary GW1–38 + nested confirm GW20–38 — **done** (provisional under protocol)
- FDB-255 development GW1–19 + confirmation GW20–38 — **done; RMSE keep fail**

### Gates
- FDB-211 RMSE ≤ control — pass (nested windows)
- FDB-255 protocol interim keep (disjoint) — **fail** (RMSE worse both halves; MAE confirm win secondary)
- Decisions suite optional — skipped (forecast claim)

### Flip PR surface
- **Done:** production rate via `dc_layered_v3` (FDB-226). Mean opt-out = `dc_layered_defcon_mean_v1`.
- **No silent un-flip** after FDB-255; explicit revert PR only if desired.

### Cost / risk
- Medium build. Protocol reconfirm did not clear provisional.
