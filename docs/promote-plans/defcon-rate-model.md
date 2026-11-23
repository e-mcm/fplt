# Promote plan: defensive-contribution rate / P(award) model

**Eval status:** **evaluated — keep** (FDB-211).  
**Path:** ModelSpec · primary **RMSE** (2025-26+).  
**Inventory:** [unpromoted-improvements.md](../unpromoted-improvements.md) · [defensive-contribution-eval-2025-26.md](../defensive-contribution-eval-2025-26.md)  
**Report:** [defcon-rate-promote.md](../defcon-rate-promote.md)

| | Control | Treatment |
|---|---|---|
| Spec (at eval) | `dc_layered_v2` mean wire | `dc_layered_defcon_rate_v1` |
| After keep | `dc_layered_defcon_mean_v1` (opt-out) | production `dc_layered_v2` (rate) |

### Artifact / runner
- Component: `defcon_mode: rate` on `other_player` (Poisson actions-per-90 → P(award)).
- `fplt promote-eval --eval defcon_rate_ab --execute` on 2025-26 (+ GW20–38 confirmation).
- Opt-in at eval time; flipped to production; EvalSpec now opt-out vs mean fork.

### Windows
- Primary: 2025-26 GW1–38 walk-forward — **done**
- Confirmation: GW20–38 — **done**

### Gates
- RMSE ≤ control — **pass** (both windows)
- Position DEF/MID RMSE not worse; overall Spearman not worse — **pass**
- Decisions suite optional — skipped

### Flip PR surface
- **Done:** `dc_layered_v2` other → `other_player_defcon_rate_v1` / `defcon_mode: rate`.
- Staging wire stays; mean opt-out = `dc_layered_defcon_mean_v1`.

### Cost / risk
- Medium build. Overfit risk on early 2025-26 GWs mitigated by GW20–38 confirmation.
