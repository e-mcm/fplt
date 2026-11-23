# Promote plan: history rating prior (`dc_layered_history_v1`)

**Eval status:** **evaluated** — written call **tune** (no PRODUCTION flip). See [history-market-prior-promote.md](../history-market-prior-promote.md).  
**Path:** ModelSpec · primary **RMSE** · scoreline RPS secondary.  
**Inventory:** [unpromoted-improvements.md](../unpromoted-improvements.md)

| | Control | Treatment |
|---|---|---|
| Spec | `dc_layered_v2` | `dc_layered_history_v1` (`market_prior_source: history`, strength 1.0) |

### Artifact / runner
- EvalSpec `configs/evals/history_market_prior_ab.yaml` (scaffold PR on `cursor/history-prior-evalspec-e89b`).
- `uv run fplt promote-eval --eval history_market_prior_ab --execute`
- Note: `odds_history_join` coverage metric is **unimplemented** — land a minimal gate **or** run with `min_coverage: null` and audit past-close join in the recommendation (cite repo `docs/feature-eval-pipeline.md` §5.1 / §6).

### Windows
- Primary: 2024-25 GW1–20 (same board as championship).
- Confirmation: 2024-25 GW21–38 or 2025-26 GW1–20 (fresh window; dose ≠ keep).

### Gates
- RMSE ≤ control (headline); RPS/log-loss not worse vs vanilla DC.
- SPA/MCS descriptive; written `keep` only after confirmation window.
- Do **not** treat championship MAE 1.0640 vs 1.0645 as a keep.

### Flip PR surface
- Adopt history score component into `dc_layered_v2` **or** ops `models: [dc_layered_history_v1]`.
- Design / prefs note.

### Cost / risk
- Low–medium (DC refits). Deadline-safe (past closes). Tiny expected lift → easy `drop`/`inconclusive`.

### Ordered steps
1. Add EvalSpec (+ optional `odds_history_join` implementation — separate small PR).
2. Execute primary → recommendation.md.
3. Confirmation window.
4. Flip PR only on keep.
