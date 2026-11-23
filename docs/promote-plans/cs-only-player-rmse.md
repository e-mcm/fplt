# Promote plan: CS-only λ — optional player/RMSE revisit

**Eval status:** **evaluated** components-only → do not promote. Optional follow-up only.  
**Path:** ModelSpec · primary **RMSE**.  
**Inventory:** [unpromoted-improvements.md](../unpromoted-improvements.md) · [cs-only-market-lambda-eval.md](../cs-only-market-lambda-eval.md)

| | Control | Treatment |
|---|---|---|
| Spec | `dc_layered_v2` | `dc_layered_cs_v1` (`market_cs_strength=0.5`, prematch PIT) |

### Artifact / runner
- Extend / clone `configs/evals/cs_only_market_lambda_ab.yaml` with `suites.prediction: true`.
- `fplt promote-eval --eval … --execute` (coverage `prematch_scheduled` ≥0.80).

### Windows
- Primary: 2024-25 GW1–38 (already have components).
- Confirmation: 2025-26 if keep signal appears.

### Gates
- RMSE ≤ control + CS Brier still better; RPS not worse.
- Else **skip** — prior call stands.

### Flip PR surface
- `PRODUCTION_MODELS` / compose CS strength into v2 — only after keep.

### Cost / risk
- Low. Prior call already negative on promote without player board.

### Ordered steps
1. Only if prioritizing market CS again.
2. Prediction suite → written call.
3. Else leave fork labelled; no flip.
