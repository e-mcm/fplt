# Promote plan: de-vig 1X2 market prior (deadline-PIT / prematch)

**Task:** FDB-207 · post-W7 redesign (raw de-vig 1X2 / 1X2-region blend; related #82).  
**Path:** ModelSpec → `fplt promote-eval` · control `dc_layered_v2`.  
**Pipeline:** [promote-pipeline-unified.md](../promote-pipeline-unified.md) · [promo-evals-consolidated.md](../promo-evals-consolidated.md) · repo `docs/feature-eval-pipeline.md`.

---

## Inventory (configs/models + configs/evals)

| Spec | Knob | PIT / odds_kind | Eval status | Notes |
|---|---|---|---|---|
| `dc_layered_odds_v1` / `dixon_coles_odds_v1` | `market_prior_strength=1.0` | Strict PIT → **0%** same-season closes; upper bound = closing rewrite | **Drop** (W7) | Closing λ prior; not deadline-safe. EvalSpec scaffold `example_data_odds_prior.yaml` only. |
| `dc_layered_1x2_v1` / `dixon_coles_1x2_v1` | `market_1x2_strength=1.0` | **`use_closing_proxy: true`** (closing oracle) | **Drop** | De-vig 1X2 *region blend* redesign after W7. Scoreline RPS 0.348 vs DC 0.334. No dedicated EvalSpec; board in [dc-odds-prior-eval.md](../dc-odds-prior-eval.md). |
| `odds_1x2_v1` | raw proportional de-vig | **`use_closing_proxy: true`** | Probe only | Score component, not a layered PointsPredictor treatment. |
| `dc_layered_cs_v1` / `dixon_coles_cs_v1` | `market_cs_strength=0.5` | **`use_closing_proxy: false`** + EvalSpec `coverage.read: prematch_scheduled` | **Do not promote** | Only prematch market fork shipped. Components-only; CS Brier↑, RPS flat. EvalSpec `cs_only_market_lambda_ab.yaml`. |
| `dc_layered_history_v1` | `market_prior_source: history` | Past closes (deadline-visible) | Descriptive / other task | Not a de-vig 1X2 upcoming-fixture prior. |

**Prematch `market_1x2` fork:** **none.** No ModelSpec pairs `market_1x2_strength > 0` with `use_closing_proxy: false`. DC code defaults `use_closing_proxy` to **true** when `market_1x2_strength > 0` unless overridden — the registered 1X2 layered fork explicitly sets the closing escape hatch.

---

## Runnable?

**No.** There is no unevaluated deadline-PIT / `prematch_scheduled` de-vig 1X2 treatment ModelSpec to put on a promote-eval board.

| Candidate | Why not run |
|---|---|
| `dc_layered_1x2_v1` | Closing-oracle arm; already **drop** on scoreline vs `dixon_coles_v2`. Promoting / re-boarding it would be a fake deadline board. |
| `dc_layered_odds_v1` | W7 closed drop; closing ≠ deadline. |
| `dc_layered_cs_v1` | Prematch, but **CS-only λ** (not 1X2 de-vig redesign); already evaluated do-not-promote. |
| Invent a prematch 1X2 YAML + EvalSpec here | Out of scope for this promote task — that is a **new ModelSpec PR**, then a fresh promote plan. |

---

## If a prematch 1X2 fork is built later

Skeleton only (do not execute until the ModelSpec lands):

| | Control | Treatment |
|---|---|---|
| Spec | `dc_layered_v2` | New fork: `market_1x2_strength` > 0, **`use_closing_proxy: false`** |
| EvalSpec | New YAML | `coverage.read: prematch_scheduled`, `min_coverage: 0.80`, refuse closing proxy |
| Primary | 2024-25 walk-forward | Player **RMSE** headline + scoreline RPS/LL |
| Confirm | Second window / 2025-26 | Dose ≠ keep |
| Flip | Only after written **keep** | Never flip closing-oracle arms |

---

## This plan's action

**Stop.** Write the promote result as inventory + closed call; no `fplt promote-eval`; no `PRODUCTION_MODELS` change. Archive: `internal/devig-1x2-prior/`. Result: [devig-1x2-prior-promote.md](../devig-1x2-prior-promote.md).
