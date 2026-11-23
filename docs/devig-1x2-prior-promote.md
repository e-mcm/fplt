# De-vig 1X2 prior promote — stop (no runnable prematch arm)

**Date:** 2026-11-23 · **Task:** FDB-207  
**Control:** `dc_layered_v2` (unchanged; still production)  
**Plan:** [promote-plans/devig-1x2-prior.md](./promote-plans/devig-1x2-prior.md)  
**Archive:** `internal/devig-1x2-prior/`

---

## Call

**Drop / stop — do not run promote-eval; do not flip `PRODUCTION_MODELS`.**

No deadline-PIT / `prematch_scheduled` de-vig 1X2 **treatment ModelSpec** exists that has never cleared promote-eval. The shipped redesign after W7 is a **closing-oracle** 1X2-region blend already rejected on scoreline. The only prematch market fork (`dc_layered_cs_v1`) is CS-only λ and already called do-not-promote.

---

## What was checked

### Models (`configs/models/`)

| Spec | Lever | Clock | Verdict on file |
|---|---|---|---|
| `dc_layered_1x2_v1` | `market_1x2_strength=1.0`, `use_closing_proxy: true` | Closing oracle | **Drop** — RPS 0.348 / LL 1.265 vs DC 0.334 / 1.235 (2024-25 GW1–38, n=380). [dc-odds-prior-eval.md](./dc-odds-prior-eval.md) · [unpromoted-improvements.md](./unpromoted-improvements.md) |
| `odds_1x2_v1` | Raw proportional de-vig | Closing oracle | Probe / score baseline only |
| `dc_layered_odds_v1` | `market_prior_strength=1.0` | Closing upper / strict no-op | **Drop** (W7) — [promo-evals-consolidated.md](./promo-evals-consolidated.md) |
| `dc_layered_cs_v1` | `market_cs_strength=0.5`, proxy **off** | Prematch PIT | **Do not promote** — [cs-only-market-lambda-eval.md](./cs-only-market-lambda-eval.md) |

### Evals (`configs/evals/`)

| EvalSpec | Treatment | Coverage read | Status |
|---|---|---|---|
| `cs_only_market_lambda_ab.yaml` | `dc_layered_cs_v1` | `prematch_scheduled` | Ran; do not promote |
| `example_data_odds_prior.yaml` | `dc_layered_odds_v1` | `strict_pit` | Scaffold / W7 shape; not a 1X2 prematch board |
| *(none)* | `dc_layered_1x2_v1` | — | No EvalSpec; evaluated via `eval-components` on closing |

### Gap that would unlock a real board

A new layered fork with `market_1x2_strength > 0` and **`use_closing_proxy: false`**, plus an EvalSpec gated on `coverage.read: prematch_scheduled` (same preflight pattern as CS-only). That is **redesign / ModelSpec work**, not this promote pass.

---

## Production

| Knob | Action |
|---|---|
| `PRODUCTION_MODELS` / ops `models` | **No change** — remain `[dc_layered_v2]` |
| Closing-oracle arms (`*_1x2_v1`, `*_odds_v1`, `odds_1x2_v1`) | **Do not promote** |
| Prematch CS-only (`dc_layered_cs_v1`) | Already not flipped; optional RMSE revisit is a separate plan |

---

## Evidence pointers (no new board artifacts)

- 1X2 blend drop numbers: [dc-odds-prior-eval.md](./dc-odds-prior-eval.md) (corrected board; ignore void RPS 0.196 elsewhere)
- Five-season closing context: [dc-vs-bookie-1x2-full.md](./dc-vs-bookie-1x2-full.md)
- Prematch CS-only: [cs-only-market-lambda-eval.md](./cs-only-market-lambda-eval.md); archive `internal/cs-only-market-lambda-eval/`
- Inventory row: [unpromoted-improvements.md](./unpromoted-improvements.md) — “1X2-region blend … skip (closed drop)”
