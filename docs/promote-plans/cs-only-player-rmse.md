# Promote plan: CS-only λ — player/RMSE revisit (FDB-239)

**Eval status:** components-only **evaluated** → do not promote; this plan
opens the **prediction** suite under eval-protocol windows.  
**Path:** ModelSpec · primary **RMSE** (CS Brier secondary; RPS sanity).  
**Inventory:** [unpromoted-improvements.md](../unpromoted-improvements.md) ·
[cs-only-market-lambda-eval.md](../cs-only-market-lambda-eval.md)

| | Control | Treatment |
|---|---|---|
| Spec | `dc_layered_v3` | `dc_layered_cs_v1` (`market_cs_strength=0.5`, prematch PIT) |

### Artifact / runner

- EvalSpec: `configs/evals/cs_only_player_rmse_ab.yaml`
  (`suites.prediction: true` + `components: true`).
- Historical components-only board remains
  `configs/evals/cs_only_market_lambda_ab.yaml` (do not mutate).
- `fplt promote-eval --eval cs_only_player_rmse_ab --execute`
  (coverage `prematch_scheduled` ≥ 0.80).

### Windows (eval protocol §1 — declare before looking at numbers)

| Role | Window |
|---|---|
| **Development** | 2024-25 GW1–20 (EvalSpec default) |
| **Confirmation** | 2024-25 GW21–38 (`--from-gw 21 --to-gw 38`) |
| **Optional 2nd confirm** | 2025-26 finished GWs (`--seasons 2025-26`) |
| **Sealed** | Unfinished 2025-26 / 2026-27 GWs |

Do **not** nest GW1–38 primary + GW20–38 confirm. Dose stays **0.5** —
no retune on confirmation.

### Gates (written interim keep §2.2)

All required for written **keep**:

1. Development: treatment better on **RMSE** (decision-relevant pool).
2. Development: CS Brier home and/or away still better (slot metric).
3. Development: RPS not worse (1X2 sanity; descriptive if flat).
4. Confirmation: same directional RMSE + CS Brier on the disjoint window.
5. Coverage / leak preflight pass.

Else written **drop** / **tune** — prior components-only “do not promote”
stands. Runner auto-verdict stays `inconclusive` until FDB-73 thresholds.

### Out of scope on this board

- Prematch / closing **1X2** region blend (`*_1x2_*`) — **dropped**;
  revisit only with new evidence, not via this EvalSpec.
- Full-strength λ prior (`dc_layered_odds_v1`) — closed.
- `forecast_combine_equal_v1` — dropped.
- `PRODUCTION_MODELS` / ops flip — **only** after interim keep + confirm.

### Flip PR surface (after keep only)

- Compose `market_cs_strength` into a new production ModelSpec bump
  (`dc_layered_v4+`); do not edit `dc_layered_v3` in place.
- Update ops `models` / prefs / design in the same flip PR.

### Ordered steps

1. Land EvalSpec + tests (this PR) — measurement plumbing.
2. Run development promote-eval; archive artifacts.
3. If RMSE + CS Brier move the right way, run confirmation (disjoint).
4. Written call under eval protocol §2; flip PR only on keep.
