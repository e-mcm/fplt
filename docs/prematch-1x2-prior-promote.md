# Prematch de-vig 1X2 prior promote (FDB-216 / FDB-219)

**Date:** 2026-11-23  
**Control:** `dc_layered_v2` (unchanged; still production)  
**Treatment:** `dc_layered_1x2_prematch_v1` / `dixon_coles_1x2_prematch_v1`  
**EvalSpec:** `prematch_1x2_prior_ab` (`coverage.read: prematch_scheduled`, min 0.80)  
**Plan ancestry:** [promote-plans/devig-1x2-prior.md](./promote-plans/devig-1x2-prior.md) · stop note [devig-1x2-prior-promote.md](./devig-1x2-prior-promote.md)  
**Archive:** `internal/prematch-1x2-prior/{primary,confirm}/`  
**ModelSpec PR:** [#159](https://github.com/e-mcm/fplt/pull/159) (merged)

---

## Call

**Do not promote — drop.** Do **not** flip `PRODUCTION_MODELS` / ops `models` (remain `[dc_layered_v2]`).

Prematch deadline-PIT 1X2-region blend at full strength still **hurts scoreline** (RPS / log-loss) on both the primary and confirmation windows. A small player-RMSE improvement does not override that: the lever’s job is match-outcome calibration, and that is where it fails — same shape as the closing-oracle `dc_layered_1x2_v1` drop ([dc-odds-prior-eval.md](./dc-odds-prior-eval.md)). Runner auto-verdict stays `inconclusive` (FDB-73 thresholds unset); the written call is **drop**.

---

## Arms

| | Spec | Knob |
|---|---|---|
| Control | `dc_layered_v2` / `dixon_coles_v2` | all market strengths off |
| Treatment | `dc_layered_1x2_prematch_v1` / `dixon_coles_1x2_prematch_v1` | `market_1x2_strength=1.0`, **`use_closing_proxy: false`** |

Not on this board: closing-oracle `dc_layered_1x2_v1` (already drop); CS-only `dc_layered_cs_v1` (already do-not-promote).

---

## Coverage

| Window | Prematch deadline join | Gate |
|---|---:|---|
| Primary 2024-25 GW1–38 | **350 / 380 = 92.1%** | pass (min 0.80) |
| Confirm 2025-26 GW1–38 | **352 / 380 = 92.6%** | pass |

Preflight refused closing proxy on the treatment (as required).

---

## Results

Promote-eval deltas are **control − treatment** (treatment better when &gt; 0 on lower-is-better metrics). Absolute tables below are season-specific.

### Player RMSE (headline, decision-relevant pool)

| Window | `dc_layered_v2` RMSE ↓ | Treatment RMSE ↓ | Δ (ctrl − treat) | GW wins (treat / ctrl) |
|---|---:|---:|---:|---|
| **2024-25** primary | 2.6389 | **2.6358** | **+0.00315** | 24 / 14 |
| **2025-26** confirm | 2.7268 | **2.7221** | **+0.00477** | (pooled board also favors treat) |

Primary HAC on per-GW RMSE Δ: mean ≈ 0.00295, p ≈ 0.039, bootstrap CI excludes 0. Magnitude is small (~0.1% relative).

### Scoreline (lever metric; components, n=380 fixtures)

| Window | Model | RPS ↓ | log-loss ↓ |
|---|---|---:|---:|
| **2024-25** | **`dixon_coles_v2`** | **0.3336** | **1.2353** |
| 2024-25 | `dixon_coles_1x2_prematch_v1` | 0.3456 | 1.2565 |
| 2024-25 | `raw_de_vig_1x2` (same clock) | 0.3575 | 1.2909 |
| **2025-26** | **`dixon_coles_v2`** | **0.3407** | **1.2520** |
| 2025-26 | `dixon_coles_1x2_prematch_v1` | 0.3454 | 1.2564 |
| 2025-26 | `raw_de_vig_1x2` | 0.3491 | 1.2670 |

| Window | RPS Δ | log-loss Δ | Read |
|---|---:|---:|---|
| 2024-25 | **−0.0120** | **−0.0212** | treatment worse |
| 2025-26 | **−0.0047** | **−0.0044** | treatment worse |

Closing-oracle context (same strength, wrong clock): RPS 0.348 / LL 1.265 vs DC 0.334 / 1.235 — [dc-odds-prior-eval.md](./dc-odds-prior-eval.md). Prematch does not rescue the blend.

---

## Production

| Knob | Action |
|---|---|
| `PRODUCTION_MODELS` / ops `models` | **No change** — `[dc_layered_v2]` |
| `dc_layered_1x2_prematch_v1` | Recorded arm; **do not promote** |
| Closing-oracle `*_1x2_v1` | Still **do not promote** |

---

## Commands / provenance

```bash
export FPLT_DATA_DIR=/tmp/fplt-prematch-1x2
# vaastav + football-data E0 (closing + prematch) staged; then:
uv run fplt promote-eval --eval prematch_1x2_prior_ab --execute \
  --output-dir /tmp/fplt-prematch-1x2-eval/primary
uv run fplt promote-eval --eval prematch_1x2_prior_ab --execute \
  --seasons 2025-26 \
  --output-dir /tmp/fplt-prematch-1x2-eval/confirm
```

| Run | `git_sha` | Exit |
|---|---|---|
| Primary | `ef459b0` (#159 merge) | 0 |
| Confirm | `00c58b3` (main tip at run) | 0 |

Artifacts: `internal/prematch-1x2-prior/primary/` · `internal/prematch-1x2-prior/confirm/` (recommendation.md, summary.json, components/, compare/, per_match.parquet, audit.json).
