# Defcon rate model promote — **keep (provisional)**

**Date:** 2026-11-23 · **Task:** FDB-211  
**Status:** **Provisional keep** — FDB-255 protocol reconfirm **failed** interim RMSE keep ([defcon-rate-confirm.md](./defcon-rate-confirm.md)); production `dc_layered_v3` rate wire unchanged (no silent un-flip).  
**Control (eval):** shrunk 0/2 mean wire (`dc_layered_v2` pre-flip / `dc_layered_defcon_mean_v1` after)  
**Treatment:** Poisson `P(award)|minutes,position,rate` (`dc_layered_defcon_rate_v1` → production `dc_layered_v3` after FDB-226)  
**Plan:** [promote-plans/defcon-rate-model.md](./promote-plans/defcon-rate-model.md)  
**Protocol confirm:** [defcon-rate-confirm.md](./defcon-rate-confirm.md) (FDB-255)  
**Prior:** W9 `tune` wire — Project store `docs/defensive-contribution-eval-2025-26.md` (not in git; see [promote-archives.md](./promote-archives.md))  
**Archive:** Project store `internal/defcon-rate-promote/` (see [promote-archives.md](./promote-archives.md))

---

## Call

**Keep — flip production compose.** *(Provisional — nested windows; FDB-255 did not clear provisional.)*

RMSE headline improves on the primary 2025-26 GW1–38 walk-forward and on the GW20–38 confirmation. MAE and overall Spearman also improve; DEF RMSE/Spearman improve; MID RMSE improves (MID Spearman ≈ flat/noise). Staging + rate path replace the blunt shrunk 0/2 prior inside `dc_layered_v2` (later renamed to `dc_layered_v3`).

Automated promote-eval verdict stays `inconclusive` (improvement thresholds deferred); this written call is the gate.

---

## Arms

| Role | Spec (at eval time) | Other slot |
|---|---|---|
| Control | `dc_layered_v2` (mean wire) | `other_player_v1` — `defcon_mode: mean` (default) |
| Treatment | `dc_layered_defcon_rate_v1` | `other_player_defcon_rate_v1` — `defcon_mode: rate` |

After keep: production `dc_layered_v2` uses rate; opt-out fork `dc_layered_defcon_mean_v1` restores the mean wire. EvalSpec `defcon_rate_ab` retargeted to that opt-out shape.

### Model

Actions-per-90 (staged `defensive_contribution`) shrunk toward the position prior over `window=10` / `shrinkage=5`, then `E[DC pts] = P(Poisson(λ) ≥ threshold) × points` at bucket midpoints 0 / 45 / 75. Bonus / saves / cards / other stay on the mean path.

---

## Windows

| Window | Season | GW | Pool | n / arm |
|---|---|---|---|---|
| **Primary** | 2025-26 | 1–38 | decision-relevant | 12,696 |
| **Confirmation** | 2025-26 | 20–38 | decision-relevant | 19,310 |

```bash
export FPLT_DATA_DIR=/tmp/fplt-defcon-rate-eval
uv run fplt promote-eval --eval defcon_rate_ab --execute \
  --output-dir /tmp/fplt-defcon-rate-ab-primary
uv run fplt promote-eval --eval defcon_rate_ab --execute \
  --from-gw 20 --to-gw 38 \
  --output-dir /tmp/fplt-defcon-rate-ab-confirm
```

HEAD at execute: `193c0cd` (pre-flip). Tip after flip includes compose change.

---

## Headline (RMSE)

Positive delta ⇒ treatment better. Decision-relevant pool (not W9’s full-player pool).

### Primary GW1–38

| Model | RMSE ↓ | MAE ↓ | Spearman ↑ | n |
|---|---:|---:|---:|---:|
| Control (mean) | 2.7280 | 1.7879 | 0.5519 | 12696 |
| Treatment (rate) | **2.7268** | **1.7822** | **0.5520** | 12696 |
| Δ (treat − control, signed better>0) | **+0.00117** | **+0.00572** | **+0.00013** | |

Per-GW RMSE wins: treatment **21** / control 17. MAE wins: treatment **31** / control 7.  
RMSE HAC CI crosses 0; MAE HAC CI is entirely positive.

### Confirmation GW20–38

| Model | RMSE ↓ | MAE ↓ | Spearman ↑ | n |
|---|---:|---:|---:|---:|
| Control (mean) | 2.6929 | 1.7683 | 0.6187 | 19310 |
| Treatment (rate) | **2.6915** | **1.7609** | **0.6192** | 19310 |
| Δ | **+0.00143** | **+0.00744** | **+0.00052** | |

Same directional keep on RMSE / MAE / Spearman.

---

## By position (primary)

| Position | RMSE Δ | MAE Δ | Spearman Δ |
|---|---:|---:|---:|
| DEF | **+0.0028** | **+0.0085** | **+0.0011** |
| MID | **+0.0012** | **+0.0066** | −0.0011 |
| FWD | −0.0022 | −0.0003 | +0.0001 |
| GK | +0.0015 | +0.0006 | ≈0 |

DEF/MID RMSE improve; MID Spearman tick is noise-scale vs overall lift. Confirmation repeats DEF/MID RMSE wins.

---

## Gates (plan)

| Gate | Result |
|---|---|
| RMSE ≤ control (headline) | **Pass** (primary + confirm) |
| DEF/MID bias / RMSE not worse | **Pass** |
| Spearman not worse (overall) | **Pass** |
| Decisions suite | Skipped (forecast keep only) |

---

## Production flip

| Knob | Action |
|---|---|
| `dc_layered_v2` other | → `other_player_defcon_rate_v1` / `defcon_mode: rate` |
| `other_player_v1.yaml` | **Unchanged** (mean default) |
| `dc_layered_defcon_mean_v1` | New opt-out for the old wire |
| `PRODUCTION_MODELS` / ops | Unchanged (`dc_layered_v2`) |

---

## Evidence

- `internal/defcon-rate-promote/primary-*.json|.md|.log`
- `internal/defcon-rate-promote/confirm-*.json|.md|.log`
