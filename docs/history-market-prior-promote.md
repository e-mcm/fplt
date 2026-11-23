# History rating prior promote — tune (no PRODUCTION flip)

**Date:** 2026-11-23 · **Task:** FDB-213  
**Control:** `dc_layered_v2` (unchanged; still production)  
**Treatment:** `dc_layered_history_v1` (`market_prior_source: history`, strength **1.0**)  
**EvalSpec:** `configs/evals/history_market_prior_ab.yaml` (#155 @ `ed84821`)  
**Plan:** [promote-plans/history-market-prior.md](./promote-plans/history-market-prior.md)  
**Archive:** `internal/history-market-prior/`

---

## Call

**Tune — do not promote strength 1.0; do not flip `PRODUCTION_MODELS`.**

Player RMSE improves on both the primary and confirmation windows, but scoreline **RPS is slightly worse than vanilla DC on both**. Plan gate “RPS/logloss not worse vs vanilla DC” fails at this dose. Runner stays **inconclusive** (FDB-73 thresholds unset). Championship MAE 1.0640 vs 1.0645 remains **not** a keep.

Next lever if pursued: a **lower-strength dose** (e.g. 0.25 / 0.5 via `extends`) on a window that did not pick the dose — not a keep at 1.0.

---

## Boards

| Window | Suite | Control RMSE | Treatment RMSE | Δ RMSE (ctrl−treat, >0 better) | RPS Δ | log-loss Δ |
|---|---|---:|---:|---:|---:|---:|
| **Primary** 2024-25 GW1–20 | prediction + components | 2.6915 | 2.6876 | **+0.0040** | **−0.00109** | −0.00054 |
| **Confirm** 2024-25 GW21–38 | prediction + components | 2.5828 | 2.5760 | **+0.0068** | **−0.00031** | +0.00034 |

Deltas are control − treatment for lower-is-better metrics (treatment better when Δ > 0).

### Primary (GW1–20, n_pool=6162, n_match=199)

| Metric | Control | Treatment | Read |
|---|---:|---:|---|
| Player RMSE ↓ | 2.6915 | **2.6876** | treatment better; HAC p≈0.046; wins 13–7 |
| Player MAE ↓ | 1.7651 | **1.7408** | treatment better (secondary) |
| Scoreline RPS ↓ | **0.3421** | 0.3432 | control better |
| Scoreline log-loss ↓ | **1.2744** | 1.2749 | control better |
| Brier CS home/away ↓ | — | — | treatment better (Δ +0.0012 / +0.0020) |

### Confirmation (GW21–38, n_pool=5958, n_match=181)

Re-run after clearing `ledger/predictions` so GW1–20 rows did not leak into intervals (first confirm pass had contaminated n_gw=38).

| Metric | Control | Treatment | Read |
|---|---:|---:|---|
| Player RMSE ↓ | 2.5828 | **2.5760** | treatment better; HAC p≈0.050; wins 13–5 |
| Player MAE ↓ | 1.6610 | **1.6390** | treatment better |
| Scoreline RPS ↓ | **0.3243** | 0.3246 | control better (tiny) |
| Scoreline log-loss ↓ | 1.1923 | **1.1919** | treatment better (tiny) |

---

## Coverage / past-close join

`odds_history_join` is still **unimplemented**; EvalSpec ran with `min_coverage: null`. Manual past-close audit (closing_proxy 1X2 joined by season+teams+kickoff, visible at each GW1–20 deadline):

| Check | Rate |
|---|---:|
| Mean per-deadline history-pool join (GW1–20) | **99.7%** |
| Min per-deadline rate | 99.7% |
| Past closes at GW20 deadline | 2650 / 2658 (**99.7%**) |
| Train seasons 2021-22…2024-25 fixture join | 99.7% |

Deadline-safe past closes are not the blocker. Artifact: `internal/history-market-prior/past_close_join_audit.json`.

---

## Production

| Knob | Action |
|---|---|
| `PRODUCTION_MODELS` / ops `models` | **No change** — remain `[dc_layered_v2]` |
| `dc_layered_history_v1` @ strength 1.0 | **Do not compose into v2** |
| Optional follow-up | Dose grid / `extends` forks — new EvalSpec, not this keep |

---

## Commands

```bash
export FPLT_DATA_DIR=/tmp/fplt-history-prior-data
# vaastav + football-data E0 staged (match_odds closing_proxy + prematch_scheduled)
uv run fplt promote-eval --eval history_market_prior_ab --execute \
  --output-dir /tmp/fplt-history-prior-primary
# clear ledger/predictions between windows (intervals read the store ledger)
rm -rf "$FPLT_DATA_DIR/ledger/predictions"
uv run fplt promote-eval --eval history_market_prior_ab --execute \
  --from-gw 21 --to-gw 38 --output-dir /tmp/fplt-history-prior-confirm
```

Tip: `main` @ `ed84821` (#155 EvalSpec).
