# CS-only player RMSE — **keep** (PRODUCTION flipped to `dc_layered_v4`)

**Date:** 2026-11-23 · **Task:** FDB-262 / FDB-239  
**Status:** Written **keep** under [`eval-protocol.md`](./eval-protocol.md) §1–§2.  
**Production:** `dc_layered_v4` (`market_cs_strength=0.5`) after FDB-267. Frozen pre-CS: `dc_layered_v3`. Opt-out: `dc_layered_nocs_v1`.  
**EvalSpec:** `cs_only_player_rmse_ab` · tip `98d06a2` (post-#187)  
**Plan:** [promote-plans/cs-only-player-rmse.md](./promote-plans/cs-only-player-rmse.md)  
**Prior components board:** [cs-only-market-lambda-eval.md](./cs-only-market-lambda-eval.md) (do-not-promote without player/RMSE)  
**Artifacts:** `internal/cs-only-player-rmse-promote/` · repo `configs/evals/results/cs_only_player_rmse/`

---

## Call

**Keep** — compose `market_cs_strength=0.5` into a new production ModelSpec bump when the flip PR lands.

- Headline **RMSE** better on development **and** disjoint confirmation (decision-relevant pool).
- Slot **CS Brier** (home and away) better on both windows.
- **RPS** ≈ flat (not worse) on both windows.
- Coverage `prematch_scheduled` ≥ 0.80 on both windows.
- Dose fixed at **0.5** (no retune on confirmation).
- Runner auto-verdict stays **`inconclusive`** (FDB-73); this written call is the gate.
- Flip landed in FDB-267 (`dc_layered_v4`); this record's keep arms stay `v3` vs `cs_v1`.

---

## Arms

| Role | Spec | Knob |
|---|---|---|
| Control | `dc_layered_v3` (`30367a901721a5d6`) | `market_cs_strength=0` |
| Treatment | `dc_layered_cs_v1` (`b16a511a480735f5`) | `market_cs_strength=0.5`, prematch PIT |

---

## Windows (declared before looking at confirmation)

| Role | Season | GW | Notes |
|---|---|---|---|
| **Development** | 2024-25 | 1–20 | EvalSpec default; SPA/MCS here descriptive only |
| **Confirmation** | 2024-25 | 21–38 | Disjoint; clean ledger (see caveat) |
| **Sealed** | unfinished 2025-26 / 2026-27 | — | Not used |

```bash
export FPLT_DATA_DIR=/tmp/fplt-cs-player-rmse
# vaastav + football-data E0 dual odds_kind staged as in cs-only-market-lambda-eval.md

uv run fplt promote-eval --eval cs_only_player_rmse_ab --execute \
  --output-dir /tmp/fplt-cs-player-rmse/dev

# Confirmation MUST use a fresh data_dir (empty ledger / ledger_cache).
# Reusing the development data_dir contaminates prediction summary + HAC
# with GW1–20 folds via the shared prediction ledger.
export FPLT_DATA_DIR=/tmp/fplt-cs-player-rmse-confirm-clean  # staged via symlink OK
uv run fplt promote-eval --eval cs_only_player_rmse_ab --execute \
  --from-gw 21 --to-gw 38 \
  --output-dir /tmp/fplt-cs-player-rmse/confirm-clean
```

---

## Numbers

Positive prediction/component deltas ⇒ treatment better (control − treatment on lower-is-better metrics).

### Development — 2024-25 GW1–20

| | Control | Treatment | Δ |
|---|---:|---:|---:|
| RMSE ↓ | 2.713905 | **2.712207** | **+0.001698** |
| MAE ↓ | 1.758362 | **1.756378** | **+0.001983** |
| n (decision pool) | 6162 | 6162 | |
| CS Brier home ↓ | — | — | **+0.004897** |
| CS Brier away ↓ | — | — | **+0.006148** |
| RPS ↓ | — | — | **+0.000003** (≈ flat) |

Coverage: prematch_scheduled **185/199 = 93.0%** (pass).  
Per-GW RMSE wins: treatment **13** / control 7.  
RMSE HAC CI crosses 0 (p≈0.083) — descriptive only on development.

### Confirmation — 2024-25 GW21–38 (clean ledger)

| | Control | Treatment | Δ |
|---|---:|---:|---:|
| RMSE ↓ | 2.601718 | **2.599584** | **+0.002133** |
| MAE ↓ | — | — | **+0.001168** |
| n (decision pool) | 5958 | 5958 | |
| CS Brier home ↓ | — | — | **+0.001727** |
| CS Brier away ↓ | — | — | **+0.004882** |
| RPS ↓ | — | — | **+0.00000018** (≈ flat) |

Coverage: prematch_scheduled **165/181 = 91.2%** (pass).  
Per-GW RMSE wins: treatment **12** / control 6.  
RMSE HAC CI entirely positive: **[0.00056, 0.00387]** (p≈0.0086).  
Audit folds: **18** (GW21–38 only).

---

## Ledger-cache caveat (operator)

First confirmation attempt reused `$FPLT_DATA_DIR` from development. The shared `ledger/predictions` + content-addressed `ledger_cache` caused the official confirm **summary / per-GW wins / HAC `n_gw`** to reflect **GW1–38** (n=12120, wins 14–24) even though walk-forward only fitted GW21–38. Components + coverage were still windowed. Contaminated artifacts kept under `internal/cs-only-player-rmse-promote/confirm-contaminated-ledger/` for forensics. **Authoritative confirmation = `confirm/` (clean ledger).** Follow-up: promote-eval should refuse or slice shared-ledger compare to the requested GW window.

---

## Flip surface

1. ~~New compose ModelSpec~~ → `dc_layered_v4` with `market_cs_strength=0.5` (FDB-267).
2. ~~Retarget ops / `PRODUCTION_MODELS`~~ → done.
3. ~~Registry / design / prefs~~ → done in the flip PR.
4. Optional second confirm on finished 2025-26 GWs remains available.

---

## Registry

Committed as `cs_only_player_rmse` in `configs/evals/results/registry.yaml` with audits under `results/cs_only_player_rmse/{development,confirm}/`.
