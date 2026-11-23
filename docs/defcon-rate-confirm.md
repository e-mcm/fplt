# Defcon-rate provisional keep — protocol confirmation (FDB-255)

**Date:** 2026-11-23 · **Task:** FDB-255  
**Protocol:** [`eval-protocol.md`](./eval-protocol.md) §1–§2  
**Prior keep:** [`defcon-rate-promote.md`](./defcon-rate-promote.md) (FDB-211, nested windows → provisional)  
**Production (at confirm):** `dc_layered_v3` (rate wire; FDB-226). Later FDB-267 moved live compose to `dc_layered_v4` (CS on; rate unchanged).  
**Archive:** `internal/defcon-rate-confirm/`

---

## Verdict

**Confirm fail — remain provisional. Do not clear provisional. Do not un-flip production.**

Under the interim keep rule (§2.2), rate does **not** beat mean on the RMSE headline on either the declared **development** window (2025-26 GW1–19) or the **non-overlapping confirmation** window (GW20–38). Coverage passes. Forecast claim only (`decisions: false`); reference-board / deterministic HiGHS / decisions-suite gates apply to season-points claims, not this RMSE board.

Production stays `dc_layered_v3` / `defcon_mode: rate` until an **explicit** flip PR — protocol forbids silent un-flip of provisional keeps (§2.4).

---

## Windows (declared before readout)

Live-only DC awards → 2025-26 only; split inside the season without GW overlap (§1).

| Role | Season | GW | Notes |
|---|---|---|---|
| **Development** | 2025-26 | 1–19 | Default map; dose not re-selected here |
| **Confirmation** | 2025-26 | 20–38 | Disjoint from development; finished / `data_checked` |
| Sealed | — | — | None left in-season (full season finished) |

Anti-pattern avoided: FDB-211 used primary GW1–38 with nested “confirm” GW20–38.

EvalSpec `defcon_rate_ab` default window updated to GW1–19; confirmation via `--from-gw 20 --to-gw 38`.

---

## Arms (current tip)

| Role | Spec | `defcon_mode` | `model_id` |
|---|---|---|---|
| Control | `dc_layered_defcon_mean_v1` | mean | `01bf5b3c85e60fa0` |
| Treatment (production) | `dc_layered_v3` | rate | `30367a901721a5d6` |

One-lever opt-out: mean fork `extends` v3. Tip base `ed7a798` (EvalSpec window edit dirty during execute → runner `force_inconclusive`; written call uses deltas).

```bash
export FPLT_DATA_DIR=/tmp/fplt-defcon-rate-confirm
uv run fplt promote-eval --eval defcon_rate_ab --execute \
  --from-gw 1 --to-gw 19 --output-dir /tmp/fplt-defcon-rate-ab-dev
# clear ledger/predictions between roles
uv run fplt promote-eval --eval defcon_rate_ab --execute \
  --from-gw 20 --to-gw 38 --output-dir /tmp/fplt-defcon-rate-ab-confirm2
```

---

## Headline (RMSE) — decision-relevant pool

Positive Δ ⇒ treatment better. Prefer RMSE over MAE when they disagree (W9 / north-star).

### Development GW1–19

| Model | RMSE ↓ | MAE ↓ | Spearman ↑ | n |
|---|---:|---:|---:|---:|
| Control (mean) | **2.8520** | **1.8507** | **0.5572** | 6082 |
| Treatment (rate) | 2.8528 | 1.8516 | 0.5561 | 6082 |
| Δ (treat − control, signed better>0) | **−0.00081** | **−0.00092** | **−0.00112** | |

Per-GW RMSE wins: control **11** / treatment 8. Coverage `dc_season_eligible` 190/190 = 1.0.

### Confirmation GW20–38

| Model | RMSE ↓ | MAE ↓ | Spearman ↑ | n |
|---|---:|---:|---:|---:|
| Control (mean) | **2.6251** | 1.7513 | 0.5158 | 6614 |
| Treatment (rate) | 2.6270 | **1.7450** | **0.5168** | 6614 |
| Δ | **−0.00188** | **+0.00628** | **+0.00097** | |

Per-GW RMSE wins: control **11** / treatment 8. MAE wins: treatment **17** / control 2. Coverage 1.0.

MAE/Spearman improve on confirmation; **RMSE does not**. Mean-seeking board → RMSE headline → confirmation fails §2.2.1–2.

---

## Interim keep checklist (§2.2)

| Gate | Result |
|---|---|
| 1 Development better on RMSE | **Fail** |
| 2 Confirmation same direction (disjoint) | **Fail** (RMSE worse; MAE secondary win ignored for keep) |
| 3 SPA/MCS on confirmation | Not used (keep already fails; two-arm A/B) |
| 4 Slot / components | Not run (forecast other-slot; skipped as in FDB-211) |
| 5 Preflight coverage / leak | **Pass** |

Written **keep** would require 1+2. **Drop** language fits confirmation worse on headline, but §2.4 blocks silent production un-flip. Registry: **provisional stays true**.

---

## Vs FDB-211 nested keep

FDB-211 reported RMSE lifts on GW1–38 and nested GW20–38 at tip `193c0cd` (control then `dc_layered_v2` mean, treatment `dc_layered_defcon_rate_v1`). Protocol reconfirm on current tip mean-fork vs `dc_layered_v3` does **not** reproduce an RMSE keep on disjoint halves. Possible contributors: stack drift on v3 since the flip, cleaner opt-out A/B, and the original nested-window anti-pattern. This doc does not re-litigate FDB-211; it answers the protocol confirmation question only.

---

## Actions

| Item | Action |
|---|---|
| Registry `defcon_rate.provisional` | **Leave true** (confirm did not pass interim keep) |
| Production `dc_layered_v3` | **Unchanged** (no silent un-flip) |
| EvalSpec `defcon_rate_ab` window | Default → GW1–19; confirm via CLI overrides |
| Follow-up | Optional explicit revert / dose-tune PR if operators want mean wire live again |

---

## Evidence

- `internal/defcon-rate-confirm/development/` — recommendation, audit, compare summary, run log
- `internal/defcon-rate-confirm/confirmation/` — same for GW20–38
- Repo audits: `configs/evals/results/defcon_rate/development/audit.json`, `.../protocol_confirm/audit.json`
