# Availability overlays — promote-eval call

**Date:** 2026-11-23  
**Task:** FDB-208  
**Path:** ModelSpec · RMSE headline · minutes Brier secondary  
**Plan:** [promote-plans/availability-overlays.md](./promote-plans/availability-overlays.md)  
**Out of scope:** Deep/PPDA / Understat (FDB-55)  
**Flip:** **none** (no keep)

---

## Verdict

| Lever | Call | Why |
|---|---|---|
| **A. FPL availability gate** (already on in `dc_layered_v2`) | **inconclusive / blocked for signal** | Ran promote-eval; under strict PIT the gate is a **no-op** on every finished GW in the archive |
| **B. Injury overlay** (`dc_layered_injuries_v1`) | **blocked** | No staged `player_injuries`; snapshots branch has no `api_football` raw; player id map empty |
| **C. Predicted XI** (`dc_layered_predicted_xi_v1`) | **blocked** | Touchline fetch works, but lineups only for **unfinished** GW11–12 — no scored window |

**Do not flip production.** Gate stays on; injury/XI stay off.

---

## What shipped for the board (repo PR)

| Artifact | Role |
|---|---|
| `use_availability_gate` (default `true`) | Eval-only off-switch for Package A |
| `dc_layered_noavail_v1` + `empirical_minutes_v2_noavail` | Control arm |
| `dc_layered_injuries_v1` | Treatment for Package B |
| EvalSpecs `fpl_availability_gate_ab`, `minutes_injury_overlay_ab`, `predicted_xi_ab` | Promote-eval runners |

`dc_layered_predicted_xi_v1` already existed.

---

## Package A — executed window

```text
uv run fplt promote-eval --eval fpl_availability_gate_ab --execute
# control dc_layered_noavail_v1 · treatment dc_layered_v2
# season 2026-27 GW1–10 · walk_forward · n_sims 10000
```

Artifacts: `internal/availability-overlays/fpl-availability-gate-ab/`  
(`recommendation.md`, `summary.json`, compare/, components/)

### Headline (clean tree)

| Metric | Δ (treatment better > 0) | Note |
|---|---:|---|
| RMSE | **−0.00027** | Noise / different MC seeds |
| MAE | +0.00038 | |
| Brier played / ≥60 (prediction suite) | −0.00016 / −0.00012 | |
| Per-GW RMSE wins | control 4 · treatment 6 | |
| HAC RMSE CI | crosses 0 | |
| Components minutes Brier | **0.0 delta** | Identical PMFs |

Runner verdict: **inconclusive** (improvement levers wait on FDB-73 thresholds).

### Why the lever did not fire

`origin/snapshots` only holds FPL captures from **2026-11-19** onward (`available_at` ≥ that). GW10 deadline is **2026-11-06**; first PIT-visible `source=fpl_api` status is at **GW11** (deadline 2026-11-21). GW11+ are not finished/`data_checked`, so there is **no finished GW where the FPL gate can differ from vaastav**.

On GW1–10 deadlines, only `source=vaastav` status is visible (`status=a`, null chance/news) → both arms match; residual RMSE gap is sim seed noise (`model_id` differs → `seed_for` differs).

Vaastav-only historical seasons would also make arms identical (same lesson as W8 pens_order).

---

## Package B — blocked

- `player_injuries` table **absent** after restage.
- Snapshots branch: **no** `raw/api_football/...`.
- `configs/api_football/id_maps.yaml` `players: {}` → even fetched injuries would stage with null `player_key` and be ignored.
- Overlay default remains **off** in production (correct until a mapped archive exists).

---

## Package C — blocked

- Live Touchline ingest (`FPLT_TOUCHLINE_LINEUPS_ENABLED=1`) staged **400** rows for GW**11–12** only (`finished=false`).
- Historical promote-eval on GW1–10 cannot see those lineups under PIT (capture time is 2026-11-23).
- No finished GW with pre-deadline predicted XI → cannot claim RMSE/Brier.
- `score_predicted_starts` not used for a keep call (nothing to promote).

---

## Next unblocks (not this PR)

1. **Backfill / densify** deadline snapshots through early 2026-27 GWs (or wait until GW11+ are checked) → re-run `fpl_availability_gate_ab`.
2. Capture + stage API-Football injuries **and** fill player id maps → `minutes_injury_overlay_ab`.
3. Archive Touchline/AF lineups **before** each deadline on finished GWs → `predicted_xi_ab` + start-accuracy smoke.

---

## Production

| Knob | Action |
|---|---|
| `dc_layered_v2` / `use_availability_gate` | **Keep** (already on; no evidence to remove) |
| `use_injury_overlay` / `predicted_xi` | **Keep off** |
| `PRODUCTION_MODELS` | **unchanged** |
