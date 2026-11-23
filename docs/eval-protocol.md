# Eval protocol — windows, keep, reference board, solver

*Standing Project protocol (2026-11-23). Complements [`feature-eval-pipeline.md`](feature-eval-pipeline.md) (runner / EvalSpec) and [`evaluation-north-star.md`](evaluation-north-star.md) (what to measure). Does not replace either.*

**Status:** Interim. Addresses the gap between promote-eval’s auto-`inconclusive` (FDB-73 thresholds unset) and recent **written** keeps (defcon-rate, price-within-horizon). Defcon-rate remains **provisional** after FDB-255 protocol reconfirm **failed** interim RMSE keep ([`defcon-rate-confirm.md`](defcon-rate-confirm.md)); within-horizon `mean_gw_delta` was **confirmed** under §2.3 (FDB-256 — [`price-forecast-confirm.md`](price-forecast-confirm.md)).

**Related tasks:** FDB-225 (this doc + reference board + north-star fixes), FDB-228 (deterministic HiGHS), FDB-227 (decisions-suite gate — §8, landed in loader), FDB-226 (v3 rename plan — stub §9).

| Item | Owner | Status |
|---|---|---|
| 1 Windows + interim keep | FDB-225 | This doc §1–§2 |
| 2 Reference board config | FDB-225 | This doc §4 + `configs/boards/reference.yaml` |
| 3 Production model rename (`dc_layered_v3`) | FDB-226 | Sibling (§9) |
| 4 Decisions suite gate | FDB-227 | Landed — §8 + loader |
| 5 Deterministic HiGHS | FDB-228 (+ FDB-225) | This doc §5 |
| 6 North-star doc fixes | FDB-225 | This doc §6 |

---

## 1. Season windows

Every promote registers **three** roles for each season it touches. Windows must be declared in the EvalSpec / promote plan **before** looking at confirmation numbers.

| Role | Purpose | Hard rules |
|---|---|---|
| **Development** | Tune doses, pick among candidates, debug coverage | May overlap other levers’ development windows. **Not** a keep window. SPA/MCS here is descriptive only and burns the shared budget (§3). |
| **Confirmation** | Written keep / drop / tune | **Non-overlapping** with that lever’s development window (disjoint GW ranges and/or a different season). Do not re-select the dose on this window. |
| **Sealed** | Held out for a later claim (Tier B/C, multi-season, or a second confirmation) | No peeking for the current lever. Unfinished / not-`data_checked` GWs are sealed by default. |

### 1.1 Default per-season map (until a lever overrides in its plan)

| Season | Development | Confirmation | Sealed |
|---|---|---|---|
| `2024-25` | GW1–20 | GW21–38 | — (fully historical; treat GW21–38 as confirmation only once) |
| `2025-26` | GW1–19 | GW20–38 (finished GWs only) | Unfinished GWs until `data_checked` |
| `2026-27+` | First half of finished GWs | Second half of finished GWs | Next unfinished stretch + next season |

Live-only fields (defcon awards, snapshot `penalties_order`) set `window.seasons` explicitly and still split development vs confirmation **inside** that season without GW overlap.

**Anti-pattern (do not repeat):** primary GW1–38 + “confirmation” GW20–38 on the same season. That confirmation is nested, not independent.

---

## 2. Interim keep rule (written calls)

### 2.1 Runner vs written call

| Layer | Behaviour |
|---|---|
| **`fplt promote-eval` auto-verdict** | Unchanged: improvement levers stay **`inconclusive`** (or **`drop`** on coverage / leak / mean-PPG). Equivalence keeps §3.2 identity / interval rules. |
| **Written call** in the promote doc / Project store | May be **`keep` / `tune` / `drop`** under this interim rule. The runner does not apply `keep_if`. |

This closes the “inconclusive-only forever” vs “we already flipped production” contradiction without inventing numeric ε in the runner.

### 2.2 Written `keep` (improvement levers)

All of the following:

1. **Primary (development or a declared primary board):** treatment better on the **headline** metric — RMSE on the decision-relevant pool for mean-seeking forecast levers; season points for decision-policy levers. W9-style MAE-worse / RMSE-better → prefer RMSE ([`w9-mae-rmse-reading.md`](w9-mae-rmse-reading.md)).
2. **Confirmation:** same directional call on a **non-overlapping** confirmation window (§1). Dose / arm selection must not use confirmation rows.
3. **SPA / MCS (§3):** if several arms shared the confirmation window, treatment is inside the model confidence set (or SPA does not reject it vs control). SPA on the development window alone is **not** a keep.
4. **Slot / components:** when the lever targets a component slot, the slot metric is not worse on confirmation (else **`tune`**).
5. **Preflight:** coverage and leak gates pass (or an explicit `allow_low_coverage` story that forces `inconclusive` / no keep).

Written **`drop`:** confirmation worse on the headline, or an entirely-bad HAC/bootstrap interval on confirmation, or a failed plan gate listed in the promote plan.

Written **`tune`:** mixed metrics, nested/overlapping windows only, or development win without confirmation.

### 2.3 Decision-policy keeps

Same shape via chip / horizon boards ([`promote-pipeline-unified.md`](promote-pipeline-unified.md) / FDB-129): primary season board → confirmation season → optional N6 SE when claiming chips / H. Control and treatment differ by **one** policy flag. Report against the **reference board** (§4).

### 2.4 Provisional / confirmed keeps (as of 2026-11-23)

| Keep | Status | Notes |
|---|---|---|
| Defcon-rate → production `dc_layered_v3` | **Provisional** (FDB-255 confirm **fail**) | Protocol disjoint GW1–19 / GW20–38 failed interim RMSE keep ([`defcon-rate-confirm.md`](defcon-rate-confirm.md)). Leave production rate until an explicit flip PR. Compose renamed in FDB-226 (frozen mean-wire `dc_layered_v2`). |
| Within-horizon `mean_gw_delta` price forecast (live default) | **Confirmed** (FDB-256) | Independent confirmation season 2024-25 logged under §2.3 — [`price-forecast-confirm.md`](price-forecast-confirm.md). Live default stays. |

Do **not** silently un-flip provisional rows. New levers must follow §1–§2; revisit provisional rows when sealed / confirmation capacity exists.

---

## 3. SPA / MCS budget

Hansen SPA and the model confidence set are **shared** across every EvalSpec / bakeoff that touches the same `(season, GW window, metric family)`.

Rules:

1. **Pre-register** multi-arm looks that share a window (bakeoff YAML, dose `arms:`, or a list in the promote plan).
2. Each shared window has a **budget of one** championship-style SPA/MCS readout for keep purposes. Extra descriptive tables are allowed but must say “does not burn / does not keep.”
3. Development-window SPA never promotes. Confirmation-window SPA may support a written keep (§2.2.3).
4. Re-running SPA after peeking at rankings on the same window is forbidden; move to a fresh window or the sealed role.

---

## 4. Reference board

Versioned flags live in [`configs/boards/reference.yaml`](../configs/boards/reference.yaml). Load via `fplt.optimise.reference_board.load_reference_board`.

Boards that claim live-comparable season points **report against this config**. Live `run_plan` / automation defaults **must match** (enforced by unit tests). Ablations flip one flag at a time and name the flip in the scoreboard / promote doc.

| Flag | Reference / live |
|---|---|
| `horizon` | `3` (`DEFAULT_PLAN_HORIZON`) |
| `chip_arm` | `optimised` (`DEFAULT_LIVE_CHIP_ARM`) |
| `force_play` | `true` |
| `lineup_reopt` | `true` |
| `price_forecast` | `mean_gw_delta` (`DEFAULT_LIVE_PRICE_FORECAST_METHOD`) |
| `bench_weight` | `0` (`configs/squad_state.yaml`) |
| `model` | `dc_layered_v3` (`configs/ops/production.yaml`) |
| HiGHS (boards) | §5 — deterministic; live planner may keep the 600s / gap `1e-3` production knobs |

Chip-board **control** A/Bs may omit `--price-forecast` on purpose; say so. The reference row for “what live would score” includes `mean_gw_delta`.

**Tip:** record git SHA (and prediction `model_id` when relevant) beside every board archive.

---

## 5. Deterministic HiGHS (boards)

Chip-board path (`run_chip_decision_board` → `simulate_decisions` → `plan_squad`) uses `HighsSolveOptions.board_deterministic()`:

- `time_limit=None` (no wall-clock cut-off; prove / finish)
- `parallel="off"`, `threads=None` (do **not** pin `threads=1` — highspy 1.16 returns `kNotset` on pruned empty-squad H≥3 MILPs)
- fixed `random_seed=0`
- `mip_rel_gap=0.0`, `accept_time_limit=False`

**Sanity check:** run the same chip-board twice on identical staged data + tip → **identical season totals** (and identical per-GW points). If totals diverge, the board is not publishable — fix seed / thread / time-limit leakage before claiming a keep.

Live `fplt plan` keeps `HighsSolveOptions.open_fpl_defaults()` (600s, gap `1e-3`) so unattended solves stay bounded.

---

## 6. North-star alignment

- RMSE headline for mean-seeking boards; MAE secondary ([`evaluation-north-star.md`](evaluation-north-star.md), [`w9-mae-rmse-reading.md`](w9-mae-rmse-reading.md)).
- `lgbm_direct_v1` trains with LightGBM `objective: regression` (L2); `metric: mae` is monitoring only — do not describe it as an MAE-objective median model.
- Decision-sim bars are **points-at-rank** under reference-board rules (anchor **2400** ≈ OR ~573k in 2024-25; **2235** in 2025-26). See [`evaluation-north-star.md`](evaluation-north-star.md) §2.

---

## 7. Checklist (operator card)

0. **Plumbing is not done.** Every feature that lands measurement or forecast plumbing (new EvalSpec, ModelSpec fork, staged table consumer, coverage gate, etc.) **must kick off this promote pipeline** — development → confirmation → written keep/drop/tune. Do **not** stop at the plumbing PR. The board that the EvalSpec defines is the next required step; a merged EvalSpec without a written call is incomplete work.
1. Name one lever; fork YAML / policy flag; production unchanged until flip PR.
2. Declare development / confirmation / sealed windows (§1).
3. Run primary / development suite; record SPA budget use (§3).
4. Run confirmation on a non-overlapping window (**fresh `FPLT_DATA_DIR` / empty ledger** — reusing the development data root contaminates prediction summary and HAC via the shared ledger cache).
5. Written call under §2; mark **provisional** if confirmation is weak or nested.
6. Flip PR only after keep; update defaults + prefs + design; never chat-only. Registry keep ≠ production flip.
7. Decision boards: reference config (§4) + deterministic HiGHS (§5); tip + run-twice check.
8. ModelSpec season/squad-points claims: `claim: squad_points` + `suites.decisions: true` (§8).

---

## 8. Decisions suite gate (FDB-227)

**Rule:** any EvalSpec that claims **season points** or **squad points** must set:

```yaml
claim: squad_points
suites:
  decisions: true
```

and run at tier **S3+** (S0–S2 force decisions off and fail this claim).

| Claim | When to use | Decisions suite |
|---|---|---|
| `forecasts` (default) | RMSE / Brier / slot metrics / equivalence | Off (default) |
| `squad_points` | “This ModelSpec earns more season / squad points” | **Required on** |

Enforced in `fplt.backtest.promote_eval.EvalSpec` / `decisions_claim_failures` (load + preflight):

- `claim: squad_points` + decisions off → fail.
- Forecast claim whose `question` / `description` / `change.notes` / `report.title` names `season points`, `squad points`, `mean ppg`, or `realised points` while decisions is off → fail (reword or flip the claim).
- Equivalence kinds cannot take `claim: squad_points`.

**Not this gate:** decision-**policy** levers (`force_play`, `DEFAULT_PLAN_HORIZON`, chip arm, lineup reopt, sell-on). Those are chip-board / `fplt sim-season` A/Bs, not ModelSpec EvalSpecs — see [`promote-pipeline-unified.md`](promote-pipeline-unified.md). Promote-eval’s decisions suite remains the chips-off-shaped bridge for ModelSpec season-points claims (P4).

**Wiring:** scaffold `configs/evals/example_model_squad_points.yaml`. Existing real EvalSpecs on main are forecast claims (`decisions: false`); none asserted season points, so none were flipped.

Canonical detail: [`feature-eval-pipeline.md`](feature-eval-pipeline.md) §4.2 and §8.

---

## 9. ModelSpec versioning (FDB-226 done)

Production `dc_layered_v2` was edited in place when defcon-rate flipped (`4d715b5`). **FDB-226** froze mean-wire `dc_layered_v2` (pre-flip `model_id`) and moved production to `dc_layered_v3`. Do not edit a released production ModelSpec in place again — bump the versioned YAML and retarget ops / forks. Boards label `name (model_id)` so a stable name alone cannot hide a hash change.
