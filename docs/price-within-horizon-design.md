# Design: within-horizon list-price forecasts

**Task:** FDB-215 · **Lever:** Decision-policy (planner prices) · **Sibling:** FDB-205 as-of 50% sell-on (different lever — do not duplicate).  
**Plan:** [promote-plans/price-within-horizon.md](./promote-plans/price-within-horizon.md) · inventory [unpromoted-improvements.md](./unpromoted-improvements.md).  
**Repo tip at design:** `main` @ `7f3f1e9` (post #124 sell-on; within-horizon still flat).

---

## 1. Problem

`plan_squad` solves a multi-step MILP. Buy costs and sell credits today are **as-of only**:

- Step-0 list prices come from PIT `player_status` / candidates.
- Owned sell credits use `fpl_selling_price(purchase, as_of_list)` (#124).
- Steps `1..H-1` reuse those same numbers (`plan.py` copies a 1-D `list_price` / `sell_price` into every cash row).

That is an honest gap, not a bug: the planner cannot prefer “buy before a predicted rise” or “sell before a predicted fall” inside one horizon. Decision sims **between** deadlines still refresh from historical staged prices (FDB-205 / #124). This redesign fills the **within-horizon** gap only.

| Lever | What changes | Status |
|---|---|---|
| As-of sell-on (FDB-205) | Sticky purchase + 50% rule at each deadline | Live via #124; promote board in flight |
| Within-horizon forecast (this) | Predicted list (and derived sell) prices for steps `>0` | Redesign + code; board **after** 205 keep/drop |

---

## 2. Goals / non-goals

**Goals**

1. Forecast **list prices** at each horizon step from **deadline-visible** inputs only.
2. Wire forecasts into `plan_squad` cash / Free Hit budget constraints (optional; default off).
3. Derive owned sell credits per step via existing `fpl_selling_price`.
4. Keep step 0 identical to today’s as-of market (no same-deadline foresight).

**Non-goals (v1)**

- Exact FPL daily net-transfer threshold algorithm (needs staged `transfers_in` / `transfers_out` — not in `player_status` today).
- Nonlinear mid-horizon buy→sell sell-on for players **not** currently owned (purchase depends on buy step; would need extra MILP state).
- Season-points promote board / default flip (blocked on FDB-205 written call).
- Changing as-of refresh / `--freeze-selling` (FDB-205 owns that surface).

---

## 3. PIT contract

Price forecasts consumed at deadline `as_of` may use only:

- `PITView` rows with `available_at ≤ cutoff`, or frames already filtered that way.
- As-of candidate / `player_status` list prices for step 0.

Forbidden: future `player_status` snapshots, post-deadline transfer tallies, or realised next-GW prices as planner inputs. Decision **scoring** between deadlines may still use historical actuals (existing sim path); that is not this forecast.

Step 0 of every forecast frame **must** equal the as-of list price for each player. Later steps are predictions, not leaks.

---

## 4. Forecast model (v1)

**Grain:** `player_key × step` → integer `list_price` (tenths of £m), schema `PriceForecastSchema`.

**Library:** `fplt.optimise.price_forecast` (planner-adjacent; not a `PointsPredictor`). Config knobs live as function args / small dataclass — no ModelSpec until a promote wants a named arm.

| Method | Behaviour | Use |
|---|---|---|
| `flat` | `list[s] = as_of` for all steps | Control; today’s behaviour |
| `mean_gw_delta` | Per player, mean of last *K* **deadline-to-deadline** price deltas visible at `as_of`; `list[s] = as_of + round(s · δ)` (floored at 0) | Treatment v1 |

**Deadline prices from PIT:** for each gameweek with `deadline_time ≤ as_of.cutoff`, take the latest `player_status.price` with `available_at ≤ that deadline`. Vaastav already stages one price per GW deadline; live `fpl_api` snapshots collapse the same way. Missing players / empty history → `δ = 0` (flat).

**Why not transfers?** Bootstrap exposes `transfers_in` / `transfers_out`, but they are not staged. Staging them is a follow-up (schema + ingest PR) before a threshold / “effective ownership” model. Mean GW Δ is the smallest PIT-honest signal we already have.

**Calibration:** no pytest quality gates. Optional later: Brier/MAE of predicted rise/fall vs next deadline on a holdout — descriptive only until the decision board.

---

## 5. Planner wiring

### 5.1 API

```text
plan_squad(..., price_forecast: pl.DataFrame | None = None)
```

- `None` or flat forecast → identical cash constraints to today (regression target).
- Frame must match `PriceForecastSchema`, cover every pool `player_key` for `step ∈ 0..H-1`, and step-0 prices must match candidates (strict check).

### 5.2 Arrays

Inside the solve, expand to shape `(H, N)`:

- `list_price[s, i]` from the forecast (else broadcast as-of).
- `sell_price[s, i] = list_price[s, i]` for non-owned.
- For owned `i`: `sell_price[s, i] = fpl_selling_price(purchase, list_price[s, i])`.

`_cash` / `_free_hit_budget` index buy/sell weights by the **step at which the transfer occurs**, not a single as-of vector.

### 5.3 Mid-horizon buy→sell (owned vs not)

| Case | Sell credit at step s |
|---|---|
| Owned at plan start | `fpl_selling_price(sticky_purchase, list[s])` |
| Bought at step t &lt; s (not owned at start) | `list[s]` (v1 approx; ignores 50% of rise since t) |

Documented approximation: modelling purchase-contingent sell-on needs extra variables. v1 still captures the main incentive — **buy before a rise / avoid buying into a fall** — and correct sell-on for the current fifteen.

### 5.4 Defaults

- Live `run_plan` / decision boards: **no** forecast (`None`) until a keep on the season-points board.
- Candidate prune / PAR still use step-0 (as-of) prices only.
- Plan pick `price` column remains step-0 / candidate price for continuity of `HorizonPickSchema` (forecast is a constraint input, not a new report column in v1).

### 5.5 Ablation

Future board flag (not required for v1 library): treatment passes `mean_gw_delta` forecast; control omits it. Do **not** reuse `--freeze-selling`.

---

## 6. Repo surfaces to touch

| Path | Change |
|---|---|
| `src/fplt/schemas/plan.py` | `PriceForecastSchema` |
| `src/fplt/optimise/price_forecast.py` | PIT deadline prices + `forecast_list_prices` |
| `src/fplt/optimise/prices.py` | Helpers to build `(H,N)` list/sell arrays; docstring gap update |
| `src/fplt/optimise/plan.py` | Optional `price_forecast`; 2-D cash / FH budget |
| `docs/design.md` §7.3 / §10 | Honest gap → optional forecast; PIT note |
| Tests | Unit: flat≡None; δ accumulation; owned sell-on per step; PIT step-0 lock |

Out of scope for first PRs: CLI flags, Dagster assets, EvalSpec, chip-board args.

---

## 7. Promote (deferred)

Do **not** run the H2H season-points board until FDB-205 writes keep/drop (`docs/price-dynamics-sell-on-promote.md` or coordinator). Then:

1. Control = as-of sell-on + flat within-horizon (post-205 baseline).
2. Treatment = same + `mean_gw_delta` forecasts in `plan_squad`.
3. Gates: season pts ≥ control on 2025-26 and 2024-25; document PIT.
4. Write `docs/price-within-horizon-promote.md`; flip only inside plan/price modules + design §7/§10.

---

## 8. Ordered delivery

1. **This design note** (done).
2. **Code PR:** schema + `price_forecast` + `plan_squad` wiring + tests; default off.
3. Update promote plan steps (design done; board still gated on 205).
4. Archive working notes under `internal/price-within-horizon/`.
5. Promote board + promote doc — **after** 205.
