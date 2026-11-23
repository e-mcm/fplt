# Promote candidate: H=8 + force_play=off

**Goal:** Flip production defaults to **`DEFAULT_PLAN_HORIZON=8`** and **`force_play=False`**, keeping **`DEFAULT_LIVE_CHIP_ARM=optimised`**.  
**Evaluate rigorously first; flip PR only after gates pass.** Do not flip from chat or from the 2025-26 board alone.

| | Control (live today) | Candidate |
|---|---|---|
| Planner horizon | `DEFAULT_PLAN_HORIZON=3` · ops `plan_horizon: 3` | **8** |
| Force-play | `True` (scheduler / `run_plan` default) | **`False`** |
| Chip arm | `optimised` | `optimised` (unchanged) |
| Model | `dc_layered_v2` | unchanged |

**Evidence so far (not yet a pass):** post-#150/#152 board ([force-play-horizon-grid.md](./force-play-horizon-grid.md), tip `28916d7`) — H=8 force-off **2086** best on 2025-26; force-off ≥ force-on at every H. Prefs still say force-play **on** + keep H=3 pending more seasons.

### Same pipeline, different artifact

This candidate uses the **same promote flow** as ModelSpec config A/Bs (control vs treatment → primary window → confirmation → written call → **later** flip PR). The artifact is a **decision board**, not an EvalSpec: lever is planner defaults (`H`, `force_play`), metric is season points, runner is `fplt decisions --chip-board` (+ N6), flip edits code/ops constants — FDB-129 shape. Do not wrap it in `configs/evals/` / `fplt promote-eval` (those compare ModelSpecs; P4 decisions there is still chips-off-shaped). Standing map: [promote-pipeline-unified.md](./promote-pipeline-unified.md). Repo gates: `docs/design.md` §10.1 / §7.4 N6, north star §2.

---

## 1. Eval cases required to promote

Primary metric everywhere: **season points** (full decision sim, chips on for optimised). RMSE / `fplt promote-eval` forecast suites do **not** gate this flip.

| # | Case | Season(s) | What it answers | Status |
|---|---|---|---|---|
| **A** | Force × H grid (optimised) | **2025-26** | Candidate cell vs control cell on live rules year | **Done** — [force-play-horizon-grid.md](./force-play-horizon-grid.md) |
| **B** | Same force × H grid | **2024-25** (FDB-138) | Confirmation season; multi-season consistency | **Done** — [post150-grid-2024-25.md](./post150-grid-2024-25.md); tip `28916d7` |
| **C** | Three-arm chip board at **H=8**, `force_play=False` on optimised | **2025-26** and **2024-25** | design §10.1: longer H only if **optimised beats chips-off** (and preferably rule_based) at that H under the **candidate** force setting | 2025-26: **Done** @ `28916d7` — rule **2016** / opt **1963** / off **1945** → opt ≥ off **PASS (+18)**; opt ≥ rule **FAIL (−53)**; opt path −123 vs single-arm 2086 — [post150-h8-three-arm-2025-26.md](./post150-h8-three-arm-2025-26.md); **2024-25 Done:** opt **2298** ≥ rule **2222** ≥ off **2191** (opt ≥ off **PASS**) — [post150-grid-2024-25.md](./post150-grid-2024-25.md) |
| **D** | Head-to-head: **candidate vs control** | Both seasons | Explicit `H=8 force_off` vs `H=3 force_on` season pts (the production question) | 2025-26: **Done** — grid 2086 vs 1897 (**+189**); three-arm opt vs control **+66** (1963) — [post150-h8-three-arm-2025-26.md](./post150-h8-three-arm-2025-26.md); **2024-25: 2298 vs 2311 (−13) — FAIL** — [post150-grid-2024-25.md](./post150-grid-2024-25.md) |
| **E** | Sim-season SE (N6) | Prefer 2025-26 ledger; smoke then full | design: required before trusting horizon / chip claims; paired SE on candidate − control | **Aborted / deferred** — CLI smokes OK @ `5cea924`; full H2H CRN killed (ETA 10–40h). No mean_diff. [post150-n6-h8-force-off.md](./post150-n6-h8-force-off.md) |

**2024-25 chip rules:** run true `ChipRules.for_season("2024-25")` as the primary confirmation. Optional labelled counterfactual (`--chip-rules-season 2025-26`) is extra, not a substitute.

**Minimum bar to open the flip PR:** **A + B + D(both seasons) + C(both seasons, optimised ≥ chips-off at H=8 force_off)**. **E** is required by design before calling the claim “trusted”; if E stays blocked, land a small `force_play` thread into sim-season first (not an EvalSpec).

---

## 2. Commands / entrypoints

### 2.1 Predictions (per season, once)

```bash
# steps must cover max planner H
export FPLT_DATA_DIR=./data   # or a season-scoped dir
rm -f "$FPLT_DATA_DIR/ledger/predictions/part.parquet"
uv run fplt backtest --models dc_layered_v2 --season 2024-25 \
  --from-gw 1 --to-gw 38 --steps 8
# repeat for 2025-26 if that ledger is stale vs tip 28916d7+
```

### 2.2 Case A / B — force × H grid (optimised only)

```bash
for H in 3 5 8; do
  # force-on (control-like at each H)
  uv run fplt decisions --chip-board --season 2024-25 --horizon $H \
    --models dc_layered_v2 --from-gw 1 --to-gw 38 --chip-arms optimised
  # archive ledger/decisions before the next cell overwrites
  # force-off (candidate-like at each H)
  uv run fplt decisions --chip-board --season 2024-25 --horizon $H \
    --models dc_layered_v2 --from-gw 1 --to-gw 38 \
    --chip-arms optimised --no-force-play
done
```

Make wrappers: `make chip-decision-board SEASON=2024-25 HORIZON=8 MODELS='dc_layered_v2' ARGS='--chip-arms optimised --no-force-play'`.

### 2.3 Case C — three-arm at H=8 under candidate force setting

```bash
uv run fplt decisions --chip-board --season 2024-25 --horizon 8 \
  --models dc_layered_v2 --from-gw 1 --to-gw 38 \
  --chip-arms off rule_based optimised --no-force-play
# --no-force-play only changes the optimised arm; off/rule ignore it
```

### 2.4 Case D — production head-to-head (record both rows)

| Arm | Command flags |
|---|---|
| **Control** | `--horizon 3 --chip-arms optimised` (force-play on = default) |
| **Candidate** | `--horizon 8 --chip-arms optimised --no-force-play` |

### 2.5 Case E — sim-season SE (after force_play is wired)

```bash
# Same-H force ablation smoke (CLI / make; one force setting for all arms):
make sim-season-smoke SEASON=2025-26 HORIZON=3 MODELS='dc_layered_v2' \
  ARGS='--chip-arms optimised'
make sim-season-smoke SEASON=2025-26 HORIZON=8 MODELS='dc_layered_v2' \
  ARGS='--chip-arms optimised --no-force-play'

# Promote H2H CRN (control H=3 force-on vs candidate H=8 force-off) —
# one run_simulated_season_eval with two DecisionPolicy rows (per-policy
# horizon + force_play). CLI has a single --horizon / --no-force-play.
```

```python
from fplt.backtest.simulated_season import (
    DecisionPolicy,
    format_simulated_season_report,
    run_simulated_season_eval,
    truth_rates_from_actuals,
)
from fplt.optimise.chip_windows import chip_arm_model_id

base = "dc_layered_v2"  # ledger model_id
control = DecisionPolicy(
    policy_id=f"{chip_arm_model_id(base, 'optimised')}::h3_force",
    predictions=predictions,
    chip_arm="optimised",
    chip_rules_season="2025-26",
    force_play=True,
    horizon=3,
)
candidate = DecisionPolicy(
    policy_id=f"{chip_arm_model_id(base, 'optimised')}::h8_noforce",
    predictions=predictions,
    chip_arm="optimised",
    chip_rules_season="2025-26",
    force_play=False,
    horizon=8,
)
report = run_simulated_season_eval(
    candidates,
    [control, candidate],
    draft_state,
    season="2025-26",
    gameweeks=list(range(1, 39)),
    horizon=3,  # unused when each policy sets horizon
    n_paths=50,  # smoke with small N first
    seed=0,
    truth_rates=truth_rates_from_actuals(actuals),
    fixtures=fixtures,
    pairs=[(control.policy_id, candidate.policy_id)],
)
print(format_simulated_season_report(report))
```

### 2.6 Do **not** use for the keep decision

```bash
uv run fplt promote-eval --eval <name> --execute   # ModelSpec levers only
uv run fplt bakeoff …                             # forecast ranking, not a keep
```

CI (`.github/workflows/ci.yml`) does not run these boards.

---

## 3. Pass / fail vs control (H=3 force-on)

All comparisons: same model `dc_layered_v2`, GW1–38, tip ≥ `28916d7` lineage (#150–#152).

| Gate | Pass if | Fail if |
|---|---|---|
| **D — head-to-head** | Candidate season pts **≥** control on **2025-26 and 2024-25** | Candidate loses either season |
| **B — force-off vs force-on** | On 2024-25, force-off ≥ force-on at H=8 (and ideally at H=3/5, matching 2025-26) | Force-on wins 2024-25 at H=8 |
| **C — three-arm @ H=8 force_off** | Optimised (force_off) **≥ chips-off** on each season; prefer also ≥ rule_based | Optimised loses to chips-off at H=8 (cannot promote H — design §10.1) |
| **H=5 sanity** | Do not promote H=5; candidate is **H=8 only**. H=5 may stay ugly for force-on — irrelevant if H=8 candidate clears D/C | Using H=5 win/loss to flip default H |
| **E — sim-SE** | Paired mean(candidate − control) **> 0** and not explained away by SE (mean_diff ≳ 2×SE as a practical bar until FDB-73-style decision thresholds exist) | mean_diff ≤ 0 or \|mean_diff\| ≪ SE |
| **Mean-PPG sanity** | Board exits 0 under default `--max-mean-ppg 90` | Gate trip / absurd totals |

**2025-26 snapshot (informational, tip `28916d7`):**

| Policy | Pts | vs control |
|---|---:|---:|
| Control H=3 force-on | 1897 | 0 |
| Candidate H=8 force-off (grid single-arm) | **2086** | **+189** |
| Candidate H=8 force-off (three-arm Case C) | 1963 | +66 |
| H=8 force-on (reject) | 1950 | +53 but ≪ grid candidate |
| H=8 three-arm rule_based | 2016 | +119 (chips, not the candidate policy) |
| H=8 three-arm chips-off | 1945 | +48 |

Case C hard gate (opt ≥ off @ H=8 force_off): **PASS (+18)**. Soft (opt ≥ rule): **FAIL (−53)**. Optimised path dependence (2086 vs 1963) is a promote risk.

**2024-25 snapshot (tip `28916d7`, [post150-grid-2024-25.md](./post150-grid-2024-25.md)):**

| Policy | Pts | vs control |
|---|---:|---:|
| Control H=3 force-on | **2311** | 0 |
| Candidate H=8 force-off | **2298** | **−13** |
| H=8 force-on | 2229 | −82 |
| H=8 three-arm rule_based | 2222 | −89 |
| H=8 three-arm chips-off | 2191 | −120 |

Case D on 2024-25: **FAIL (−13)**. Case C hard gate: **PASS** (2298 ≥ 2191). **Do not open the flip PR.**

Live rank / overall percentile is **not** a gate (north star §3).

---

## 4. Ordered checklist → flip PR

1. **Lock candidate definition** — H=8 + `force_play=False` + optimised; control = H=3 + `force_play=True` + optimised. (This doc.)
2. **Keep 2025-26 board as Case A** — no full re-grid unless tip/data changes; cite [force-play-horizon-grid.md](./force-play-horizon-grid.md).
3. **Unblock Case E tooling** — ~~blocked~~: `force_play` threaded through `DecisionPolicy` / `chip_arm_policies` / `run_chip_arm_sim_season` / `fplt sim-season` / `run_plan` (`--no-force-play`). Per-policy `horizon` for H2H CRN. **No default flip.**
4. **FDB-138 preds** — ~~queued~~ **Done** @ `28916d7` (`--steps 8`, 221904 rows) — [post150-grid-2024-25.md](./post150-grid-2024-25.md).
5. **Run Cases B + C + D on 2024-25** — **Done**; archive `internal/force-play-horizon-grid/2024-25/`. **D FAIL** (candidate 2298 < control 2311).
6. **Re-confirm Case C on 2025-26 at one tip** (three-arm H=8 `--no-force-play`) if off/rule rows from older tips are not trusted next to 2086. **Done** @ `28916d7`: opt 1963 ≥ off 1945 (+18); rule leads at 2016; opt −123 vs single-arm 2086 (path dependence).
7. **Run Case E** — ~~full GW paired sim-SE~~ **aborted/deferred** (CLI smokes only; ETA 10–40h) — [post150-n6-h8-force-off.md](./post150-n6-h8-force-off.md).
8. **Write the keep call** on this doc / board: pass/fail table for both seasons + SE. If any hard gate fails → **stop**; leave defaults and prefs unchanged.
9. **Only after all hard gates pass — flip PR** (FDB-129 shape), single PR:
   - `DEFAULT_PLAN_HORIZON = 8` in `fplt.optimise.plan`
   - `force_play` default **`False`** in `schedule_chip_windows` / `simulate_decisions` / chip-board CLI; **`run_plan` must pass it** (today live always force-on)
   - `configs/ops/production.yaml` → `plan_horizon: 8`
   - Makefile / CLI help defaults that hardcode `3` for plan horizon
   - Repo `docs/design.md` §5.5 / §10
   - Project `preferences.md` (force-play off; H=8 live)
   - Store notes: [season-chip-scheduler.md](./season-chip-scheduler.md), this doc → **Promoted**
   - Tests asserting the new defaults
10. **Merge flip PR** when CI green (Project pref). Do **not** open this PR at step 1.

**No EvalSpec scaffold PR** — pipeline does not require `configs/evals/` for decision-policy knobs. The only optional pre-flip code PR is step 3 (sim-season / `run_plan` `force_play` wiring).

---

## 5. Blockers

| Blocker | Impact |
|---|---|
| **2024-25 H2H fail (D)** | Candidate **2298** < control **2311** (−13) — **hard stop** on promote ([post150-grid-2024-25.md](./post150-grid-2024-25.md)) |
| ~~FDB-138 / 2024-25 preds~~ | Done on tip `28916d7` |
| ~~**`fplt sim-season` lacks `force_play`**~~ | Wired — CLI `--no-force-play`; case E H2H via two `DecisionPolicy` rows |
| ~~**`run_plan` ignores force_play`**~~ | Wired — `run_plan(..., force_play=)` / `fplt plan --no-force-play` |
| **Prefs: force-on + H=3** | Update only in the flip PR after gates pass |
| **P4 promote-eval decisions** | Unfinished and chips-off-shaped — irrelevant; do not wait on it |
| **Single-season path dependence** | Why A alone is insufficient (north star §2 uncertainty) |
| **Pre-#150 boards** | Superseded; do not mix with post-#152 totals |

---

## Related

- Unified flow: [promote-pipeline-unified.md](./promote-pipeline-unified.md)
- Board 2025-26: [force-play-horizon-grid.md](./force-play-horizon-grid.md)
- Board 2024-25: [post150-grid-2024-25.md](./post150-grid-2024-25.md)
- Prior H=8 three-arm (force-on era): [h8-chip-board.md](./h8-chip-board.md)
- C6 / live optimised promote: [post-chips-bakeoff-report.md](./post-chips-bakeoff-report.md) · PR [#122](https://github.com/e-mcm/fplt/pull/122)
- N6 sim-season: [simulated-season-eval.md](./simulated-season-eval.md) · PR [#125](https://github.com/e-mcm/fplt/pull/125)
- North star: [evaluation-north-star.md](./evaluation-north-star.md)
- Repo: `docs/design.md` §10.1 (horizon gate), §7.4 N6; `docs/feature-eval-pipeline.md` (ModelSpec path — not this lever)
