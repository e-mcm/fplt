# Design: joint simulator samples → optimiser (FDB-241)

**Task:** FDB-241 · **Lever rank:** 4 · **Prior:** Wave D1 scenario XI (`docs/design.md` §10.2)  
**Metric:** scenario lineup points vs EV (`compare_ev_vs_saa_lineup`) → season-points promote  
**Flip:** none until written keep (opt-in `--joint-scenarios`)

---

## 1. Problem

D1 shipped SAA lineup selection (`ScenarioBundle` → `select_lineup_scenarios`) and default EV `lineup_reopt`, but **live / decision boards never fed joint Monte Carlo draws**. The layered predictor threw away per-sim points after averaging into `xpts`. Correlated stacks, blank insurance, and captain/vice fallback under joint scorelines were unavailable to the optimiser.

## 2. Goals / non-goals

**Goals**

1. `CompositePredictor.sample(view, horizon, n)` → `PredictionSampleSchema`.
2. Aggregate fixture draws to player–GW `ScenarioBundle` (DGW sum on shared `sim`).
3. Opt-in wire: `plan_squad(scenarios=…)`, `run_plan` / `run_decisions` / CLI `--joint-scenarios`.
4. Metric helper: EV vs SAA mean realised points under the same joint bundle.

**Non-goals (this PR)**

- Persist samples to disk / Dagster asset (D1 deferred).
- Joint transfer SAA inside the multi-period MILP objective.
- Flip live default to SAA (needs promote keep).
- Change `n_sims` used by `predict` / RMSE boards.

## 3. API

```python
# Layered model
samples = model.sample(view, [GWKey(season, gw)], n=500)  # PredictionSampleSchema

# Bridge
from fplt.optimise.scenarios import scenario_bundle_from_samples
bundle = scenario_bundle_from_samples(samples, season=season, gw=gw)

# Lineup metric (scenario points vs EV)
from fplt.optimise.lineup import compare_ev_vs_saa_lineup
cmp = compare_ev_vs_saa_lineup(locked_fifteen, bundle)
# cmp.delta = saa_mean - ev_mean

# Plan / decisions (opt-in)
plan_squad(..., scenarios=bundle)
run_plan(..., joint_scenarios=True, n_scenario_sims=500)
# CLI: fplt plan --joint-scenarios · fplt decisions --chip-board --joint-scenarios
```

Default SAA draw count: `DEFAULT_SAA_SIMS = 500` (not ModelSpec predict `n_sims`).

## 4. Promote

Decision-policy A/B: EV reopt (control, live default) vs `--joint-scenarios` (treatment).  
Plan: [`promote-plans/joint-scenarios-saa.md`](./promote-plans/joint-scenarios-saa.md).
