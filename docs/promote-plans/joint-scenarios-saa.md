# Promote plan: joint scenarios SAA vs EV lineup

**Task:** FDB-241 · **Path:** Decision-policy · **Pipeline:** [promote-pipeline-unified.md](../promote-pipeline-unified.md)  
**Design:** [joint-samples-optimiser-design.md](../joint-samples-optimiser-design.md) · inventory [unpromoted-improvements.md](../unpromoted-improvements.md)

Closes the open “optional SAA arm” left by the `lineup_reopt` keep. Control is live EV reopt; treatment feeds joint MC samples into step-0 SAA.

---

## 1. Control vs treatment

| | **Control** | **Treatment** |
|---|---|---|
| Lever | `lineup_reopt=True`, no scenarios (EV) | `--joint-scenarios` (SAA on step 0) |
| CLI | `fplt decisions --chip-board …` | same + `--joint-scenarios [--n-scenario-sims 500]` |
| Forecasts | `dc_layered_v4` | same |
| Chips / H / force | optimised · **H=3** · force_play **on** | same |

One lever only — do not co-vary horizon, force_play, chip arm, or model.

---

## 2. Metrics

| Level | Metric |
|---|---|
| **Primary (season)** | Season points (north star §2) on chip-board optimised arm |
| **Diagnostic** | `compare_ev_vs_saa_lineup` mean delta on locked fifteens (scenario lineup points vs EV) |

---

## 3. Artifact / runner

```bash
# Control (EV reopt — live default)
uv run fplt decisions --chip-board --season $SEASON --horizon 3 \
  --models dc_layered_v4 --from-gw 1 --to-gw 38 --chip-arms optimised

# Treatment (joint SAA)
uv run fplt decisions --chip-board --season $SEASON --horizon 3 \
  --models dc_layered_v4 --from-gw 1 --to-gw 38 --chip-arms optimised \
  --joint-scenarios --n-scenario-sims 500
```

Separate `$FPLT_DATA_DIR` per arm. Archive under store `internal/joint-scenarios-saa-promote/`.

---

## 4. Windows / gates

| # | Window | Role |
|---|---|---|
| **A (dev)** | **2025-26** short smoke then GW1–38 H=3 opt force_on | Primary |
| **B (confirm)** | **2024-25** same shape | Confirmation |

- **Keep** if treatment ≥ control on **both** seasons (ties OK).
- **Drop** if treatment loses either → leave default EV; document.
- **Tune** if mixed (e.g. adjust `n_scenario_sims` / haul) — no flip.
- Mean-PPG sanity ≤ 90.

---

## 5. Flip PR surface

| Call | Action |
|---|---|
| **Keep** | PR: live `run_plan` / automation default `joint_scenarios=True`; prefs + design. |
| **Drop** | Document only; flag stays opt-in. |
| **Tune** | Written follow-up; leave default EV. |
