# Promote pipeline — one flow, two artifacts

**Status:** Standing Project note (2026-11-23). Describes **repo reality**, not a new process.  
**Canonical runners:** repo `docs/feature-eval-pipeline.md` (ModelSpec) · `fplt decisions` / design §10 (decision policy) · north star §1–2 (what to measure).

---

## Same overall flow

Both lever types follow the same promote checklist. Only the **artifact** and **metric suite** change.

```text
1. Name control vs treatment (one lever). Production stays put until the flip PR.
2. Run the matching suite(s) on a primary window.
3. Confirm on a fresh window / season (dose or single-season win ≠ promote).
4. Write keep / tune / drop / inconclusive (or decision equivalent).
5. Production change is a *later* PR — defaults, prefs, design docs.
```

That is already the ModelSpec operator card (`feature-eval-pipeline.md` §14 steps 1→6) and the FDB-129 decision-policy shape (board → call → flip PR). Do **not** invent a third framework; pick the artifact for the lever.

---

## What differs (repo reality)

| | **Model / config promote** | **Decision-policy promote** |
|---|---|---|
| Lever examples | defcon flag, odds prior, minutes swap, lockfile | `force_play`, `DEFAULT_PLAN_HORIZON`, `DEFAULT_LIVE_CHIP_ARM` |
| Artifact | Two **ModelSpecs** + `configs/evals/<name>.yaml` | Two **policies** on the same forecasts (CLI flags / code defaults) |
| Runner | `fplt promote-eval --eval …` → compare (± components) | `fplt decisions --chip-board` (± `--no-force-play`, `--horizon`) · optional `fplt sim-season` |
| Primary metric | Forecast RMSE / slot metrics (north star §1); decisions suite **opt-in** | **Season points** (north star §2); RMSE secondary / irrelevant for the keep |
| Preflight | Orientation, one-lever, coverage, PIT | Board mean-PPG gate; tip / `--steps` match H; archive ledgers (overwrite) |
| Auto `keep` | Improvement levers stay `inconclusive` until thresholds; equivalence has identity rule | No EvalSpec `keep` — written call from boards + confirmation season (+ N6 SE when claiming chips/H) |
| Flip PR edits | Adopt treatment ModelSpec / turn flag off; rarely `PRODUCTION_MODELS` | Code constants + `configs/ops/production.yaml` + prefs (`force_play`, horizon, chip arm) |
| Precedent | W7–W9, involvement xG | FDB-129 optimised chips · this force-play × H candidate |

`suites.decisions` on an EvalSpec is the **bridge when a ModelSpec claim is about squad points** (`kind: model`, S3+). It is still the chips-off-shaped `fplt decisions` line today (P4 unfinished) — not a substitute for `--chip-board` policy A/Bs.

---

## How to choose

| Question | Path |
|---|---|
| Did **forecasts** get better/worse from a data/feature/param/component/model change? | ModelSpec EvalSpec → `promote-eval` |
| Did **season points** get better from a planner/chip/horizon default, same model? | Decision board → FDB-129 shape |
| Both? (e.g. new predictor you want live for transfers) | Forecast promote first; then decision board before claiming the squad / flipping live plan defaults |

Unifying under one *mental* checklist is right. Unifying under one *YAML runner* is not — force_play / H are not ModelSpec fields, and stuffing them into `promote-eval` would invent process the loader cannot validate.

---

## Predictions vs realised points (policy tests)

**Do not** make every optimiser/policy board plan on realised points only. Repo design already splits the jobs:

| Job | Planner input | Why |
|---|---|---|
| **N5 oracle / attainable share** | Realised pts as `xpts` | Constrained ceiling + regret; not a live policy |
| **Decision / chip / H promote boards** | Model predictions at each deadline | Same information live has; tests the policy under *our* forecasts |
| **N6 sim-SE** | Predictions to plan; drawn/sim **actuals** to score (CRN) | SE on policy diffs without giving the planner foresight |
| **Live `fplt plan`** | Predictions only | Actuals do not exist yet |

**Yes, actuals-as-xpts:** oracle, share-of-attainable, some regret components.  
**No, not for policy promote:** force_play / horizon / chip-arm boards that claim a live default — foresight turns H and packing into a different problem and overstates what production can do. North star §2 still *scores* realised season points; it does not feed them to the planner.

---

## Checklist template (either lever)

1. **Control / treatment** — one lever; production unchanged.
2. **Primary window** — run the matching artifact (EvalSpec execute **or** chip/decision board).
3. **Confirmation** — second season / window (ModelSpec: fresh eval window; decision: e.g. FDB-138).
4. **Uncertainty** — when the claim is chips / horizon / lineup: N6 sim-SE (design §7.4).
5. **Written call** — keep / drop / tune / inconclusive.
6. **Flip PR** — only after keep; update defaults + prefs + design; never chat-only.

Worked decision-policy instance: [force-play-eval-promote.md](./force-play-eval-promote.md) (H=8 + force_play=off).  
Worked ModelSpec instances: [promo-evals-consolidated.md](./promo-evals-consolidated.md).
