# Eval protocol

Standing operator rules that sit beside [`feature-eval-pipeline.md`](feature-eval-pipeline.md) (ModelSpec promote-eval) and the Project evaluation north star. Windows, interim keep, reference chip-board, and HiGHS determinism land under FDB-225 / FDB-228; this file owns the **decisions-suite gate** (FDB-227 / item 4).

---

## Decisions suite gate (item 4)

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

**Not this gate:** decision-**policy** levers (`force_play`, `DEFAULT_PLAN_HORIZON`, chip arm, lineup reopt, sell-on). Those are chip-board / `fplt sim-season` A/Bs, not ModelSpec EvalSpecs — see Project `docs/promote-pipeline-unified.md`. Promote-eval’s decisions suite remains the chips-off-shaped bridge for ModelSpec season-points claims (P4).

**Wiring:** scaffold `configs/evals/example_model_squad_points.yaml`. Existing real EvalSpecs on main are forecast claims (`decisions: false`); none asserted season points, so none were flipped.

Canonical detail: [`feature-eval-pipeline.md`](feature-eval-pipeline.md) §4.2 and §8.

---

## Other protocol items

| Item | Owner | Status |
|---|---|---|
| 1 Windows + interim keep | FDB-225 | Protocol PR |
| 2 Reference board config | FDB-225 | Protocol PR |
| 3 Production model rename (`dc_layered_v3`) | FDB-226 | Sibling |
| 4 Decisions suite gate | **FDB-227 (this doc § above)** | Landed in loader |
| 5 Deterministic HiGHS | FDB-228 (+ FDB-225) | Sibling |
| 6 North-star doc fixes | FDB-225 | Protocol PR |
