# Feature / data / config / component eval pipeline

Standard A/B path for **any** model lever: new staged data, feature version, hyperparam fork, layered **component registry swap** (e.g. minutes LightGBM → handwritten), or full **predictor fork** (e.g. `lgbm_direct` → XGBoost).

Everything is a **control vs treatment `ModelSpec`** plus an EvalSpec under `configs/evals/`.

## Quick start (scaffold)

```bash
# Dry-run: print suites + commands (no fits)
uv run fplt promote-eval --eval example_model_lgbm_vs_rolling --dry-run
make promote-eval EVAL=example_component_minutes_v1_vs_v2 ARGS='--dry-run --tier S1'
```

Example EvalSpecs:

| File | `kind` / `lever` | Shows |
|---|---|---|
| `example_data_odds_prior.yaml` | `data` / `data_table` | New staged table consumer (W7 shape) |
| `example_component_minutes_v1_vs_v2.yaml` | `component` / `component_swap` | Layered slot swap |
| `example_model_lgbm_vs_rolling.yaml` | `model` / `model_fork` | Top-level predictor fork |
| `example_config_dc_v1_vs_v2.yaml` | `config` / `hyperparam` | Params-only fork shape |

Execute path (compare / eval-components / decisions orchestration) lands in later PRs. Until then, run the printed commands by hand.

## Design

Authoritative design (UX, gates, W7–W9 migration, phases P0–P6):

- Project store: `docs/feature-eval-pipeline.md` on the fplt design-build agent store
- Repo rules: `docs/design.md` §7.7; hard constraints in `AGENTS.md`

## Rules of thumb

1. Fork a **new** treatment YAML — never edit the control default in place.
2. One lever per EvalSpec (`change.diff` lists it).
3. New `kind` → `register_model` / `register_component` + contract **leakage** tests before eval.
4. `kind: component` → components suite first, then prediction.
5. Recommendation ≠ production promote (`PRODUCTION_MODELS` is a separate PR).
