# configs/

YAML that drives models, scoring, chips, evals, and live ops. **No hyperparameters in source** — change behaviour here (or via env overrides documented below).

| Path | Purpose | Entrypoint / loader |
|---|---|---|
| `models/*.yaml` | `ModelSpec` per registered kind | `fplt.models.spec` / `@register_model` |
| `scoring/<season>.yaml` | FPL points rules for that season | `fplt.scoring.Rules.for_season` |
| `chips/<season>.yaml` | Chip grants / windows | `fplt.scoring.chips` |
| `ops/production.yaml` | Live unattended models + plan horizon | `fplt.automation.production.load_production_ops` |
| `evals/*.yaml` | Promote-eval / bakeoff specs | `fplt promote-eval` / `fplt bakeoff` |
| `squad_state.yaml` | Local squad / banking / bench knobs | Optimiser / `fplt plan` |
| `understat/`, `api_football/`, `football_data/` | External id / name maps | See each subdir README |

## Invariants

- New model → add YAML under `models/` and register the kind; do not hardcode params in Python.
- New season → add `scoring/<season>.yaml` (and chip rules when needed) before backtesting that season.
- Live list stays **short**: edit `ops/production.yaml` or set `FPLT_PRODUCTION_MODELS` / `FPLT_PLAN_HORIZON` (env wins). Never silently multi-arm live.
- CLI / `make plan` still use `DEFAULT_PLAN_HORIZON` in `fplt.optimise.plan`; ops YAML feeds catch-up / pre-deadline only.
- Map files are reviewed in git; unmapped external ids stage with null `player_key` / `team_key`.

## Agents

- Architecture: [`docs/design.md`](../docs/design.md) §5 (models), §6 (scoring), §7–8 (eval / ops).
- Eval workflow: [`docs/feature-eval-pipeline.md`](../docs/feature-eval-pipeline.md).
- External map maintenance: READMEs in `understat/`, `api_football/`, `football_data/`.
