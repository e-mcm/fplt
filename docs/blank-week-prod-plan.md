# Blank-week / empty-squad plan (production model)

How agents (and humans) draft an **initial fifteen** or **wildcard-from-empty** plan with the **live production** model — not bakeoff forks.

## Production model

| Source of truth | Today |
|---|---|
| `configs/ops/production.yaml` → `models:` | **`dc_layered_v4`** |
| Env override | `FPLT_PRODUCTION_MODELS` (comma-separated; env wins) |
| Import mirror | `PRODUCTION_MODELS` in `fplt.automation.cycle` (prefer `load_production_ops()`) |

**Always pass `--model` explicitly** (or `MODEL=` on `make plan`) using the first ops entry — do not rely on stale Makefile defaults or experimental specs.

```bash
# Resolve the live name without guessing:
uv run python -c "from fplt.automation.production import load_production_ops; print(load_production_ops().models[0])"
```

## Live planner defaults (leave alone unless ablating)

| Knob | Default | Ablation flag |
|---|---|---|
| Horizon | **3** (`DEFAULT_PLAN_HORIZON` / ops `plan_horizon`) | `--horizon N` / `FPLT_PLAN_HORIZON` |
| Chip arm | **optimised** (`DEFAULT_LIVE_CHIP_ARM`) | `--chip-arm off\|rule_based\|optimised` |
| Force-play chips | **on** | `--no-force-play` |
| Price forecast | **mean_gw_delta** | (library / boards only; live `run_plan` default) |
| Lineup re-opt | **on** | (library `lineup_reopt=False`; no CLI off-switch today) |

## Blank / empty squad (no `FPLT_ENTRY_ID`)

`configs/squad_state.yaml` already ships with `owned: []` and `bank: 1000` (£100.0m). That is an **initial draft**.

1. **Unset** `FPLT_ENTRY_ID` (otherwise `resolve_squad_state` loads the live entry).
2. Keep `owned: []` (or clear it). Chip inventory stays in the YAML.
3. Do **not** pass `--from-api` without an entry / `--entry-json`.
4. `--rebuild` is for **WC-from-owned**: it clears owned and sets `bank = ITB + Σ selling_price`. It needs selling prices from authenticated `my-team` or `--entry-json`. Unnecessary when already empty.

```bash
unset FPLT_ENTRY_ID
# optional: ensure owned is empty
#   owned: []   bank: 1000   in configs/squad_state.yaml

MODEL=$(uv run python -c "from fplt.automation.production import load_production_ops; print(load_production_ops().models[0])")
SEASON=2026-27   # or your season
GW=12            # next deadline; omit --gw on predict to use resolve_as_of

# 1) Fit / score into the ledger (required before plan)
uv run fplt predict --model "$MODEL" --season "$SEASON" --gw "$GW" --horizon 3

# 2) Draft fifteen + H-step transfers/chips
uv run fplt plan --model "$MODEL" --season "$SEASON" --gw "$GW" --horizon 3
# same via make (MODEL must be production):
# make plan SEASON=$SEASON GW=$GW MODEL=$MODEL HORIZON=3

# Report:
#   $FPLT_DATA_DIR/plans/<model_id>/as_of=<season>_gwNN_h03/report.md
```

CLI will log `planning from YAML squad (no FPLT_ENTRY_ID)` and `initial=True` / `owned=0`.

## WC-from-existing (rebuild)

```bash
# Needs FPLT_ENTRY_ID + FPLT_FPL_API_TOKEN (my-team selling prices) OR --entry-json
uv run fplt plan --model "$MODEL" --season "$SEASON" --gw "$GW" --horizon 3 --rebuild
# or: --from-api --rebuild
```

## Data prerequisites

Staged FPL tables + enough history for the production stack (vaastav + `snapshots` ingest). Typical one-shot:

```bash
uv run python -c "
from pathlib import Path
from fplt.data.snapshots import fetch_snapshots
from fplt.data.store import DataStore
from fplt.data.raw import ingest_raw
fetch_snapshots(Path('data/raw'), remote='origin')  # or https://github.com/e-mcm/fplt.git
ingest_raw(DataStore(Path('data')), Path('data/raw'))
"
make bootstrap-vaastav   # once / when seasons missing
```

Or `uv run fplt ops-catchup` when the daemon was down (also plans missing checked GWs for production models).

## Do not

- Point `--model` at bakeoff forks (`dc_layered_cs_v1`, history/odds arms, `lgbm_direct_v1`, …) for a “prod” blank week.
- Leave `FPLT_ENTRY_ID` set and expect a blank draft.
- Assert model quality in pytest; this path is decision output only.

See also: [`design.md`](design.md) §10 (optimiser / squad state), [`dagster-cadence-and-setup.md`](dagster-cadence-and-setup.md) (ops models + sensors), `configs/ops/production.yaml`.
