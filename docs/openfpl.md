# OpenFPL ops runbook

[OpenFPL](https://github.com/daniegr/OpenFPL) (MIT) is an external points
benchmark. `fplt` calls it through `fplt.models.external.openfpl` as a
subprocess under `external/openfpl/`. Spec: `configs/models/openfpl_v1.yaml`.

OpenFPL is **opt-in** on `make compare-models` / `make eval-board` (not in
`CORE_MODELS`). Include it with `--with-openfpl`, `--openfpl-model-dir`, or a
non-empty YAML `params.model_dir` — and only for seasons from **2024-25**
onwards.

## Setup

1. Sync the isolated runner env (joblib / pandas / sklearn / xgboost). The
   adapter prefers `external/openfpl/.venv/bin/python` when this exists:

   ```bash
   cd external/openfpl && uv sync
   ```

2. Clone OpenFPL somewhere outside this repo (it ships the fitted `models/`
   ensemble). Point the YAML at that clone when the feature matrix is ready
   (see **Feature matrix gap** below):

   ```yaml
   # configs/models/openfpl_v1.yaml
   params:
     compare_from: "2024-25"
     model_dir: "/absolute/path/to/OpenFPL"   # must contain models/
     timeout_seconds: 120
   ```

   `model_dir` must contain `models/xscaler.save` (and the rest of the
   published ensemble). A relative path is fine if your working directory is
   stable; an absolute path is clearer for Dagster / launchd.

3. Optionally override for one compare / eval run without editing the YAML:

   ```bash
   make eval-board SEASON=2024-25 ARGS='--openfpl-model-dir /path/to/OpenFPL'
   make compare-models SEASON=2024-25 ARGS='--with-openfpl --openfpl-model-dir /path/to/OpenFPL'
   # or: uv run fplt backtest --season 2024-25 --models openfpl_v1
   # or: uv run fplt compare --season 2024-25 --with-openfpl
   ```

## Season floor

OpenFPL was trained on **2020-21 through 2023-24**. Only evaluate on
**2024-25 onwards**. The adapter raises if `as_of` or a target gameweek is
earlier (`compare_from` in the YAML, default `2024-25`).

## Empty `model_dir` ⇒ zeros

With `params.model_dir: ""` (the checked-in default), the runner still starts
and returns **0.0 for every row**. That keeps the contract suite offline; those
zeros are **not** OpenFPL scores. Do not put an empty-dir run on the
leaderboard as a real benchmark.

## Feature matrix gap

Real OpenFPL inference needs the wide sample columns expected by
`models/xscaler.save` (~228 features: rolling FPL stats, Understat/xG-style
fields, opponent context, availability, … — the same schema as OpenFPL’s
`data/samples.csv`). The fplt adapter currently sends only identity fields
(`season`, `gw`, `position`, `openfpl_player_id`). Until a feature builder
lands (rebuild from PIT/Understat, or call OpenFPL’s own pipeline), a
non-empty `model_dir` that contains the published ensemble **raises** rather
than inventing scores. Keep OpenFPL out of the default compare/eval set.

## Smoke

```bash
# Offline zeros only (empty model_dir) — contracts / wiring, not a baseline
uv run fplt backtest --season 2024-25 --from-gw 1 --to-gw 1 --models openfpl_v1

# Opt into the compare / eval set (still zeros until features exist)
make eval-board SEASON=2024-25 ARGS='--with-openfpl --from-gw 1 --to-gw 1'
make compare-models SEASON=2024-25 ARGS='--with-openfpl --test-from-gw 1 --test-to-gw 1'
make leaderboard
```

Architecture notes: [`docs/design.md`](design.md) §7.4.
