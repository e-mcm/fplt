# OpenFPL ops runbook

[OpenFPL](https://github.com/daniegr/OpenFPL) (MIT) is an external points
benchmark. `fplt` calls it through `fplt.models.external.openfpl` as a
subprocess under `external/openfpl/`. Spec: `configs/models/openfpl_v1.yaml`.

OpenFPL is **opt-in** on `make compare-models` / `make eval-board` (not in
`CORE_MODELS`). Include it with `--with-openfpl` or `--openfpl-model-dir`.
A YAML `model_dir` other than the default checkout also opts it in. Only
seasons from **2024-25** onwards are valid.

## Setup

`configs/models/openfpl_v1.yaml` sets `params.model_dir` to
`external/openfpl/OpenFPL`. That path is gitignored. The first
`fplt predict --model openfpl_v1` (or any real score) sparse-clones
[daniegr/OpenFPL](https://github.com/daniegr/OpenFPL) into it and runs
`uv sync` in `external/openfpl/` when the runner venv is missing.

The ensemble is about 750MB, and some files approach GitHub's 100MB limit,
so it is not committed. `model_dir` must end up containing
`models/xscaler.save`. An empty `model_dir`, or a directory without that
file, raises. The runner does not fill in zeros.

Override the checkout for one compare / eval run without editing the YAML:

```bash
make eval-board SEASON=2024-25 ARGS='--with-openfpl --from-gw 1 --to-gw 1'
make compare-models SEASON=2024-25 ARGS='--with-openfpl'
# or: uv run fplt predict --model openfpl_v1 --season 2024-25 --gw 1 --horizon 1
```

## Season floor

OpenFPL was trained on **2020-21 through 2023-24**. Only evaluate on
**2024-25 onwards**. The adapter raises if `as_of` or a target gameweek is
earlier (`compare_from` in the YAML, default `2024-25`).

## Feature matrix

`fplt.models.external.openfpl_samples` builds the 228 OpenFPL xscaler columns
from PIT FPL + Understat tables and checks names via `OPENFPL_FEATURE_NAMES`.
The adapter attaches that map as each runner row's `features` object.
Team/opponent **Deep** and **PPDA** are still 0.0 (those metrics are on
Understat match HTML, not `getMatchData`). Re-stage after the ICT/Understat
widen before a real `model_dir` smoke. Keep OpenFPL out of the default
compare/eval set until that smoke looks sane.

## Smoke

```bash
uv run fplt predict --model openfpl_v1 --season 2024-25 --gw 1 --horizon 1
make eval-board SEASON=2024-25 ARGS='--with-openfpl --from-gw 1 --to-gw 1'
make compare-models SEASON=2024-25 ARGS='--with-openfpl --test-from-gw 1 --test-to-gw 1'
make leaderboard
```

`fplt plan` raises when the ledger has no rows for that model and deadline,
when a horizon week is missing or all zero, or when a week covers fewer than
half the players. Predict before planning. Live predict also refuses to
record an all-zero frame.

Architecture notes: [`docs/design.md`](design.md) §7.4.
