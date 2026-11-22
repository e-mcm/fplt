# external/

Isolated environments for **external** predictors so their deps do not pollute the main `uv` lock.

| Path | Purpose |
|---|---|
| `openfpl/` | Subprocess runner + gitignored OpenFPL checkout |

## OpenFPL

- Spec: `configs/models/openfpl_v1.yaml`
- Adapter: `fplt.models.external.openfpl`
- Ops runbook: [`docs/openfpl.md`](../docs/openfpl.md)

**Season floor:** only evaluate from **2024-25** onwards (trained through 2023-24). Opt in via `--with-openfpl` / `--openfpl-model-dir` on compare / eval-board — not in the default model list.

The large model weights live under `external/openfpl/OpenFPL/` (gitignored); first real score sparse-clones and `uv sync`s the runner.
