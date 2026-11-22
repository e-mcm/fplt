# data/

Default `$FPLT_DATA_DIR` (gitignored contents). Layout and table names are defined in [`docs/design.md`](../docs/design.md) §4.3 — do not invent a parallel schema here.

## Invariants

- **`raw/` is write-once.** Never modify or delete raw captures in library code.
- Derived trees (`staged/`, `features/`, `artifacts/`, `predictions/`, `plans/`, ledger, …) are rebuildable.
- Paths come from `fplt.settings` — never hardcode `./data` in new code.

Bootstrap historical seasons: `make bootstrap-vaastav`. Capture cadence: [`docs/dagster-cadence-and-setup.md`](../docs/dagster-cadence-and-setup.md).
