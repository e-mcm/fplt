# tests/

Everything that lands on `main` is exercised here. Tests **never** touch the network — enforced with `pytest-socket`, randomised order (`pytest-randomly`; reproduce with `--randomly-seed=<n>`), per-test timeouts (`pytest-timeout`), and Hypothesis `ci`/`dev` profiles (`HYPOTHESIS_PROFILE`). See `AGENTS.md` §3.

| Path | Role |
|---|---|
| `unit/` | Library behaviour, edge cases, error paths |
| `contracts/` | `PointsPredictor` / component contracts **incl. leakage** |
| `integration/` | End-to-end wiring / library integration |
| `fixtures/` | Recorded gzipped API JSON (and CSV) per source |
| `builders.py` | Synthetic polars frames for unit tests |

## Commands

```bash
make check   # ruff + pyright + fast tests (`not slow`) + diff coverage
make test    # full suite + coverage gates
```

## Invariants

- Bug → regression test in the same PR (or before the fix).
- No quality assertions (MAE/RMSE/log-loss) in pytest — that lives in the backtester / ledger.
- Mark >~2s tests with `@pytest.mark.slow`.
- Optimiser tests: constraint satisfaction on small known problems, not optimality on real data.
- New predictors must be registered so the parametrised contract suite picks them up.

## Agents

- Coverage floors: diff ≥95%, overall line+branch ≥90% (`AGENTS.md` §3).
- Prefer fixtures under `tests/fixtures/<source>/` over inventing one-off JSON in the test body when the payload is large.
