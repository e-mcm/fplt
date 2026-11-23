# Promote results registry

Committed index of written promote calls — not a second production list.

| File | Role |
|---|---|
| `registry.yaml` | Verdict, EvalSpec (or decision-policy lever), windows, audit paths, data hashes |
| `<id>/<role>/audit.json` | Reproducibility record copied from the promote run |

Load with `fplt.backtest.results_registry.load_results_registry()`.
Mark provisional keeps with `provisional: true` + `provisional_reason`.
