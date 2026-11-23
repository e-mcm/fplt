# Price dynamics / 50% sell-on — promote board aborted

**Status:** **Aborted** (2026-11-23). Out of scope for decision-policy promote.

As-of FPL 50% sell-on ([#124](https://github.com/e-mcm/fplt/pull/124)) is a **bug fix** (correct selling prices vs frozen-at-purchase), not a forecast/policy lever to keep/drop via season-points A/B.

| | |
|---|---|
| Planned board | Treatment (live sell-on) vs control (frozen / pre-#124) on H=3 optimised force-on |
| Outcome | **Stopped** — no keep / drop / tune call; **no flip PR** |
| Ablation | `--freeze-selling` work abandoned locally; **not** pushed / opened |

Partial 2025-26 cells under `internal/price-dynamics-sell-on/` are archival scratch only and are **not** evidence for a promote gate.

Related plan (superseded): [promote-plans/price-dynamics-sell-on.md](./promote-plans/price-dynamics-sell-on.md). Within-horizon rise forecast remains a separate redesign item: [promote-plans/price-within-horizon.md](./promote-plans/price-within-horizon.md).
