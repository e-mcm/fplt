# Promote result: prune-N real-ledger equivalence (`DEFAULT_TARGET_POOL=130`)

**Call: keep** — week-1 actions match the full-pool reference on **≥95%** of states with mean objective gap **&lt;0.1**; **N≈130 already live**; **no retune / flip PR**.  
**Plan:** [promote-plans/prune-n-equivalence.md](./promote-plans/prune-n-equivalence.md) · **Prior synthetic:** Project store `docs/candidate-pruning-recommendations.md` (see [promote-archives.md](./promote-archives.md)) · FDB-159  
**Artifacts:** Project store `internal/prune-n-equivalence/` (see [promote-archives.md](./promote-archives.md)) · tip `af1d32b` · model `dc_layered_v2` (`b12aad1575ff0d34`)

| | Reference (control) | Treatment (live default) |
|---|---|---|
| Pool | No prune (~700 candidates) | `DEFAULT_PRUNE_POLICY` / `n=130` (~150 after dominance) |
| Metric | week-1 transfers / XI / captain / vice | agree ≥95%; mean obj gap &lt;0.1 |
| Horizon | H=5 (prune auto-applies) | same |
| Chips / lineup_reopt | off / False (transfer-MILP equivalence) | same |

---

## Boards (week-1 action equivalence)

24 squad states (12 per season × early / DGW-runup / rebuild / mid-banked FTs). Ownership reconstructed from archived decision boards; xPts from real prediction ledgers.

| Arm | n | Agree rate | Mean obj gap | Mean pool |
|---|---:|---:|---:|---:|
| **n=130** / **current** | 24 | **1.000** | **0.000** | 149.0 |
| n=100 | 24 | 1.000 | 0.000 | 131.9 |
| n=80 | 24 | 0.958 | 0.000 | 116.4 |
| n=160 / 200 / 300 | 24 | 0.958 | 0.000 | 160–176 |
| reference | 24 | 1.000 | 0.000 | 729.3 |

### By season

| Season | n=130 agree | Mean gap | Notes |
|---|---:|---:|---|
| **2025-26** | 12/12 (1.000) | 0.000 | All sweep arms 100% |
| **2024-25** | 12/12 (1.000) | 0.000 | n=80 and n≥160 each missed 1 state with **gap 0** (alternate optima) |

All MILP statuses `kOptimal` within time limits (ref 180s / sweep 90s). Disagreements on non-130 arms had **objective gap 0** — different week-1 optima at the same objective, not pruning loss.

---

## Call

**Keep `DEFAULT_TARGET_POOL = 130`** (and the scaled quotas already shipped in FDB-159). Gates pass on real ledgers for both 2024-25 and 2025-26. Do **not** open a prune-defaults PR. Do **not** flip `PRODUCTION_MODELS`.

**Not claimed:** season-points impact of prune; chips-on boards; H&gt;5; losslessness of N&lt;100 (n=80 is 95.8% here but one rebuild miss — leave at 130).

---

## Commands / reproduce

```bash
# Stage preds (2025-26 shared pack + fresh 2024-25 backtest --steps 5)
# Then:
uv run python internal/prune-n-equivalence/run_real_pruning_loss.py \
  --data-dir "$FPLT_DATA_DIR" --seasons 2024-25 2025-26 \
  --n-scenarios 24 --horizon 5 \
  --reference-time-limit 180 --sweep-time-limit 90 \
  --output-dir internal/prune-n-equivalence/combined-run
```

Library harness: `fplt.optimise.pruning_loss`. Rollup: `internal/prune-n-equivalence/combined/equivalence_call.json`.
