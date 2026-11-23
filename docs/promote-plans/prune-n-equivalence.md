# Promote plan: prune-N real-ledger equivalence

**Eval status:** **evaluated — keep** ([prune-n-equivalence-promote.md](../prune-n-equivalence-promote.md); FDB-212).  
**Path:** equivalence / decision harness · primary **week-1 action agreement**.  
**Inventory:** [unpromoted-improvements.md](../unpromoted-improvements.md) · [candidate-pruning-recommendations.md](../candidate-pruning-recommendations.md)

| | Control | Treatment |
|---|---|---|
| Policy | Reference (no prune / very large pool) | `DEFAULT_PRUNE_POLICY` (~130) |
| Metric | week-1 transfers/XI/captain/vice agree ≥95%; mean obj gap &lt;0.1 | |

### Artifact / runner
- `fplt.optimise.pruning_loss` harness on **real** prediction ledgers (not synthetic-only FDB-159).
- Equivalence claim — not forecast RMSE. Optional later EvalSpec `kind: refactor`-style if wired.

### Windows
- 20–30 squad states across 2024-25 / 2025-26 (early, DGW, rebuild, mid-season).

### Gates
- Agree rate ≥95% vs reference; mean objective gap &lt;0.1.
- If fail → retune N upward or loosen claim; do not claim lossless.

### Flip PR surface
- Only if retune changes `DEFAULT_TARGET_POOL` / quotas; document in `docs/design.md`.
- If pass → written keep of current N=130 (already live).

### Cost / risk
- Medium CPU (many MILPs). Cheaper than full N6 season CRN.

### Ordered steps
1. Stage ledgers for sample states.
2. Run pruning-loss sweep on real data.
3. Written equivalence call; retune PR only if needed.
