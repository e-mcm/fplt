# Promote plan: predicted-XI minutes override

**Eval status:** **in flight** — FDB-208 · agent `bc-be9d4b44-bb4c-5547-9b90-78724e58b4e6` (availability overlays). **Do not duplicate.**  
**Path:** ModelSpec · primary **RMSE** · minutes Brier / start accuracy gate.  
**Inventory:** [unpromoted-improvements.md](../unpromoted-improvements.md)

| | Control | Treatment |
|---|---|---|
| Spec | `dc_layered_v2` | `dc_layered_predicted_xi_v1` (`predicted_xi: true`) |

### Artifact / runner
- EvalSpec `configs/evals/predicted_xi_ab.yaml` (scaffold in repo PR).
- `uv run fplt promote-eval --eval predicted_xi_ab --execute`
- Preflight: `fplt predicted-xi-audit` / `score_predicted_starts` on Touchline/API-Football lineups vs official XIs — **abort promote** if start accuracy is below the audit gate (default 0.55).
- Enable ingest: `FPLT_TOUCHLINE_LINEUPS_ENABLED=1` (or API-Football lineups).

### Windows
- Primary: 2025-26 GWs with staged `predicted_lineups` before deadline.
- Confirmation: later GWs / second season slice.

### Gates
- Start-accuracy smoke passes (document threshold in recommendation; do not promote on broken feed).
- RMSE headline ≤ control on decision pool; Brier ≥60 not worse.
- Coverage: predicted_lineups join on fixtures in window (manual audit if no coverage metric yet).

### Flip PR surface
- Merge treatment params into `dc_layered_v2` minutes **or** set ops model to `dc_layered_predicted_xi_v1`.
- Keep flag/docs: production SoR only after audit.

### Cost / risk
- Medium ops (feed quality). Touchline observed wrong-club noise — largest promote risk.

### Ordered steps
1. Stage lineups; run `score_predicted_starts`.
2. If accuracy fail → **stop** (redesign source), no promote-eval claim.
3. Execute EvalSpec → written call.
4. Confirm → flip PR.
