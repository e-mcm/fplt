# Promote plan: price dynamics / 50% sell-on (#124)

**Eval status:** **Aborted** — as-of 50% sell-on is a **bug fix**, not a promote A/B. See [price-dynamics-sell-on-promote.md](../price-dynamics-sell-on-promote.md).  
**Path:** ~~Decision-policy · season points~~ (withdrawn).  
**Pipeline:** [promote-pipeline-unified.md](../promote-pipeline-unified.md).  
**Task:** FDB-205 (stopped).

| | Control (frozen / pre-#124) | Treatment (live sell-on) |
|---|---|---|
| Selling prices | Sticky at purchase; no as-of refresh | FPL 50% sell-on (`fpl_selling_price`); refresh each deadline from as-of `player_status` / candidates |
| Toggle | `freeze_selling=True` / CLI `--freeze-selling` | default (`freeze_selling=False`) |
| Forecasts | `dc_layered_v2` | same |
| Chips / H / force | optimised · **H=3** · **force_play on** | same |

### Where it lives (repo `main`)

| Path | Behaviour |
|---|---|
| `fplt.optimise.prices` | `fpl_selling_price`, `apply_market_prices`, `refresh_owned_selling_prices` |
| `simulate_decisions` | Calls `apply_market_prices` before each `plan_squad`; `_advance` sticks purchase and recomputes selling |
| `plan_squad` | Owned sell credits = `fpl_selling_price(purchase, list_price)` |
| `run_plan` / `optimize_transfers` | Same as-of refresh before solve |
| `OwnedPlayer.purchase_price` | Sticky; YAML defaults purchase to selling when omitted |

There is **no production off-switch** after #124 — sell-on is always on. Control requires the ablation flag above (same pattern as `--no-force-play` / lineup_reopt A/B).

### Artifact / runner

```bash
# Treatment (live default)
make chip-decision-board SEASON=2025-26 HORIZON=3 MODELS='dc_layered_v2' \
  ARGS='--chip-arms optimised'

# Control (frozen)
make chip-decision-board SEASON=2025-26 HORIZON=3 MODELS='dc_layered_v2' \
  ARGS='--chip-arms optimised --freeze-selling'
```

Archive `ledger/decisions` after each cell (overwrite risk). Do **not** use `fplt promote-eval` (ModelSpec path).

### Windows

| # | Case | Season | Settings |
|---|---|---|---|
| **A** | Primary H2H | **2025-26** | optimised · H=3 · force_play=on · treatment vs control |
| **B** | Confirmation | **2024-25** | same settings; true `ChipRules.for_season("2024-25")` |

### Gates

| Gate | Pass if | Fail if |
|---|---|---|
| **A — 2025-26 H2H** | Treatment season pts **≥** control | Treatment loses |
| **B — 2024-25 H2H** | Treatment ≥ control | Treatment loses either season |
| **Mean-PPG** | Board exits 0 under default `--max-mean-ppg 90` | Gate trip |
| **N6 sim-SE** | Only if \|mean_diff\| is tiny and coordinator approves (prefer cheaper boards) | Do not open multi-hour N6 first |

**Minimum bar for written keep:** A + B both pass.  
**Flip PR:** only after keep on **both** seasons. Because sell-on is already live:

- **Keep** → document promote; **no** code default flip (already on). Optional prefs / inventory note.
- **Drop** → flip PR restoring frozen selling as default (`freeze_selling=True` path or revert refresh).
- **Tune** → document; no silent default change.

### Cost / risk

- ~one optimised H=3 chip-board wall time ×2 arms ×2 seasons (~4–8 min/arm historically).
- Honest gap unchanged: within-horizon steps still do **not** forecast future rises ([price-within-horizon.md](./price-within-horizon.md)).
- Shared preds: `internal/h8-shared-data` (2025-26); 2024-25 needs staged + `dc_layered_v2` ledger with `--steps ≥ 3`.

### Ordered steps

1. Land thin `--freeze-selling` / `freeze_selling=` on chip-board + `simulate_decisions` / `plan_squad` / `_advance` (tests).
2. Write this plan (done).
3. Run Case A 2025-26 → archive under `internal/price-dynamics-sell-on/`.
4. Run Case B 2024-25 confirmation → archive.
5. Write results + call → [`docs/price-dynamics-sell-on-promote.md`](../price-dynamics-sell-on-promote.md).
6. Flip PR **only** after keep on both (drop/tune may open a revert or prefs PR; keep = docs only).
