# Unpromoted improvements backlog

**Status:** Inventory for FDB-203 (2026-11-23). Production stays put in this task — plans only.  
**Pipeline:** [promote-pipeline-unified.md](./promote-pipeline-unified.md) · repo `docs/feature-eval-pipeline.md` · [force-play-eval-promote.md](./force-play-eval-promote.md).  
**Exclude:** optimised chips (live), `dc_layered_v2` production, `involvement_xg` soft-keep in v2, pure tooling (ensemble-sensitivity), closed hard drops with no new redesign (H=8+force_off; W7 closing **λ** prior as dropped arm).

**Eval status:** `never` · `descriptive` (bakeoff, no keep) · `evaluated` (written call) · **`in flight`** (worker launched — do not duplicate).

---

## Eval in flight (do not relaunch)

| Lever | Worker / task | Plan (reference only) |
|---|---|---|
| De-vig 1X2 prior revisit | FDB-207 · `bc-79047462-cc54-5d09-8a3a-53e549bc0384` | [promote-plans/devig-1x2-prior.md](./promote-plans/devig-1x2-prior.md) |
| Availability overlays (injury + predicted XI) | FDB-208 · `bc-be9d4b44-bb4c-5547-9b90-78724e58b4e6` | [promote-plans/minutes-injury-overlay.md](./promote-plans/minutes-injury-overlay.md) · [promote-plans/predicted-xi-override.md](./promote-plans/predicted-xi-override.md) |

---

## Remaining never / descriptive — still need evaluate-and-promote (not in flight)

None currently queued outside in-flight / evaluated rows.

---

## Evaluated — kept / not kept / open / blocked (no new worker unless noted)

| Lever | Eval status | Verdict | Next |
|---|---|---|---|
| Forecast combine (`forecast_combine_equal_v1`) | evaluated | **Drop** — confirm RMSE fail; H=3 2024-25 −82 vs DC. [forecast-combine-promote.md](./forecast-combine-promote.md) | skip PRODUCTION; keep #156 prune |
| Within-horizon price forecast (`mean_gw_delta`) | evaluated | **Keep (provisional)** — 2025-26 **+29**, 2024-25 **+9**; flip live `run_plan` default. [price-within-horizon-promote.md](./price-within-horizon-promote.md) | done (FDB-218); provisional until FDB-73 |
| As-of 50% sell-on (#124) | canceled board | Bug fix, not promote (Eamon) | already live |
| Prune N (`DEFAULT_TARGET_POOL≈130`) | evaluated | **Keep** — real-ledger week-1 agree **24/24**, gap **0**; already live; no retune PR. [prune-n-equivalence-promote.md](./prune-n-equivalence-promote.md) | done (FDB-212) |
| `lineup_reopt` / EV default | evaluated | **Keep** — default already True; no flip. 2025-26 **+130**, 2024-25 **0**. [lineup-reopt-promote.md](./lineup-reopt-promote.md) | optional SAA arm later |
| History rating prior (`dc_layered_history_v1`) | evaluated | **Tune** @ 1.0 — no PRODUCTION flip. [history-market-prior-promote.md](./history-market-prior-promote.md) | optional dose grid |
| Defcon P(award)/rate (`defcon_mode: rate`) | evaluated | **Keep (provisional)** — flipped into `dc_layered_v2`. [defcon-rate-promote.md](./defcon-rate-promote.md) | done (FDB-211); provisional until FDB-73 |
| CS-only λ (`dc_layered_cs_v1` → `dc_layered_v4`) | evaluated | **Keep** — flipped into `dc_layered_v4` (`market_cs_strength=0.5`). [cs-only-player-rmse-promote.md](./cs-only-player-rmse-promote.md). Opt-out `dc_layered_nocs_v1`. | done (FDB-267) |
| LGBM production flip | evaluated (bakeoff) | Ranking companion only | **skip** PRODUCTION |
| W7 λ prior (`dc_layered_odds_v1`) | evaluated | Drop (closing ≠ PIT) | **skip** / closed |
| 1X2-region blend (`dc_layered_1x2_v1`) | evaluated; **revisit in flight** FDB-207 | Prior: do not promote | defer to FDB-207 |
| W8 `penalties_order` | evaluated | Inconclusive (vaastav null) | **blocked** (snapshot order) |
| OpenFPL / Understat Deep+PPDA | never full | FDB-55 | **blocked** (Understat) |
| FPL Review Tier B | never | Ingest ready | **blocked** (Patreon CSVs) |
| N6 full chip/H CRN | aborted | ETA 10–40h killed | **skip** as first step |

---

## Closed / excluded

| Item | Why |
|---|---|
| H=8 + force_play=off | 2024-25 H2H **−13**. Prefs: H=3 + force-on. [force-play-eval-promote.md](./force-play-eval-promote.md) |
| Optimised chips | Promoted live |
| `involvement_xg` in v2 | Soft keep |
| Ensemble-sensitivity | Tooling only |

---

## Production snapshot

| Knob | Live |
|---|---|
| Model | `dc_layered_v2` |
| H / force_play / chips | **3** / **on** / **optimised** |
| `lineup_reopt` | **True** (**keep**; [lineup-reopt-promote.md](./lineup-reopt-promote.md)) |
| Within-horizon prices | **`mean_gw_delta`** on live `run_plan` (**keep**; [price-within-horizon-promote.md](./price-within-horizon-promote.md)) |
| Prune N | ~**130** (**keep**; [prune-n-equivalence-promote.md](./prune-n-equivalence-promote.md)) |
| Market strengths | off |

---

## Priority for *new* launches (after in-flight)

### P2
1. ~~CS-only λ **flip PR** (`dc_layered_v4`)~~ — done (FDB-267); [cs-only-player-rmse-promote.md](./cs-only-player-rmse-promote.md)  
2. OpenFPL / Understat / Review / W8 — **blocked on user/data**

**Blocked pending user/data:** Understat Deep/PPDA; FPL Review Patreon archives; snapshot `penalties_order` for W8; API-Football/Touchline for FDB-208; `FPLT_ENTRY_ID` only for live-squad paths.

**Closed this pass:** forecast-combine → **drop** ([forecast-combine-promote.md](./forecast-combine-promote.md)).
