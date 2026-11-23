# Live prediction regen (FDB-221)

**Tip SHA:** `35b9ac5` (`main` — includes #165 `mean_gw_delta` live default; defcon rate already in `dc_layered_v2`)  
**Model:** `dc_layered_v2` · **model_id:** `12152f0e53162d2f`  
**Season / GW:** `2026-27` GW **12** (next deadline Fri 27 Nov 2026 18:30 UTC; GW11 current, not `data_checked`)  
**Data:** `$FPLT_DATA_DIR=/workspace/data` — cleared `ledger/predictions` before regen (same contamination note as history-prior: stale GW rows)

## What ran

1. Cleared `ledger/predictions` (empty after bootstrap; tree removed before predict).
2. `fplt predict --model dc_layered_v2 --season 2026-27 --gw 12 --horizon 3` → 2022 rows; ledger total 2022.
3. One forward plan with live defaults: H=3, `chip_arm=optimised`, `force_play=True`, `lineup_reopt=True`, `price_forecast=mean_gw_delta`.

**Plan summary:**  
`/workspace/data/plans/12152f0e53162d2f/as_of=2026-27_gw12_h03/report.md`  
(durable copy: `internal/live-prediction-regen/plan-report-2026-27-gw12.md`)

## Plan snapshot (empty squad — no `FPLT_ENTRY_ID` on this VM)

| | |
|---|---|
| Horizon obj (decayed) | **199.63** |
| Chips H=3 | **3xc** (GW12 forced), freehit, bboost |
| Captain (VC) | **Groß** (McGinn) |
| XI | Trafford · De Cuyper · Vuskovic · Gabriel · Calafiori · **Groß (C)** · McGinn (V) · B.Fernandes · Gibbs-White · E.Le Fée · N.Jackson |
| Bench | Raya · Tarkowski · Calvert-Lewin · Evanilson |

Promote posture: forecast combine **dropped**; decision flips already on (`mean_gw_delta` #165, `lineup_reopt` default). Ready for **FDB-222** explainable next-GW write-up (use entry auth / live XV there — this regen used empty £100m draft).
