# Confirm: within-horizon `mean_gw_delta` price forecast

**Task:** FDB-256  
**Verdict: confirm keep** — provisional cleared.  
**Lever:** live default `DEFAULT_LIVE_PRICE_FORECAST_METHOD = mean_gw_delta` (flip [#165](https://github.com/e-mcm/fplt/pull/165)).  
**Prior promote:** [`price-within-horizon-promote.md`](./price-within-horizon-promote.md) · plan [`promote-plans/price-within-horizon.md`](./promote-plans/price-within-horizon.md)  
**Protocol:** [`eval-protocol.md`](./eval-protocol.md) §1–§2.3 · §4 reference board · §5 deterministic HiGHS  
**Archive:** Project store `internal/price-within-horizon/` (see [promote-archives.md](./promote-archives.md)) · tip `f29a5a6` (board flag [#163](https://github.com/e-mcm/fplt/pull/163))

---

## Windows (declared before re-read)

Decision-policy shape ([`eval-protocol.md`](./eval-protocol.md) §2.3): primary season board → independent confirmation season. One flag flip (`price_forecast`).

| Role | Season | GW | Notes |
|---|---|---|---|
| **Development / primary** | `2025-26` | 1–38 | Full chip board; dose fixed as binary flat vs `mean_gw_delta` (no dose search on confirm). |
| **Confirmation** | `2024-25` | 1–38 | Independent season — non-overlapping with primary. |
| **Sealed** | `2025-26` unfinished / `2026-27+` | — | Not peeked for this lever. |

This overrides the §1.1 half-season default map the same way other decision-policy keeps do (lineup_reopt, force-play): confirmation is a **different season**, not a nested GW slice of the primary.

Control omits `--price-forecast` (flat) on purpose; treatment passes `--price-forecast mean_gw_delta`. All other flags match the reference board (§4).

---

## Reference board + solver + decisions

| Check | Status |
|---|---|
| Horizon `3` | Pass |
| Chip arm `optimised` | Pass |
| `force_play` on | Pass |
| `lineup_reopt` on | Pass |
| `bench_weight` 0 | Pass (live / squad_state default) |
| Model shared both arms | Pass (`dc_layered_v2` hash `12152f0e…` at board time; production rename `dc_layered_v3` in FDB-226 does not change this lever) |
| Price forecast A/B | Control flat / treatment `mean_gw_delta` (single-flag flip) |
| Decisions path | Pass — `fplt decisions --chip-board` (decision-policy suite; not an EvalSpec `claim: squad_points`) |
| Deterministic HiGHS (§5) | Board tip `f29a5a6` **predates** FDB-228 / [#170](https://github.com/e-mcm/fplt/pull/170). Chip-board path now calls `HighsSolveOptions.board_deterministic()`. Archive solves logged `HighsModelStatus.kOptimal` throughout. Directional confirmation uses the existing independent-season archive; future boards inherit §5. |

SPA/MCS: two-arm A/B only; no multi-arm championship on the confirmation window (§3). N6 SE not claimed (deltas not tiny).

---

## Boards (season points)

| Window | Role | Treatment | Control | Δ |
|---|---|---:|---:|---:|
| **2025-26 GW1–38** | Primary | **2002** | 1973 | **+29** |
| **2024-25 GW1–38** | Confirmation | **2253** | 2244 | **+9** |

Mean-PPG ≤ 90: pass both arms both seasons. Same prediction frame within each season both arms.

### Descriptive half-season slices (do not burn SPA; not the keep window)

Path-dependent slices of the continuous full-season sims (same archives). Reported only because §1.1 names these GW ranges; **keep rests on the confirmation season row above**, not these slices.

| Slice | Treatment | Control | Δ |
|---|---:|---:|---:|
| 2025-26 GW1–19 | 888 | 902 | −14 |
| 2025-26 GW20–38 | 1114 | 1071 | +43 |
| 2024-25 GW1–20 | 1108 | 1103 | +5 |
| 2024-25 GW21–38 | 1145 | 1141 | +4 |

Early 2025-26 path noise against treatment; second-half and full-season (and the independent 2024-25 season) all favour treatment. Development is not a keep window (§1).

---

## Call

**Confirm keep.** Under [`eval-protocol.md`](./eval-protocol.md) §2.3:

1. Primary headline (season points) better on 2025-26 (+29).
2. Same directional call on non-overlapping confirmation season 2024-25 (+9).
3. Single policy flag; reference-board flags otherwise matched; decisions chip-board path.
4. Live default `mean_gw_delta` stays; no un-flip.

**Provisional status:** cleared for this lever. Defcon-rate remains provisional until FDB-255.

**Not claimed:** transfer-volume threshold model; mid-horizon buy→sell sell-on for non-owned; N6 sim-SE; re-board under post-#170 HiGHS seed (archive Optimal; §5 now wired for new boards).
