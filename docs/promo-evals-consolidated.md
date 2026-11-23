# Promo A/B evals — consolidated (W7–W9 + Wave B + A3)

One-page read of the promote-eval levers. Numbers copied from the detail docs.

| Lever | Verdict | Why | Next |
|---|---|---|---|
| **W7** DC ± odds prior (`dc_layered_odds_v1`) | **Do not promote** | Closing proxy is not deadline-PIT. Raw de-vig 1X2 **beats DC** (RPS 0.196 vs 0.334); `odds_poisson` loses 1X2 but wins CS Brier — λ map ≠ bookie. Prior blend hurts scoreline; player tiny / top10 worse; decision +134 oracle-only. | Redesign prior off de-vig 1X2; deadline-PIT odds for live; keep vanilla default. #79 fixture-id×season done. |
| **W8** `penalties_order` involvement | **Inconclusive** | Strict vaastav: **0%** non-null order → arms identical. EOS `players_raw` proxy (not PIT) slightly **worse** on MAE. | Keep W2 live default + `*_nopens_*` controls; re-run when deadline-visible order exists. |
| **W9** Defensive contribution wire | **Tune / keep wire** | DC is real (~8% of points) but shrunk 0/2 prior: MAE **worse**, RMSE better, Spearman ≈ flat. Rules-aligned bias on DEF/MID. | Superseded by FDB-211 rate keep — [defcon-rate-promote.md](defcon-rate-promote.md). |
| **FDB-211** Defcon rate P(award) | **Keep (provisional)** | Poisson rate → P(award) beats mean wire on RMSE (primary + GW20–38 confirm); MAE/Spearman also up. | Flipped into `dc_layered_v2`; mean opt-out `dc_layered_defcon_mean_v1`. Provisional until FDB-73. |
| **Wave B** CS-only λ (`dc_layered_cs_v1`) | **Do not promote** | Prematch PIT coverage **92.1%**. CS Brier improves (home 0.0935→0.0901, away 0.0715→0.0660); RPS/LL ≈ flat. Runner still **inconclusive** (no FDB-73 thresholds); components-only. | Keep fork labelled; no `PRODUCTION_MODELS` flip. Optional: player/RMSE board + multi-season before revisit. Detail: [cs-only-market-lambda-eval.md](cs-only-market-lambda-eval.md). |
| **A3 / M7** `involvement_xg` vs shares | **Soft keep xG** | After #97 restage: RMSE **better** (HAC CI +), MAE **worse** (W9 split → prefer RMSE). Goal-share ≈tie. Understat still 0. | Keep `involvement_xg` in `dc_layered_v2`; no `PRODUCTION_MODELS` flip; re-run when Understat returns. |

---

## W7 — DC ± odds prior (FDB-70)

Detail: [dc-odds-prior-eval.md](dc-odds-prior-eval.md)

- **Window:** 2024-25; prediction GW1–20; decisions GW1–38 H=2 (upper bound only). Match-outcome H2H GW1–38.
- **Arms:** `dc_layered_v2` / `dixon_coles_v2` vs `dc_layered_odds_v1` / `dixon_coles_odds_v1` (`market_prior_strength=1.0`); football-data closing averages staged as `odds_kind=closing_proxy`.
- **PIT:** staged `available_at = kickoff` → **0 / 199** deadline-visible same-season joins. Upper bound rewrites to kickoff−7d (**informational only**). Pre-fix fixture-id join leaked prior-season closings into the “strict” arm (#79 season-scopes the join). Closing H2H uses `closing_proxy_match_odds()` — **not deadline PIT**.
- **★ Match outcomes (GW1–38, n=380, closing probe):** raw de-vig 1X2 **beats DC** — RPS **0.196** / LL **0.967** vs DC **0.334** / **1.235**. `odds_poisson_v1` **loses** on 1X2 (RPS 0.363 / LL 1.286) but **wins CS Brier** (home 0.0921 / away 0.0546 vs DC 0.0935 / 0.0715). Poisson λ map ≠ bookie 1X2.
- **Scoreline prior blend (upper, n=199):** prior hurts — RPS 0.353 / log-loss 1.295 vs vanilla **0.342 / 1.274**.
- **Player (upper, n=13,242):** MAE **1.060** vs 1.065; Spearman **0.672** vs 0.670; top10 **0.10** vs 0.20.
- **Decisions (upper):** **2261** vs 2127 (**+134**); hit pts 20 vs 36.
- **Call:** books win raw 1X2; keep vanilla production default; do not promote closing_proxy λ prior; redesign prior off de-vig 1X2.

---

## W8 — `penalties_order` involvement (FDB-71)

Detail: [penalties-order-involvement-eval.md](penalties-order-involvement-eval.md)

- **Window:** 2024-25 GW1–38 walk-forward; control `*_nopens_*` (`use_penalties_order: false`) vs treatment production (`true`). Flag #78; behaviour #72.
- **Strict PIT (vaastav):** 0 non-null `penalties_order` of 27,231 status rows → involvement MAE identical (**0.0840**); layered ΔMAE ~0 (MC noise).
- **EOS `players_raw` proxy (not PIT):** 1,875 non-null / 18 distinct `order==1`. Involvement MAE **+0.00006** worse; layered MAE **+0.0013** worse; top10 0.1→0.2.
- **Call:** inconclusive — keep W2 default for live/snapshot seasons; no historical lift claim.

---

## W9 — Defensive contribution (FDB-72)

Detail: [defensive-contribution-eval-2025-26.md](defensive-contribution-eval-2025-26.md)

- **Lever:** `other_player` ± staged `defensive_contribution` — `dc_layered_v2` vs `dc_layered_nodefcon_v1`. Prediction-only (no decision bakeoff).
- **Window:** 2025-26 GW1–38 walk-forward; n=**29,444** scored player–fixtures/arm; HEAD `ddb195a`.
- **Result:** with-defcon MAE **0.991 vs 0.983** (worse), RMSE **1.983 vs 1.994** (better), Spearman **0.6970 vs 0.6966** (≈tie). Mean ΔxPts ≈ **+0.05** (DEF **+0.09**).
- **Actuals:** DC ≈ **8.2%** of played points; DEF award rate **20.8%**.
- **Call:** **`tune`** — keep staging/wire for rules alignment; do not promote on MAE; next is a proper DC probability/rate model.
- **Follow-up:** FDB-211 rate model **kept** — [defcon-rate-promote.md](defcon-rate-promote.md).

---

## FDB-211 — Defcon rate P(award) vs mean wire

Detail: [defcon-rate-promote.md](defcon-rate-promote.md)

- **Lever:** `defcon_mode: mean → rate` on `other_player` (Poisson actions-per-90 → P(award)).
- **Window:** 2025-26 walk-forward; primary GW1–38 (n=12,696 decision pool); confirmation GW20–38 (n=19,310).
- **Result:** rate **RMSE better** (+0.00117 / +0.00143), MAE better, Spearman ≈tie/+. DEF/MID RMSE up.
- **Call:** **`keep` (provisional)** — production `dc_layered_v2` uses rate; opt-out `dc_layered_defcon_mean_v1`. Formal keep rule still waits on FDB-73.

---

## Wave A3 / M7 — involvement_xg vs shares (FPL-native Opta)

Detail: [involvement-xg-vs-shares-eval.md](involvement-xg-vs-shares-eval.md)

- **Window:** 2024-25 GW1–20 walk-forward; control `dc_layered_shares_v2` vs treatment `dc_layered_v2`. Staging #97; EvalSpec `component_swap` fix #120. HEAD `8697c04`.
- **Data:** `expected_*` **100%** on window after vaastav restage; Understat fetch still fails (maintenance HTML) → **0** staged understat rows. Board is FPL Opta only.
- **Components:** goal-share MAE **0.08510** (xg) vs **0.08513** (shares) — ≈tie.
- **Prediction (n=6162/arm):** RMSE **2.691 vs 2.697** (xG; HAC CI entirely +; 15–5 GW); MAE **1.765 vs 1.754** (shares; 1–19 GW). Spearman flat. Goal/assist/haul Brier slight xG.
- **Call:** **soft keep** production `involvement_xg`; do not demote to shares; do not flip `PRODUCTION_MODELS`. Runner stays `inconclusive` (FDB-73).

---

## Cross-cutting

- Prediction quality ≠ transfer quality (W7 upper-bound decisions vs flat/negative scoreline/player boards).
- Closing books beat DC on raw 1X2; `odds_poisson` is a weak stand-in for that market (throws away 1X2 calibration, keeps some CS signal).
- Two of three W7–W9 levers are blocked or diluted by **data PIT**, not by model code — odds closing ≠ deadline; vaastav `penalties_order` null historically. A3 is unblocked on FPL Opta; Understat still missing.
- Production posture: keep `dc_layered_v2` (pens order on, defcon wire on, **involvement_xg on**); leave odds prior off until deadline-safe odds exist; if revisiting the prior, target de-vig 1X2 not Poisson λ alone.
