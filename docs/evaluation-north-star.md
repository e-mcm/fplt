# How to tell if an FPL engine is state of the art

*As of 2026-11-23. Adopted as the Project evaluation north star.*

A state-of-the-art claim has to be made at three separate levels: **forecasts**, **decisions** and **live results**. Each has its own benchmark and its own noise. Season rank, the number everyone looks at, is the least informative of the three.

## Adoption notes (Project)

Agreed as the guiding standard for whether the repo is achieving its aims. Additions below do not weaken the tiers; they constrain how we apply them:

1. **RMSE is the headline for mean-seeking boards**; keep MAE as a secondary continuity metric so old A/Bs stay readable. Do not silently rewrite past claims. See [w9-mae-rmse-reading.md](w9-mae-rmse-reading.md) for how to read MAE/RMSE splits. Windows, interim written keeps, and the reference board live in [eval-protocol.md](eval-protocol.md).
2. **Promote-eval (HAC / bootstrap / SPA / MCS, PIT, leakage)** remains the significance layer. This doc defines *what* to measure; that stack defines *whether a difference is real*.
3. **FPL Review archives** only after a terms check. Until then, archive `ep_next` and OpenFPL every deadline; Tier B stays blocked on Review data.
4. **OpenFPL comparisons** must label Deep/PPDA-degraded runs until Understat stages cleanly (FDB-55).
5. **Share of attainable points** needs an explicit computable definition (constrained hindsight oracle), not an unbounded perfect-foresight ideal.
6. **Live rank** is a multi-season consistency check, never a single-GW or single-season promotion gate.
7. **Decision-sim north star is points-at-rank, not a fixed 2400.** Raw season points are not comparable across seasons (defcon, chip-rule changes, scoring inflation/deflation). Anchor: **2400** at end of **2024-25** ≈ overall rank **~573k** (top ~5%). Per-season bars = points at that rank on each season's end-of-season ladder. See §2. Does not replace Tier A–C relative aims or A/B promote-eval gates.
8. **Keeps under the interim rule:** defcon-rate (`defcon_mode: rate` on `dc_layered_v4`) remains **provisional** after FDB-255 protocol reconfirm failed interim RMSE keep ([`defcon-rate-confirm.md`](defcon-rate-confirm.md); no silent un-flip). Within-horizon `mean_gw_delta` price forecast is **confirmed** under [`eval-protocol.md`](eval-protocol.md) §2.3 (FDB-256 — [`price-forecast-confirm.md`](price-forecast-confirm.md); prior promote [`price-within-horizon-promote.md`](price-within-horizon-promote.md)).

---

## 1. Forecasts: are your projections better than the best available?

**Benchmark against the actual leader, on identical player–fixtures, prospectively.** That's how OpenFPL made its claim. It compared against FPL Review's Massive Data Model (a subscription service) and a simple last-5 baseline, and reported results for four groups of outcomes: Zeros (didn't play), Blanks (≤2 points), Tickers (3–4 points) and Haulers (≥5 points). For fplt that means:

- Save FPL Review's projections before each deadline. It's a paid service, so check its terms.
- Run OpenFPL, and save `ep_next` from your snapshots.
- Compare paired, fixture by fixture, with the Diebold–Mariano / HAC machinery you already have, over at least a full season.

**Use the scoring rule that matches what the optimiser consumes.** This is the most important correction to the current boards.

- The optimiser maximises **expected points**. The scoring function that's consistent for a mean is **squared error**, so **RMSE should be the headline metric, not MAE**.
- MAE is consistent for the *median*. With zero-inflated FPL points, MAE rewards forecasts pulled towards zero and penalises correctly priced haul upside.
- That's why W9 split (MAE worse, RMSE better), and why the RMSE side was the right call.
- `lgbm_direct_v1` is trained with LightGBM `objective: regression` (L2 / mean-seeking). The YAML/`metric: mae` setting is monitoring only — it is **not** an MAE training objective. Older notes that called it an MAE/median model are wrong.

**Score probabilities as well as means.**

- Brier or log score on the events decisions hinge on: played, 60+ minutes, goal, assist, clean sheet, haul.
- CRPS or ranked probability score on the full points distribution.
- Calibration curves by xPts decile and by position.

**Score only the players who matter.** Including 300 bench-fodder zeros flatters everyone. Restrict to a decision-relevant pool, e.g. players any optimiser run would consider, or a price × minutes filter. Then report:

- the OpenFPL groups (zeros, blanks, tickers, haulers);
- Spearman within position;
- top-k precision **per gameweek**, averaged across gameweeks, not the single pooled top-10 used now.

---

## 2. Decisions: do the forecasts turn into points?

**Separate the forecast from the optimiser.** Run a 2×2 swap: your forecasts vs the benchmark's, each through your optimiser and through a reference optimiser (e.g. FPL-Optimization-Tools). Otherwise you can't tell which part earns the gain.

### North-star score: points-at-rank (was “2400”)

Full-rules decision backtests — chips on, sell-on, real banking; same board rules as live defaults (H=3, force-play on, lineup_reopt on, `dc_layered_v4` unless noted) — aim for the **points total that maps to the anchor overall rank** on that season's end-of-season ladder.

| | |
|---|---|
| **Anchor** | **2400** points at end of **2024-25** |
| **Equivalent overall rank** | **≈573k** (exact 2400 finishes landed 569.8k–575.5k; median **572.6k**) |
| **Percentile** | top **~5.0%** of ~11.49M managers (2024-25) |
| **Why not fixed 2400?** | Defcon (2025-26), dual-half chips, and season scoring level make raw points incomparable. 2025-26 Overall #1 was **2582** vs **2810** in 2024-25 — a lower-scoring year at the top despite defcon. Hitting 2400 in 2025-26 would have been roughly top-~30k territory, not the same bar as 2400 in 2024-25. |

#### Per-season bars

| Season | Rules context | North-star points (= pts at OR ≈573k) |
|---|---|---:|
| **2024-25** | Pre-defcon; classic chip set (2×WC, 1×FH/BB/TC) | **2400** |
| **2025-26** | Defcon + dual-half chips (8 chips) | **2235** |

Use the **per-season bar** when scoring a single season board (e.g. [stacked-prod-2400-board.md](./stacked-prod-2400-board.md)). Gap = season points − that season's bar.

#### Mean bar (dashboard rollup only)

| | |
|---|---|
| **Mean bar** | **2318** = mean(2400, 2235) |

- **Per-season bar** is the gate for each season. Never judge 2025-26 against 2400, or 2024-25 against 2235.
- **Mean bar** is a single headline for multi-season dashboards that want one number. It is **not** a cross-season comparable points total and **not** a substitute for the per-season bars.
- Multi-season consistency should roll up **rank / percentile** (N7 / `fplt.backtest.live_percentile.mean_percentile`) or mean of (points − season_bar), not (mean points − 2400).

These bars are stretch targets for decision sims, not a claim we already hit them, and not a replacement for A/B promote calls or Tier A–C relative aims.


**Metrics**

| Metric | What it is | Why it matters |
|---|---|---|
| Season points under full rules | Chips on, price changes and the 50% sell-on, real free-transfer banking | Anything less isn't comparable to a real season; **aim for the per-season points-at-rank bar** (2400 / 2235), not a fixed absolute |
| Share of attainable points captured | Realised points ÷ the best possible season in hindsight under the same constraints | Normalises across seasons; high-scoring years inflate everyone |
| Regret by component | Captaincy (chosen vs best captain in your XI), XI and bench order, transfers, chip timing | Shows where points are lost |
| Uncertainty | Many simulated seasons plus several real seasons | A single season is one path-dependent draw; the +72 OpenFPL result and the H=3 choice both rest on one |

---

## 3. Live results: what rank would actually tell you

Rank is the final test, but it's very noisy.

- **The field is huge.** More than 11.5 million people took part in 2024/25; ~12–13 million in 2025/26.
- **Single seasons involve luck.** The 2024/25 champion had never previously finished inside the top 500k; the 2025/26 champion was a first-season manager.
- **Skill shows up as consistency.** The best managers repeat: one has six consecutive top-10k finishes, and another finished inside the top 10k every season from 2019/20 to 2022/23.

So the live criterion is **percentile across several seasons**, not one rank.

- The FPL API's overall-league standings give the points total at any rank (e.g. the 10,000th). Use them to convert any season total, real or backtested, into an equivalent rank.
- The decision-sim north star in §2 is the same idea applied to backtests: convert a points target through the ladder, then keep the **rank** fixed across seasons.
- Use your simulator to estimate how much a season total varies from luck alone before reading anything into a gap.

---

## 4. Reasonable aims

| Tier | Forecasts | Decisions | Live |
|---|---|---|---|
| **A: credible** (achievable after the chips and minutes work) | Beats `ep_next`, rolling mean and OpenFPL on paired RMSE and event Brier over a prospective season. DC matches or beats closing de-vigged 1X2 on mature seasons (already true) | Chips-on season points beat a rule-based chip baseline, with standard errors from simulated seasons. Captures a stable share of attainable points across 3 seasons | Consistently top 1% (roughly top 100k) |
| **B: competitive with the state of the art** | Not significantly worse than FPL Review on paired RMSE and log score over a full season; better in some groups (e.g. haulers) | In the 2×2 swap, your forecasts through the same optimiser are statistically tied with FPL Review's | Around the top-10k equivalent on average across seasons |
| **C: state of the art** | Significantly better than FPL Review on paired RMSE/log score over at least a season, prospectively | Ahead in the swap test across real and simulated seasons | Repeated top-10k finishes |

Tier A is a realistic one-season aim. Tier B is the honest test of a state-of-the-art claim, and it needs about a year of saved benchmark projections. Tier C can only be confirmed over several seasons, however good the backtests look. The **points-at-rank** decision-sim bars sit beside these tiers as an absolute stretch target (top ~5% / OR ≈573k equivalent); hitting them does not skip promote-eval or Tier B/C relative tests. Tier B/C live aims (top 10k) remain harder than this decision-sim bar.

---

## 5. Next steps

1. **Start archiving benchmark projections now:** FPL Review (subject to its terms), OpenFPL and `ep_next`, saved before every deadline. Tier B can't be tested without them.
2. **Switch the headline metric from MAE to RMSE**, scored on the decision-relevant pool and reported per gameweek. Several earlier calls would read differently under that rule.
3. **Add the probabilistic event scores** (Brier/log score for played, 60+, goal, assist, clean sheet, haul) and calibration curves to the standard report.
4. **Build the 2×2 forecast × optimiser swap** and the share-of-attainable-points metric into the decision board.
5. **Add simulated-season evaluation**, so decision differences come with standard errors.
6. **Keep end-of-season points-at-rank ladders** (N7 / `make fetch-overall-rank`) so new seasons can mint a per-season bar from the same OR ≈573k anchor without re-deriving from scratch.

---

## Sources

- Groos, *OpenFPL: An open-source forecasting method rivaling state-of-the-art Fantasy Premier League services* (arXiv 2508.09992): https://arxiv.org/html/2508.09992v1
- Premier League, *FPL records and amazing stats from 2024/25* (11.5m participants): https://www.premierleague.com/en/news/4320685/fpl-records-and-amazing-stats-from-202425
- Fantasy Football Scout, 2024/25 champion and hall-of-fame notes: https://www.fantasyfootballscout.co.uk/comment/27215677 and https://www.fantasyfootballscout.co.uk/comment/27215759
- Fantasy Football Scout, 2024/25 mini-league and community winners: https://www.fantasyfootballscout.co.uk/2025/05/28/ffs-mini-leagues-and-community-competitions-the-2024-25-winners
- FPL entry histories for exact 2400 → OR mapping (2024/25) and OR ≈573k → points mapping (2025/26); Overall #1 totals 2810 (2024/25) / 2582 (2025/26)
- Field sizes: Fantasy Football Pundit rank-history table (2024/25 11,488,573; 2025/26 12,176,091 registered)
