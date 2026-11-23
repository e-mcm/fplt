# Reading MAE vs RMSE splits (W9)

*Companion to [`evaluation-north-star.md`](evaluation-north-star.md) and [`feature-eval-pipeline.md`](feature-eval-pipeline.md). Specimen: defensive-contribution wire (W9).*

## Why the two metrics disagree

- The optimiser maximises **expected points**. Squared error (RMSE / MSE) is consistent for the **mean**.
- MAE is consistent for the **median**. With zero-inflated FPL points, MAE rewards forecasts pulled toward zero and penalises correctly priced haul upside.
- A lever can therefore look worse on MAE and better on RMSE (or the reverse). That is not a bug in the board.

## How to call it

| Situation | Call |
|---|---|
| RMSE better, MAE worse (W9 shape) | Prefer **RMSE** for mean-seeking / expected-points promotes. Do not turn the wire off on MAE alone. |
| Both worse | **`drop`** (or `tune` if the lever is still exploratory). |
| RMSE flat, MAE better | Usually **`tune`** — not evidence the EV optimiser got a better mean. |
| Decision-policy board | Season points are the headline; RMSE is secondary / irrelevant for the keep. |

## W9 numbers (reminder)

With-defcon vs without on the early 2025-26 board: MAE ~0.991 vs 0.983 (MAE preferred off), RMSE ~1.983 vs 1.994 (RMSE preferred on). Written call: **`tune`** — leave production on; do not claim a metric win from MAE.

For the interim written-keep rule (confirmation windows, SPA budget), see [`eval-protocol.md`](eval-protocol.md).
