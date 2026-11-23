# Promote plan: within-horizon price-rise forecast (D5 remainder)

**Eval status:** **evaluated — keep** (FDB-218). FDB-205 canceled — as-of sell-on is a bug fix (#124), not a gate.  
**Result:** [price-within-horizon-promote.md](../price-within-horizon-promote.md)  
**Design:** [price-within-horizon-design.md](../price-within-horizon-design.md) · code [#160](https://github.com/e-mcm/fplt/pull/160) · board flag [#163](https://github.com/e-mcm/fplt/pull/163)

| | Control | Treatment |
|---|---|---|
| Policy | As-of sell-on + **flat** within-horizon | `mean_gw_delta` list-price forecast |
| CLI | omit `--price-forecast` | `--price-forecast mean_gw_delta` |
| Season pts | 1973 / 2244 | **2002** / **2253** (+29 / +9) |

### Ordered steps

1. ~~Wait on FDB-205.~~ Canceled (sell-on = bug fix).
2. ~~Design + library wiring.~~ [#160](https://github.com/e-mcm/fplt/pull/160).
3. ~~Board CLI flag.~~ [#163](https://github.com/e-mcm/fplt/pull/163).
4. ~~H2H boards + written call.~~ Keep — [price-within-horizon-promote.md](../price-within-horizon-promote.md).
5. ~~Flip live `run_plan` default to `mean_gw_delta`.~~ [#165](https://github.com/e-mcm/fplt/pull/165) merged (`35b9ac5`).
