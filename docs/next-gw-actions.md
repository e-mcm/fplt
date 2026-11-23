# Next gameweek actions (live squad, production model)

How to turn **production** predictions into the transfers, XI, captain, and chip for the **next** deadline, using a browser login on this machine.

This is the logged-in squad. An empty fifteen / wildcard-from-scratch is [`blank-week-prod-plan.md`](blank-week-prod-plan.md).

## What the request means

| Phrase | Do this | Do not |
|---|---|---|
| “gw actions”, “what should I do this week” | `fplt plan` for the next deadline. The **week-1** block of `report.md` is the action: transfers, hit, XI, captain / vice, bench, chip. Later weeks are look-ahead only. | Treat optimiser “week-1 actions” tests, or a bakeoff, as the live plan. |
| “prod model” / “production predictions” | First name in `configs/ops/production.yaml` (`models:`). Today **`dc_layered_v4_native`**. Env `FPLT_PRODUCTION_MODELS` wins over the file. Requires the `fplt-mc` wheel (Pages index); missing wheel fails loud. | `openfpl_v1`, `rolling_mean_v1`, python parent `dc_layered_v4`, or a promote fork (`dc_layered_cs_v1`, injuries, match-other, …). |
| “next gw” | Earliest staged gameweek whose `deadline_time` is still in the future (`resolve_as_of`). | Reuse a season or GW from an old command or from a README example (`2025-26`, `GW=10`). |
| “run” / “recommend” | Predict (if this model has no rows for that deadline), then plan. Print the report. | `fplt apply --confirm` / `make apply CONFIRM=1`. That **POSTs** the team. Apply is a separate ask. Dry-run apply does not submit either. |

`make plan` does **not** resolve the next deadline. It requires `SEASON` and `GW`. Passing the wrong pair plans a different week (or fails) even when auth is fine. Example that is wrong once 2026-27 GW 12 is the next deadline: `make plan SEASON=2025-26 GW=11`.

## 1. Auth from the browser

Being logged into [fantasy.premierleague.com](https://fantasy.premierleague.com) in **Chrome, Chromium, Edge, Brave, or Firefox** is enough. The access token is `oidc.user:*` in localStorage, not a cookie. Safari is not supported. No password is read.

```bash
uv run fplt auth from-browser --entry-id YOUR_ID
uv run fplt auth status    # token: set (ok), plus expires_at
```

That writes `~/.config/fplt/env` (or `$FPLT_ENV_FILE`), mode `600`. **That file wins over the repo `.env`.**

If `~/.config/fplt/env` does not exist, a `FPLT_FPL_API_TOKEN` pasted in the checkout `.env` is what `my-team` uses. Logs then say `token=set` and still fail with `FPL API forbidden` once that JWT is stale (~8 hours). Refresh the FPL tab (so localStorage has a new token) and run `auth from-browser` again. `auth status` prints `expired` when the `exp` claim is already past.

Close the browser only if the profile DB is locked; the default copies Chromium LevelDB first.

Never paste the token into the repo `.env` or commit it.

## 2. Resolve model and next deadline

```bash
uv run python -c "
from fplt.automation.production import load_production_ops
from fplt.backtest.predict_live import resolve_as_of
from fplt.data.store import DataStore
from fplt.settings import get_settings
ops = load_production_ops()
key = resolve_as_of(DataStore(get_settings().data_dir))
print(ops.models[0], ops.plan_horizon, key.season, key.gw)
"
```

Use those four values below. Horizon stays at ops `plan_horizon` (today **3**) unless the ask is an ablation.

## 3. Predictions, then the plan

`fplt plan` reads the ledger. It errors with `has no predictions for this deadline` when this `model_id` has no rows for that as-of. Predict at least as many steps as the plan horizon (`--horizon` on predict defaults to **10**; plan defaults to **3**). A second predict for the same model and deadline raises `FileExistsError` (artifacts are not overwritten).

```bash
MODEL=dc_layered_v4_native   # from the snippet, not from memory
SEASON=2026-27        # from the snippet
GW=12                 # from the snippet
HORIZON=3

uv run fplt predict --model "$MODEL" --season "$SEASON" --gw "$GW" --horizon "$HORIZON"
uv run fplt plan    --model "$MODEL" --season "$SEASON" --gw "$GW" --horizon "$HORIZON"
# or: make plan SEASON=$SEASON GW=$GW MODEL=$MODEL HORIZON=$HORIZON
```

Squad comes from `FPLT_ENTRY_ID` plus the token (`/api/my-team/{id}/`: selling prices, bank, free transfers). `--entry-json` wins over the API. Chip arm stays **optimised** (`--chip-arm off` only when ablating).

Report:

`$FPLT_DATA_DIR/plans/<model_id>/as_of=<season>_gwNN_h03/report.md`

The same model, deadline, and horizon are never overwritten. A repeat `fplt plan` reprints the summary and says the parquet already exists.

## Submit only when asked

```bash
make apply SEASON=$SEASON GW=$GW MODEL=$MODEL HORIZON=$HORIZON           # dry-run
make apply SEASON=$SEASON GW=$GW MODEL=$MODEL HORIZON=$HORIZON CONFIRM=1 # POST
```

Needs the same token. Initial drafts (15 buys, no sells) are refused.

## Do not

- Guess the season from a doc example. Resolve it.
- Leave `FPLT_ENTRY_ID` unset and expect the live squad (that path is the blank-week draft).
- Point `--model` at whatever was used in the last shell command.
- Confirm-apply as part of “run the actions”.
