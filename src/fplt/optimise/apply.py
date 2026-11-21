"""Map a saved plan onto FPL transfer / lineup POST payloads.

Dry-run by default. ``confirm=True`` (CLI ``--confirm``) POSTs via
``FPLClient`` using ``FPLT_FPL_API_TOKEN``. Never called from Dagster sensors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

from fplt.data.fpl_client import CurrentTeam, FPLClient
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON, covered_horizon, plan_parquet
from fplt.schemas.plan import PlanSchema
from fplt.settings import Settings, get_settings
from fplt.types import GWKey

logger = logging.getLogger(__name__)

_POS_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


class ApplyError(ValueError):
    """The plan cannot be turned into a valid FPL write payload."""


@dataclass(frozen=True)
class TransferMove:
    """One element_in / element_out pair for ``POST /transfers/``."""

    element_in: int
    element_out: int
    purchase_price: int
    selling_price: int
    player_key_in: int
    player_key_out: int


@dataclass(frozen=True)
class ApplyPayload:
    """Bodies ready for dry-run print or confirmed POST."""

    entry_id: int
    event: int
    transfers: tuple[TransferMove, ...]
    picks: tuple[dict[str, Any], ...]
    chip: str | None = None

    def transfers_body(self) -> dict[str, Any]:
        """JSON body for ``POST /api/transfers/``."""
        return {
            "entry": self.entry_id,
            "event": self.event,
            "transfers": [
                {
                    "element_in": move.element_in,
                    "element_out": move.element_out,
                    "purchase_price": move.purchase_price,
                    "selling_price": move.selling_price,
                }
                for move in self.transfers
            ],
            "chip": self.chip,
        }

    def lineup_body(self) -> dict[str, Any]:
        """JSON body for ``POST /api/my-team/{id}/``."""
        return {"chip": self.chip, "picks": list(self.picks)}


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of a dry-run or confirmed apply."""

    payload: ApplyPayload
    posted_transfers: bool
    posted_lineup: bool
    mode: Literal["dry_run", "confirmed"]


def build_apply_payload(
    plan: pl.DataFrame,
    team: CurrentTeam,
    *,
    element_by_key: dict[int, int],
    purchase_price_by_key: dict[int, int],
    entry_id: int,
    event: int | None = None,
) -> ApplyPayload:
    """Build transfer + lineup bodies from step-0 of a ``PlanSchema`` frame.

    Buys and sells are paired by sorted ``player_key`` (deterministic). Purchase
    prices come from ``purchase_price_by_key`` (usually ``player_status.price``);
    selling prices from ``team`` picks. An initial draft (15 buys, no sells) is
    refused. Zero transfers with a lineup change is allowed.

    Args:
        plan: Rows matching ``PlanSchema``.
        team: Live ``my-team`` (or equivalent) for selling prices.
        element_by_key: ``player_key`` → season ``element_id``.
        purchase_price_by_key: ``player_key`` → now_cost / status price.
        entry_id: FPL manager entry id.
        event: Target gameweek; defaults to ``as_of_gw`` on the plan.

    Returns:
        An ``ApplyPayload`` ready to print or POST.

    Raises:
        ApplyError: Draft plan, unequal buy/sell counts, or missing map/price.
    """
    checked = PlanSchema.validate(plan)
    assert isinstance(checked, pl.DataFrame)
    step0 = checked.filter(pl.col("step") == 0)
    if step0.is_empty():
        raise ApplyError("plan has no step 0 rows")
    as_of = int(step0["as_of_gw"][0])
    gw = as_of if event is None else event

    bought = (
        step0.filter(pl.col("bought") & ~pl.col("sold"))
        .sort("player_key")
        .select("player_key", "position")
    )
    sold = (
        step0.filter(pl.col("sold")).sort("player_key").select("player_key", "position")
    )
    kept = step0.filter(~pl.col("sold"))

    if bought.height >= 15 and sold.height == 0:
        raise ApplyError(
            "plan looks like an initial draft (15 buys, no sells); "
            "FPL /transfers/ cannot create a first squad — use the website"
        )
    if bought.height != sold.height:
        raise ApplyError(
            f"unequal transfers: bought={bought.height} sold={sold.height}"
        )

    sell_price = {
        pick.element: pick.selling_price
        for pick in team.picks
        if pick.selling_price is not None
    }
    moves: list[TransferMove] = []
    for buy_key, sell_key in zip(
        bought["player_key"].to_list(),
        sold["player_key"].to_list(),
        strict=True,
    ):
        buy_key = int(buy_key)
        sell_key = int(sell_key)
        try:
            element_in = element_by_key[buy_key]
            element_out = element_by_key[sell_key]
        except KeyError as exc:
            raise ApplyError(f"no element_id for player_key={exc.args[0]}") from exc
        try:
            purchase = purchase_price_by_key[buy_key]
        except KeyError as exc:
            raise ApplyError(f"no purchase price for player_key={exc.args[0]}") from exc
        try:
            selling = sell_price[element_out]
        except KeyError as exc:
            raise ApplyError(
                f"no selling_price on team for element={exc.args[0]}"
            ) from exc
        moves.append(
            TransferMove(
                element_in=element_in,
                element_out=element_out,
                purchase_price=int(purchase),
                selling_price=int(selling),
                player_key_in=buy_key,
                player_key_out=sell_key,
            )
        )
    moves.sort(key=lambda move: (move.element_out, move.element_in))

    if kept.height != 15:
        raise ApplyError(f"kept squad must be 15 players, got {kept.height}")
    picks = _lineup_picks(kept, element_by_key)
    logger.info(
        "built apply payload entry=%s event=%s transfers=%s",
        entry_id,
        gw,
        len(moves),
    )
    return ApplyPayload(
        entry_id=entry_id,
        event=gw,
        transfers=tuple(moves),
        picks=tuple(picks),
    )


def apply_plan(
    data_dir: Path,
    model_name: str,
    season: str,
    gameweek: int,
    *,
    horizon: int = DEFAULT_PLAN_HORIZON,
    confirm: bool = False,
    transfers_only: bool = False,
    lineup_only: bool = False,
    settings: Settings | None = None,
    client: FPLClient | None = None,
    team: CurrentTeam | None = None,
) -> ApplyResult:
    """Load a saved plan and dry-run or POST transfers / lineup to FPL.

    Args:
        data_dir: Data root with ``plans/`` and ``staged/``.
        model_name: YAML spec name (resolved to ``model_id`` for the parquet).
        season: Season label on the plan.
        gameweek: Deadline ``as_of_gw`` of the plan.
        horizon: Horizon used when the plan was written (same as ``fplt plan``).
        confirm: When False (default), build and return without POSTing.
        transfers_only: Skip the lineup POST.
        lineup_only: Skip the transfers POST.
        settings: Process settings; defaults to ``get_settings()``.
        client: Optional ``FPLClient`` (tests). Owns nothing when passed.
        team: Optional preloaded ``CurrentTeam`` (tests).

    Returns:
        ``ApplyResult`` with the payload and whether each write was posted.

    Raises:
        ApplyError: Plan cannot be applied.
        FileNotFoundError: Plan parquet missing.
        KeyError: Unknown model name.
    """
    if transfers_only and lineup_only:
        raise ApplyError("choose at most one of transfers_only / lineup_only")
    cfg = settings if settings is not None else get_settings()
    if cfg.entry_id is None:
        raise ApplyError("set FPLT_ENTRY_ID to apply a plan")
    if confirm and not cfg.fpl_api_token and client is None:
        raise ApplyError(
            "confirm needs FPLT_FPL_API_TOKEN (Bearer from x-api-authorization)"
        )
    spec = load_model_specs()[model_name]
    covered = covered_horizon(gameweek, horizon)
    path = plan_parquet(data_dir, spec.model_id, season, gameweek, covered)
    if not path.exists():
        raise FileNotFoundError(path)
    plan = pl.read_parquet(path)
    view = PITData(DataStore(data_dir)).view(GWKey(season, gameweek))
    element_by_key = _key_to_element(view)
    prices = _purchase_prices(view)

    live = team
    owns_client = False
    api = client
    if live is None:
        if api is None:
            api = FPLClient(session=cfg.fpl_session, api_token=cfg.fpl_api_token)
            owns_client = True
        try:
            live = api.current_team(cfg.entry_id, event=gameweek)
        finally:
            if owns_client and not confirm:
                api.close()
                api = None
                owns_client = False

    payload = build_apply_payload(
        plan,
        live,
        element_by_key=element_by_key,
        purchase_price_by_key=prices,
        entry_id=cfg.entry_id,
        event=gameweek,
    )

    if not confirm:
        logger.info("dry-run only; pass confirm=True / --confirm to POST")
        if owns_client and api is not None:
            api.close()
        return ApplyResult(
            payload=payload,
            posted_transfers=False,
            posted_lineup=False,
            mode="dry_run",
        )

    if api is None:
        api = FPLClient(session=cfg.fpl_session, api_token=cfg.fpl_api_token)
        owns_client = True
    posted_transfers = False
    posted_lineup = False
    try:
        if not lineup_only and payload.transfers:
            api.submit_transfers(payload.transfers_body())
            posted_transfers = True
        elif not lineup_only and not payload.transfers:
            logger.info("no transfers to POST; lineup-only path")
        if not transfers_only:
            api.submit_lineup(payload.entry_id, payload.lineup_body())
            posted_lineup = True
    finally:
        if owns_client:
            api.close()

    return ApplyResult(
        payload=payload,
        posted_transfers=posted_transfers,
        posted_lineup=posted_lineup,
        mode="confirmed",
    )


def format_apply_preview(result: ApplyResult) -> str:
    """Human-readable dry-run / confirm summary for the CLI.

    Args:
        result: Outcome from ``apply_plan``.

    Returns:
        Multi-line text (no trailing secrets).
    """
    body = result.payload
    lines = [
        f"mode={result.mode} entry={body.entry_id} event={body.event}",
        f"transfers={len(body.transfers)} "
        f"posted_transfers={result.posted_transfers} "
        f"posted_lineup={result.posted_lineup}",
    ]
    for move in body.transfers:
        lines.append(
            f"  out element={move.element_out} (key={move.player_key_out}) "
            f"sell={move.selling_price} -> "
            f"in element={move.element_in} (key={move.player_key_in}) "
            f"buy={move.purchase_price}"
        )
    if not body.transfers:
        lines.append("  (no transfers)")
    captain = next((pick for pick in body.picks if pick.get("is_captain")), None)
    vice = next((pick for pick in body.picks if pick.get("is_vice_captain")), None)
    lines.append(
        f"lineup picks={len(body.picks)} "
        f"captain_element={captain['element'] if captain else '-'} "
        f"vice_element={vice['element'] if vice else '-'}"
    )
    if result.mode == "dry_run":
        lines.append("dry-run: nothing posted; re-run with --confirm to submit")
    return "\n".join(lines) + "\n"


def _lineup_picks(
    kept: pl.DataFrame, element_by_key: dict[int, int]
) -> list[dict[str, Any]]:
    rows = kept.select("player_key", "position", "in_xi", "captain", "vice").to_dicts()
    xi = [row for row in rows if row["in_xi"]]
    bench = [row for row in rows if not row["in_xi"]]
    if len(xi) != 11 or len(bench) != 4:
        raise ApplyError(f"need 11 XI + 4 bench, got xi={len(xi)} bench={len(bench)}")

    def sort_key(row: dict[str, Any]) -> tuple[int, int]:
        return (_POS_ORDER[str(row["position"])], int(row["player_key"]))

    ordered = sorted(xi, key=sort_key) + sorted(bench, key=sort_key)
    picks: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        key = int(row["player_key"])
        try:
            element = element_by_key[key]
        except KeyError as exc:
            raise ApplyError(f"no element_id for player_key={exc.args[0]}") from exc
        picks.append(
            {
                "element": element,
                "position": index,
                "is_captain": bool(row["captain"]),
                "is_vice_captain": bool(row["vice"]),
            }
        )
    return picks


def _key_to_element(view: PITView) -> dict[int, int]:
    frame = (
        view.latest("players", ["season", "player_key"])
        .filter(pl.col("season") == view.as_of.season)
        .select("player_key", "element_id")
        .collect()
    )
    return {int(key): int(element) for key, element in frame.iter_rows()}


def _purchase_prices(view: PITView) -> dict[int, int]:
    frame = (
        view.latest("player_status", ["season", "player_key"])
        .filter(pl.col("season") == view.as_of.season)
        .select("player_key", "price")
        .collect()
    )
    return {int(key): int(price) for key, price in frame.iter_rows()}
