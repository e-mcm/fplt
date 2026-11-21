"""Season decisions from a plan, and the reads that feed ``fplt plan``.

A decision backtest starts from an optimal GW1 draft. Chips stay off. Only
the first step of each plan is played; later steps exist so a free transfer
can be banked. Predictions are whatever the ledger recorded at that deadline.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import polars as pl

from fplt.backtest.ledger import read_predictions
from fplt.data.fpl_client import CurrentTeam, load_current_team
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.plan import (
    GameweekPlan,
    next_free_transfers,
    plan_squad,
    plan_table,
    render_report,
    write_plan,
)
from fplt.optimise.realised import format_decision_scoreboard, summarise_decisions
from fplt.optimise.state import (
    OwnedPlayer,
    SquadState,
    apply_entry_picks,
    load_squad_state,
)
from fplt.schemas.plan import CandidateSchema, DecisionSchema
from fplt.settings import Settings, get_settings
from fplt.types import GWKey

TeamLoader = Callable[..., CurrentTeam]
logger = logging.getLogger(__name__)

# A blank gameweek drops a couple of clubs. Below half the candidate pool,
# the missing rows are being treated as zero and the plan is not usable.
_MIN_COVERAGE = 0.5

__all__ = [
    "PlanExistsError",
    "candidates_as_of",
    "format_decision_scoreboard",
    "points_from_ledger",
    "read_decisions",
    "resolve_squad_state",
    "run_decision_bakeoff",
    "run_decisions",
    "run_plan",
    "simulate_decisions",
    "summarise_decisions",
    "write_decisions",
]


class PlanExistsError(FileExistsError):
    """The plan parquet for this model, deadline, and horizon is already on disk.

    ``report`` is the transfer summary computed before the write was refused.
    """

    def __init__(self, path: Path, report: str) -> None:
        """Remember the existing path and the summary that could not be saved.

        Args:
            path: Parquet path that already exists.
            report: Markdown summary of the plan.
        """
        super().__init__(str(path))
        self.report = report


def candidates_as_of(view: PITView) -> pl.DataFrame:
    """Players and prices visible at ``view``'s deadline.

    Args:
        view: Point-in-time view for one gameweek.

    Returns:
        A frame matching ``CandidateSchema``.
    """
    season = view.as_of.season
    players = (
        view.latest("players", ["season", "player_key"])
        .filter(pl.col("season") == season)
        .select("player_key", "position", "team_key")
        .collect()
    )
    status = (
        view.latest("player_status", ["season", "player_key"])
        .filter(pl.col("season") == season)
        .select("player_key", "price")
        .collect()
    )
    joined = players.join(status, on="player_key", how="inner")
    checked = CandidateSchema.validate(
        joined.select("player_key", "position", "team_key", "price")
    )
    assert isinstance(checked, pl.DataFrame)
    return checked


def points_from_ledger(
    ledger: pl.DataFrame,
    model_id: str,
    season: str,
    as_of_gw: int,
    gameweeks: list[int],
) -> pl.DataFrame:
    """Sum a deadline's predictions onto horizon steps.

    Args:
        ledger: Prediction ledger, possibly empty.
        model_id: Predictor whose rows to keep.
        season: Season of both the deadline and the targets.
        as_of_gw: Deadline the predictions were issued at.
        gameweeks: Target gameweeks, in horizon order.

    Returns:
        ``player_key``, ``step``, and ``xpts``. Players with no row are absent.
    """
    empty = pl.DataFrame(
        schema={"player_key": pl.Int64(), "step": pl.Int64(), "xpts": pl.Float64()}
    )
    if ledger.is_empty():
        return empty
    chosen = ledger.filter(
        (pl.col("model_id") == model_id)
        & (pl.col("as_of_season") == season)
        & (pl.col("as_of_gw") == as_of_gw)
        & (pl.col("target_season") == season)
        & pl.col("target_gw").is_in(gameweeks)
    )
    if chosen.is_empty():
        return empty
    step_of = {gameweek: step for step, gameweek in enumerate(gameweeks)}
    return (
        chosen.group_by("player_key", "target_gw")
        .agg(pl.col("xpts").sum())
        .with_columns(pl.col("target_gw").replace_strict(step_of).alias("step"))
        .select("player_key", "step", "xpts")
    )


def simulate_decisions(
    candidates: pl.DataFrame,
    predictions: pl.DataFrame,
    actuals: pl.DataFrame,
    state: SquadState,
    *,
    model_id: str,
    season: str,
    gameweeks: list[int],
    horizon: int,
) -> pl.DataFrame:
    """Play ``gameweeks`` from ``state``, acting only on each plan's first step.

    Args:
        candidates: ``gw``, ``player_key``, ``position``, ``team_key``, ``price``.
        predictions: ``as_of_gw``, ``target_gw``, ``player_key``, ``xpts``.
        actuals: ``gw``, ``player_key``, ``actual_pts``, and optional
            ``actual_minutes``. A sold player scores nothing.
        state: Starting bank, free transfers, and owned players.
        model_id: Written onto every row.
        season: Season label.
        gameweeks: Deadlines to play, in order.
        horizon: How many weeks each plan looks ahead, capped at gameweek 38.

    Returns:
        A frame matching ``DecisionSchema``.

    Raises:
        ValueError: A week has no candidates, or the squad rules cannot be met.
    """
    scored = _actuals(actuals)
    rows: list[dict[str, object]] = []
    current = state
    for gameweek in gameweeks:
        window = [
            gameweek + offset for offset in range(horizon) if gameweek + offset <= 38
        ]
        pool = candidates.filter(pl.col("gw") == gameweek).drop("gw")
        if pool.is_empty():
            raise ValueError(f"no candidates for gw {gameweek}")
        solved = plan_squad(
            pool,
            _window_points(predictions, gameweek, window),
            current,
            initial=not current.owned,
            horizon=len(window),
        )
        rows.extend(_decision_rows(solved, scored, model_id, season, gameweek))
        # Selling stays at purchase price (accepted bakeoff approx; no rises).
        current = _advance(current, solved, pool)
    return _decisions(pl.DataFrame(rows) if rows else _empty_decisions())


def read_decisions(root: Path) -> pl.DataFrame:
    """Load ``ledger/decisions/part.parquet``, or an empty decision frame."""
    path = root / "ledger" / "decisions" / "part.parquet"
    if not path.exists():
        return _empty_decisions()
    return _decisions(pl.read_parquet(path))


def write_decisions(root: Path, frame: pl.DataFrame) -> Path:
    """Append decisions, keeping the latest row for each player-gameweek.

    Args:
        root: Data directory that contains ``ledger/``.
        frame: Rows matching ``DecisionSchema``.

    Returns:
        ``ledger/decisions/part.parquet``.
    """
    path = root / "ledger" / "decisions" / "part.parquet"
    if frame.is_empty():
        return path
    checked = _decisions(frame)
    if path.exists():
        existing = pl.read_parquet(path)
        checked = pl.concat([existing, checked], how="vertical_relaxed")
        checked = checked.unique(
            subset=["model_id", "season", "gw", "player_key"],
            keep="last",
        )
        checked = _decisions(checked)
    path.parent.mkdir(parents=True, exist_ok=True)
    checked.write_parquet(path)
    return path


def resolve_squad_state(
    view: PITView,
    *,
    entry_json: Path | None = None,
    event: int | None = None,
    settings: Settings | None = None,
    load_team: TeamLoader | None = None,
) -> SquadState:
    """Build squad state from YAML, optional entry JSON, or the live FPL team.

    Preference order:

    1. ``entry_json`` when provided (CLI ``--entry-json``).
    2. Live / public team when ``FPLT_ENTRY_ID`` is set
       (``load_current_team`` → ``apply_entry_picks``).
    3. Otherwise the YAML draft (empty ``owned`` is an initial squad).

    Free transfers from authenticated ``my-team`` replace the YAML value when
    they fit ``ft_bank_cap``. Public entry picks leave YAML free transfers
    unchanged. Chip overrides always stay in YAML.

    Args:
        view: Point-in-time view used to map FPL element ids to ``player_key``.
        entry_json: Optional picks JSON path. Wins over the live entry.
        event: Finished gameweek for the public picks fallback when no
            session cookie is set. Ignored when ``my-team`` succeeds.
        settings: Process settings; defaults to ``get_settings()``.
        load_team: Injectable ``load_current_team`` (for tests).

    Returns:
        Squad state ready for ``plan_squad`` / ``run_plan``.

    Raises:
        FPLAPIError: Live load fails when ``entry_id`` is set.
        KeyError: A pick's element has no ``player_key`` in ``view``.
        ValueError: An entry payload is malformed or lacks selling prices.
    """
    state = load_squad_state()
    keys = _element_keys(view)
    if entry_json is not None:
        logger.info("resolving squad from entry_json=%s", entry_json)
        payload = json.loads(entry_json.read_text(encoding="utf-8"))
        resolved = apply_entry_picks(state, payload, keys)
        logger.info(
            "squad from JSON: owned=%s bank=%s",
            len(resolved.owned),
            resolved.bank,
        )
        return resolved

    cfg = settings if settings is not None else get_settings()
    if cfg.entry_id is None:
        logger.info(
            "no FPLT_ENTRY_ID; using YAML squad (owned=%s bank=%s)",
            len(state.owned),
            state.bank,
        )
        return state

    logger.info(
        "resolving squad via FPL API entry_id=%s event=%s",
        cfg.entry_id,
        event,
    )
    loader = load_current_team if load_team is None else load_team
    team = loader(event=event, settings=cfg)
    state = apply_entry_picks(state, team.picks_payload(), keys)
    state = _with_live_free_transfers(state, team)
    logger.info(
        "squad ready: owned=%s bank=%s free_transfers=%s source=%s",
        len(state.owned),
        state.bank,
        state.free_transfers,
        team.source,
    )
    return state


def run_plan(
    data_dir: Path,
    model_name: str,
    season: str,
    gameweek: int,
    horizon: int,
    entry_json: Path | None = None,
    squad: SquadState | None = None,
) -> tuple[GameweekPlan, Path, str]:
    """Build a plan from the ledger and write the parquet and the report.

    When ``squad`` is omitted, squad state comes from ``resolve_squad_state``:
    ``entry_json`` if provided, else the live / public team when
    ``FPLT_ENTRY_ID`` is set, else the YAML draft.

    Args:
        data_dir: Root that contains ``staged/`` and ``ledger/``.
        model_name: YAML spec name.
        season: Season to plan.
        gameweek: Deadline to plan from.
        horizon: Weeks to look ahead, capped at gameweek 38.
        entry_json: Optional ``entry/{id}/event/{gw}/picks`` payload. It
            replaces the owned squad and the bank when ``squad`` is omitted.
        squad: Pre-resolved squad state (e.g. from ``resolve_squad_state``).
            When set, ``entry_json`` and ``FPLT_ENTRY_ID`` are ignored.

    Returns:
        The plan, the parquet path, and the markdown report.

    Raises:
        KeyError: The spec name or the gameweek is unknown.
        PlanExistsError: This deadline and horizon were already written.
            ``report`` on the exception is the summary.
        FPLAPIError: Live load fails when ``FPLT_ENTRY_ID`` is set and no
            ``entry_json`` / ``squad`` was provided.
        ValueError: This deadline has no predictions, a horizon week is
            missing or all zero, or a week covers fewer than half the
            candidates.
    """
    logger.info(
        "run_plan model=%s season=%s gw=%s horizon=%s",
        model_name,
        season,
        gameweek,
        horizon,
    )
    spec = load_model_specs()[model_name]
    view = PITData(DataStore(data_dir)).view(GWKey(season, gameweek))
    if squad is not None:
        state = squad
        logger.info("using pre-resolved squad (owned=%s)", len(state.owned))
    else:
        state = resolve_squad_state(
            view,
            entry_json=entry_json,
            event=gameweek,
        )
    window = [gameweek + offset for offset in range(horizon) if gameweek + offset <= 38]
    candidates = candidates_as_of(view)
    points = points_from_ledger(
        read_predictions(data_dir), spec.model_id, season, gameweek, window
    )
    logger.info(
        "optimiser inputs: candidates=%s prediction_rows=%s window=%s initial=%s",
        candidates.height,
        points.height,
        window,
        not state.owned,
    )
    _require_usable_points(points, candidates, window, model_name)
    solved = plan_squad(
        candidates,
        points,
        state,
        initial=not state.owned,
        horizon=len(window),
    )
    table = plan_table(solved, spec.model_id, season, gameweek, window)
    report = render_report(solved, season, window, names=_player_names(view))
    try:
        path = write_plan(data_dir, table)
    except FileExistsError as exc:
        raise PlanExistsError(Path(str(exc)), report) from exc
    (path.parent / "report.md").write_text(report, encoding="utf-8")
    logger.info(
        "plan written path=%s hits=%s points=%.2f",
        path,
        solved.hits,
        solved.points,
    )
    return solved, path, report


def _require_usable_points(
    points: pl.DataFrame,
    candidates: pl.DataFrame,
    gameweeks: list[int],
    model_name: str,
) -> None:
    """Reject a plan whose expected points are missing or all zero.

    Args:
        points: ``player_key``, ``step``, ``xpts`` from ``points_from_ledger``.
        candidates: Players who can be picked. Missing rows would score zero.
        gameweeks: Horizon targets in step order.
        model_name: Spec name used in the error.

    Raises:
        ValueError: No rows, a week with no rows, a week of all zeros, or a
            week covering fewer than half the candidates.
    """
    if points.is_empty():
        raise ValueError(
            f"{model_name} has no predictions for this deadline; run fplt predict first"
        )
    present = {int(step) for step in points["step"].unique().to_list()}
    missing = [gameweeks[step] for step in range(len(gameweeks)) if step not in present]
    if missing:
        label = ", ".join(str(gameweek) for gameweek in missing)
        raise ValueError(f"{model_name} has no predictions for gameweek {label}")
    if candidates.height == 0:
        return
    for step, gameweek in enumerate(gameweeks):
        part = points.filter(pl.col("step") == step)
        peak = part.select(pl.col("xpts").abs().max()).item()
        if peak is None or float(peak) == 0.0:
            raise ValueError(
                f"{model_name} predictions for gameweek {gameweek} are all zero"
            )
        covered = (
            candidates.select("player_key")
            .join(part.select("player_key").unique(), on="player_key", how="inner")
            .height
        )
        if covered / candidates.height < _MIN_COVERAGE:
            missing_players = candidates.height - covered
            raise ValueError(
                f"{model_name} is missing predictions for {missing_players} of "
                f"{candidates.height} players in gameweek {gameweek}"
            )


def run_decisions(
    data_dir: Path,
    model_name: str,
    season: str,
    from_gw: int,
    to_gw: int,
    horizon: int,
) -> Path:
    """Simulate an optimal GW1 squad across a season and append the ledger.

    The yaml squad is ignored. The draft spends a 1000 budget and the first
    transfer window has one free transfer. Decay, the banking cap, and the
    hit cost still come from ``configs/squad_state.yaml``.

    Args:
        data_dir: Root that contains ``staged/`` and ``ledger/``.
        model_name: YAML spec name.
        season: Season to play.
        from_gw: First gameweek. Gameweek 1 is a draft when the state is empty.
        to_gw: Last gameweek, inclusive.
        horizon: Weeks each plan looks ahead.

    Returns:
        The decision parquet path.

    Raises:
        KeyError: The spec name or a gameweek deadline is unknown.
    """
    spec = load_model_specs()[model_name]
    rules = load_squad_state()
    draft = rules.model_copy(update={"owned": [], "bank": 1000, "free_transfers": 1})
    store = DataStore(data_dir)
    gameweeks = list(range(from_gw, to_gw + 1))
    pools = [
        candidates_as_of(PITData(store).view(GWKey(season, gw))).with_columns(
            pl.lit(gw).alias("gw")
        )
        for gw in gameweeks
    ]
    ledger = read_predictions(data_dir)
    predictions = _prediction_rows(ledger, spec.model_id, season)
    actuals = _actual_rows(ledger, spec.model_id, season)
    frame = simulate_decisions(
        pl.concat(pools),
        predictions,
        actuals,
        draft,
        model_id=spec.model_id,
        season=season,
        gameweeks=gameweeks,
        horizon=horizon,
    )
    return write_decisions(data_dir, frame)


def run_decision_bakeoff(
    data_dir: Path,
    models: list[str],
    season: str,
    from_gw: int,
    to_gw: int,
    horizon: int,
    *,
    simulate: bool = True,
) -> pl.DataFrame:
    """Simulate decisions for each model (optional) and return a season scoreboard.

    Args:
        data_dir: Root with ``staged/`` and ``ledger/``.
        models: YAML spec names to compare.
        season: Season label.
        from_gw: First gameweek.
        to_gw: Last gameweek, inclusive.
        horizon: Planner horizon (must match the backtest ``--steps`` used).
        simulate: When true, call ``run_decisions`` for each model first.

    Returns:
        Summary from ``summarise_decisions`` for ``season`` and ``models``.

    Raises:
        KeyError: An unknown model name.
        ValueError: ``models`` is empty.
    """
    if not models:
        raise ValueError("models must be non-empty")
    if simulate:
        for name in models:
            logger.info(
                "decision bakeoff simulate model=%s season=%s gw=%s-%s horizon=%s",
                name,
                season,
                from_gw,
                to_gw,
                horizon,
            )
            run_decisions(data_dir, name, season, from_gw, to_gw, horizon)
    specs = load_model_specs()
    model_ids = {specs[name].model_id for name in models}
    frame = read_decisions(data_dir).filter(
        (pl.col("season") == season) & pl.col("model_id").is_in(list(model_ids))
    )
    if from_gw > 1 or to_gw < 38:
        frame = frame.filter((pl.col("gw") >= from_gw) & (pl.col("gw") <= to_gw))
    return summarise_decisions(frame)


def _prediction_rows(ledger: pl.DataFrame, model_id: str, season: str) -> pl.DataFrame:
    empty = pl.DataFrame(
        schema={
            "as_of_gw": pl.Int64(),
            "target_gw": pl.Int64(),
            "player_key": pl.Int64(),
            "xpts": pl.Float64(),
        }
    )
    if ledger.is_empty():
        return empty
    chosen = ledger.filter(
        (pl.col("model_id") == model_id)
        & (pl.col("as_of_season") == season)
        & (pl.col("target_season") == season)
    )
    if chosen.is_empty():
        return empty
    return chosen.group_by("as_of_gw", "target_gw", "player_key").agg(
        pl.col("xpts").sum()
    )


def _actual_rows(ledger: pl.DataFrame, model_id: str, season: str) -> pl.DataFrame:
    empty = pl.DataFrame(
        schema={
            "gw": pl.Int64(),
            "player_key": pl.Int64(),
            "actual_pts": pl.Int64(),
            "actual_minutes": pl.Int64(),
        }
    )
    if ledger.is_empty():
        return empty
    chosen = ledger.filter(
        (pl.col("model_id") == model_id) & (pl.col("target_season") == season)
    )
    if chosen.is_empty():
        return empty
    if "actual_minutes" in chosen.columns:
        return (
            chosen.group_by("target_gw", "player_key")
            .agg(pl.col("actual_pts").sum(), pl.col("actual_minutes").sum())
            .rename({"target_gw": "gw"})
        )
    return (
        chosen.group_by("target_gw", "player_key")
        .agg(pl.col("actual_pts").sum())
        .rename({"target_gw": "gw"})
        .with_columns(pl.lit(None).cast(pl.Int64).alias("actual_minutes"))
    )


def _window_points(
    predictions: pl.DataFrame, as_of: int, window: list[int]
) -> pl.DataFrame:
    empty = pl.DataFrame(
        schema={"player_key": pl.Int64(), "step": pl.Int64(), "xpts": pl.Float64()}
    )
    if predictions.is_empty():
        return empty
    chosen = predictions.filter(
        (pl.col("as_of_gw") == as_of) & pl.col("target_gw").is_in(window)
    )
    if chosen.is_empty():
        return empty
    step_of = {gameweek: step for step, gameweek in enumerate(window)}
    return chosen.with_columns(
        pl.col("target_gw").replace_strict(step_of).alias("step")
    ).select("player_key", "step", "xpts")


def _decision_rows(
    plan: GameweekPlan,
    actuals: pl.DataFrame,
    model_id: str,
    season: str,
    gameweek: int,
) -> list[dict[str, object]]:
    played = plan.picks.filter(pl.col("step") == 0)
    scored = actuals.filter(pl.col("gw") == gameweek)
    points_by = {
        int(key): pts
        for key, pts in scored.select("player_key", "actual_pts").iter_rows()
    }
    minutes_by: dict[int, int | None] = {}
    if "actual_minutes" in scored.columns:
        minutes_by = {
            int(key): (None if mins is None else int(mins))
            for key, mins in scored.select("player_key", "actual_minutes").iter_rows()
        }
    rows: list[dict[str, object]] = []
    for row in played.iter_rows(named=True):
        key = int(row["player_key"])
        points = None if row["sold"] else points_by.get(key)
        minutes = None if row["sold"] else minutes_by.get(key)
        rows.append(
            {
                "model_id": model_id,
                "season": season,
                "gw": gameweek,
                "player_key": key,
                "position": str(row["position"]),
                "in_xi": bool(row["in_xi"]),
                "captain": bool(row["captain"]),
                "vice": bool(row["vice"]),
                "bought": bool(row["bought"]),
                "sold": bool(row["sold"]),
                "xpts": float(row["xpts"]),
                "actual_pts": points,
                "actual_minutes": minutes,
                "hit": float(plan.hits[0]),
                "free_transfers": int(plan.free_transfers[0]),
            }
        )
    return rows


def _advance(state: SquadState, plan: GameweekPlan, pool: pl.DataFrame) -> SquadState:
    step = plan.picks.filter(pl.col("step") == 0)
    previous = {player.player_key: player.selling_price for player in state.owned}
    kept: list[OwnedPlayer] = []
    bank = state.bank
    for row in step.iter_rows(named=True):
        key = int(row["player_key"])
        if row["sold"]:
            bank += previous[key]
            continue
        price = int(row["price"])
        selling = price if row["bought"] or key not in previous else previous[key]
        kept.append(OwnedPlayer(player_key=key, selling_price=selling))
        if row["bought"]:
            bank -= price
    moves = step.filter(pl.col("bought")).height
    if state.owned:
        free = next_free_transfers(state.free_transfers, moves, state.ft_bank_cap)
    else:
        free = state.free_transfers
    return state.model_copy(
        update={"owned": kept, "bank": bank, "free_transfers": free}
    )


def _actuals(frame: pl.DataFrame) -> pl.DataFrame:
    empty = pl.DataFrame(
        schema={
            "gw": pl.Int64(),
            "player_key": pl.Int64(),
            "actual_pts": pl.Int64(),
            "actual_minutes": pl.Int64(),
        }
    )
    if frame.is_empty():
        return empty
    if "actual_minutes" in frame.columns:
        return frame.group_by("gw", "player_key").agg(
            pl.col("actual_pts").sum(), pl.col("actual_minutes").sum()
        )
    return (
        frame.group_by("gw", "player_key")
        .agg(pl.col("actual_pts").sum())
        .with_columns(pl.lit(None).cast(pl.Int64).alias("actual_minutes"))
    )


def _player_names(view: PITView) -> dict[int, str]:
    """Map ``player_key`` to the latest visible ``web_name`` for this season."""
    frame = (
        view.latest("players", ["season", "player_key"])
        .filter(pl.col("season") == view.as_of.season)
        .select("player_key", "web_name")
        .collect()
    )
    return {int(key): str(name) for key, name in frame.iter_rows()}


def _element_keys(view: PITView) -> dict[int, int]:
    frame = (
        view.latest("players", ["season", "player_key"])
        .filter(pl.col("season") == view.as_of.season)
        .select("element_id", "player_key")
        .collect()
    )
    return {int(element): int(key) for element, key in frame.iter_rows()}


def _with_live_free_transfers(state: SquadState, team: CurrentTeam) -> SquadState:
    """Apply ``my-team`` free transfers when they fit the banking cap."""
    free = team.free_transfers
    if free is None or free < 1 or free > state.ft_bank_cap:
        return state
    return state.model_copy(update={"free_transfers": free})


def _empty_decisions() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "model_id": pl.Utf8(),
            "season": pl.Utf8(),
            "gw": pl.Int64(),
            "player_key": pl.Int64(),
            "position": pl.Utf8(),
            "in_xi": pl.Boolean(),
            "captain": pl.Boolean(),
            "vice": pl.Boolean(),
            "bought": pl.Boolean(),
            "sold": pl.Boolean(),
            "xpts": pl.Float64(),
            "actual_pts": pl.Int64(),
            "actual_minutes": pl.Int64(),
            "hit": pl.Float64(),
            "free_transfers": pl.Int64(),
        }
    )


def _decisions(frame: pl.DataFrame) -> pl.DataFrame:
    checked = DecisionSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked
