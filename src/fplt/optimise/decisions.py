"""Season decisions from a plan, and the reads that feed ``fplt plan``.

A decision backtest starts from an optimal GW1 draft. Only the first step of
each plan is played; later steps exist so a free transfer can be banked.
Predictions are whatever the ledger recorded at that deadline.

Chip policy is selected per run: ``off`` (chips-off MILP), ``rule_based``
(forced BB/TC/FH/WC on double/blank landmarks), or ``optimised``
(season-level ``schedule_chip_windows`` → short-horizon MILP under
``ChipRules``). Live ``run_plan`` / automation defaults to ``optimised``
(``DEFAULT_LIVE_CHIP_ARM``). The chips-off decision backtest
(``run_decisions`` / ``fplt decisions`` without ``--chip-board``) stays
``off`` for continuity. See ``run_chip_decision_board``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import polars as pl

from fplt.backtest.ledger import read_predictions
from fplt.data.fpl_client import CurrentTeam, load_current_team
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.chip_scheduler import schedule_chip_windows
from fplt.optimise.chip_windows import (
    ChipArm,
    ChipUseWindows,
    chip_arm_model_id,
    chip_use_windows_payload,
    format_chip_use_windows,
    is_counterfactual_rules,
    parse_chip_arm_model_id,
    rule_based_chip_windows,
    rules_label,
)
from fplt.optimise.highs_options import HighsSolveOptions
from fplt.optimise.plan import (
    GameweekPlan,
    next_free_transfers,
    plan_squad,
    plan_table,
    render_report,
    write_plan,
)
from fplt.optimise.price_forecast import (
    PriceForecastMethod,
    forecast_list_prices_from_view,
)
from fplt.optimise.prices import (
    apply_market_prices,
    fpl_selling_price,
    market_prices_from_candidates,
)
from fplt.optimise.realised import (
    assert_sane_mean_ppg,
    explain_gameweek,
    format_decision_scoreboard,
    model_board_label,
    summarise_decisions,
)
from fplt.optimise.scenarios import ScenarioBundle
from fplt.optimise.state import (
    ChipInventory,
    ChipPlay,
    OwnedPlayer,
    SquadState,
    apply_entry_picks,
    load_squad_state,
)
from fplt.schemas.plan import CandidateSchema, ChipWindowsSchema, DecisionSchema
from fplt.scoring.chips import ChipName, ChipRules, empty_remaining
from fplt.settings import Settings, get_settings
from fplt.store.runs import write_plan_run
from fplt.types import GWKey
from fplt.utils.frames import require_dataframe, require_not_none
from fplt.utils.timing import (
    TimingSession,
    binds_timing_artifacts,
    format_kind_summary,
    timed_phase,
    timing_session,
    write_timing_record,
)

TeamLoader = Callable[..., CurrentTeam]
logger = logging.getLogger(__name__)

# A blank gameweek drops a couple of clubs. Below half the candidate pool,
# the missing rows are being treated as zero and the plan is not usable.
_MIN_COVERAGE = 0.5
_CHIP_ARMS: tuple[ChipArm, ...] = ("off", "rule_based", "optimised")

# Live ``fplt plan`` / automation / ``catch_up`` / ``pre_deadline``. Chosen
# after the C6 three-arm board (optimised > chips-off > rule-based on
# 2025-26 H=3). Keep ``DEFAULT_PLAN_HORIZON`` at 3; do not flip from this.
DEFAULT_LIVE_CHIP_ARM: ChipArm = "optimised"

# Live within-horizon list prices. Kept off until the FDB-218 season-points
# promote (both seasons ≥ control); ``mean_gw_delta`` is the kept treatment.
# Decision boards still default off (omit ``--price-forecast``) for A/B.
DEFAULT_LIVE_PRICE_FORECAST_METHOD: PriceForecastMethod = "mean_gw_delta"

__all__ = [
    "DEFAULT_LIVE_CHIP_ARM",
    "DEFAULT_LIVE_PRICE_FORECAST_METHOD",
    "PlanExistsError",
    "advance_squad_from_plan",
    "assert_sane_mean_ppg",
    "candidates_as_of",
    "explain_decision_gameweek",
    "format_chip_decision_scoreboard",
    "format_decision_scoreboard",
    "model_board_label",
    "points_from_ledger",
    "read_chip_windows",
    "read_decisions",
    "resolve_squad_state",
    "run_chip_decision_board",
    "run_decision_bakeoff",
    "run_decisions",
    "run_plan",
    "simulate_decisions",
    "summarise_chip_decisions",
    "summarise_decisions",
    "write_chip_windows",
    "write_decisions",
    "prediction_rows_from_ledger",
    "actual_rows_from_ledger",
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
    checked = require_dataframe(checked)
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
    # Re-running backtest appends the same fixture; sum only distinct fixtures
    # so DGWs still accumulate while duplicate ledger rows do not inflate xpts.
    if "fixture_id" in chosen.columns:
        chosen = chosen.unique(
            subset=["as_of_gw", "target_gw", "player_key", "fixture_id"],
            keep="last",
        )
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
    chip_arm: ChipArm = "off",
    chip_rules_season: str | None = None,
    chip_windows: ChipUseWindows | None = None,
    fixtures: pl.DataFrame | None = None,
    force_play: bool = True,
    lineup_reopt: bool = True,
    chip_windows_log: dict[int, ChipUseWindows] | None = None,
    price_forecast_method: PriceForecastMethod | None = None,
    store: DataStore | None = None,
    highs_options: HighsSolveOptions | None = None,
    step_weights: Sequence[float] | None = None,
    scenario_for_gw: Callable[[int], ScenarioBundle] | None = None,
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
        chip_arm: ``off`` (chips disabled), ``rule_based`` (forced landmark
            chips), or ``optimised`` (MILP chooses under chip rules).
        chip_rules_season: Season whose ``configs/chips`` file to load when
            chips are on. Defaults to ``season``.
        chip_windows: Optional precomputed windows; when omitted, rule-based
            builds landmarks from ``fixtures`` and optimised rebuilds
            ``schedule_chip_windows`` each deadline.
        fixtures: Staged fixtures for rule-based / scheduled windows.
        force_play: When true (default), the optimised arm force-packs
            remaining half-window chips. Ignored for ``off`` / ``rule_based``.
        lineup_reopt: When true (default), re-choose XI / captain / vice after
            the transfer MILP via EV (or SAA when scenarios are passed into
            ``plan_squad``). Pass false for a raw-MILP lineup ablation.
        chip_windows_log: Optional mutable map filled with each deadline's
            ``ChipUseWindows`` (for ``ledger/chip_windows`` / explain dumps).
        price_forecast_method: Within-horizon list-price forecast. ``None``
            or ``flat`` keeps as-of prices for every step (default).
            ``mean_gw_delta`` needs ``store`` for PIT deadline history.
        store: Staged tables for PIT price forecasts when
            ``price_forecast_method`` is ``mean_gw_delta``.
        highs_options: HiGHS knobs forwarded to ``plan_squad``. Chip boards
            pass ``HighsSolveOptions.board_deterministic()`` (FDB-228).
        step_weights: Optional per-step objective weights for ``plan_squad``
            (skill shrink). ``None`` keeps geometric ``state.decay ** step``.
        scenario_for_gw: Optional ``gw -> ScenarioBundle`` for step-0 lineup
            SAA (FDB-241). When set, ``plan_squad`` receives joint samples
            instead of EV-only reopt on the actionable week.

    Returns:
        A frame matching ``DecisionSchema``.

    Raises:
        ValueError: A week has no candidates, the squad rules cannot be met,
            or ``mean_gw_delta`` is set without ``store``.
    """
    if chip_arm not in _CHIP_ARMS:
        raise ValueError(f"unknown chip_arm {chip_arm!r}")
    method = _normalize_price_forecast_method(price_forecast_method)
    if method == "mean_gw_delta" and store is None:
        raise ValueError("mean_gw_delta price forecasts require a DataStore")
    rules_season = chip_rules_season if chip_rules_season is not None else season
    rules = ChipRules.for_season(rules_season) if chip_arm != "off" else None
    static_windows = _static_windows_for_arm(
        chip_arm,
        fixtures=fixtures,
        season=season,
        rules=rules,
        predictions=predictions,
        gameweeks=gameweeks,
        chip_windows=chip_windows,
    )
    schedule_each_week = (
        chip_arm == "optimised" and chip_windows is None and static_windows is None
    )
    current = _ensure_chip_inventory(state, rules) if rules is not None else state
    scored = _actuals(actuals)
    rows: list[dict[str, object]] = []
    prior_plan: GameweekPlan | None = None
    for gameweek in gameweeks:
        window = [
            gameweek + offset for offset in range(horizon) if gameweek + offset <= 38
        ]
        pool = candidates.filter(pl.col("gw") == gameweek).drop("gw")
        if pool.is_empty():
            raise ValueError(f"no candidates for gw {gameweek}")
        # As-of sell-on refresh; optional within-horizon list forecasts below.
        current = apply_market_prices(current, market_prices_from_candidates(pool))
        active_windows = static_windows
        if schedule_each_week:
            rules = require_not_none(rules, what="rules")
            with timed_phase("plan/chips"):
                active_windows = _scheduled_windows(
                    fixtures=fixtures,
                    season=season,
                    rules=rules,
                    as_of_gw=gameweek,
                    short_horizon=len(window),
                    predictions=predictions,
                    pool=pool,
                    state=current,
                    gameweeks=gameweeks,
                    force_play=force_play,
                )
        if active_windows is not None:
            logger.info(
                "chip windows gw=%s %s",
                gameweek,
                format_chip_use_windows(active_windows),
            )
            if chip_windows_log is not None:
                chip_windows_log[gameweek] = active_windows
        forecast = _price_forecast_for_deadline(
            method=method,
            store=store,
            season=season,
            gameweek=gameweek,
            pool=pool,
            horizon=len(window),
        )
        started = time.perf_counter()
        scenarios = scenario_for_gw(gameweek) if scenario_for_gw is not None else None
        solved = plan_squad(
            pool,
            _window_points(predictions, gameweek, window),
            current,
            initial=not current.owned,
            horizon=len(window),
            season=season if chip_arm != "off" else None,
            gameweeks=window if chip_arm != "off" else None,
            chip_windows=active_windows if chip_arm != "off" else None,
            chip_rules_season=rules_season if chip_arm != "off" else None,
            warm_start=prior_plan,
            lineup_reopt=lineup_reopt,
            scenarios=scenarios,
            price_forecast=forecast,
            highs_options=highs_options,
            step_weights=None if step_weights is None else step_weights[: len(window)],
        )
        logger.info(
            "simulate gw=%s arm=%s horizon=%s pool=%s elapsed=%.2fs points=%.1f",
            gameweek,
            chip_arm,
            len(window),
            pool.height,
            time.perf_counter() - started,
            solved.points,
        )
        rows.extend(_decision_rows(solved, scored, model_id, season, gameweek))
        current = _advance(current, solved, pool, gameweek=gameweek, rules=rules)
        prior_plan = solved
    if not rows:
        return _empty_decisions()
    # Explicit schema: early null chips must not freeze the column as Null
    # before a later GW writes e.g. "bboost" (polars infer_schema_length).
    return _decisions(pl.DataFrame(rows, schema=_empty_decisions().schema))


def read_decisions(root: Path) -> pl.DataFrame:
    """Load ``ledger/decisions/part.parquet``, or an empty decision frame."""
    path = root / "ledger" / "decisions" / "part.parquet"
    if not path.exists():
        return _empty_decisions()
    return _decisions(pl.read_parquet(path))


def read_chip_windows(root: Path) -> pl.DataFrame:
    """Load ``ledger/chip_windows/part.parquet``, or an empty typed frame."""
    path = root / "ledger" / "chip_windows" / "part.parquet"
    if not path.exists():
        return _empty_chip_windows()
    return _chip_windows(pl.read_parquet(path))


def write_chip_windows(
    root: Path,
    *,
    model_id: str,
    season: str,
    windows_by_gw: dict[int, ChipUseWindows],
) -> Path:
    """Append per-deadline chip-window rows, replacing matching keys.

    Args:
        root: Data directory that contains ``ledger/``.
        model_id: Decision-ledger model id (may include ``::chip_*``).
        season: Season label.
        windows_by_gw: Deadlines → ``ChipUseWindows`` to persist.

    Returns:
        ``ledger/chip_windows/part.parquet``.
    """
    path = root / "ledger" / "chip_windows" / "part.parquet"
    if not windows_by_gw:
        return path
    rows = [
        {
            "model_id": model_id,
            "season": season,
            "gw": gameweek,
            **_chip_windows_row(windows),
        }
        for gameweek, windows in sorted(windows_by_gw.items())
    ]
    frame = _chip_windows(pl.DataFrame(rows, schema=_empty_chip_windows().schema))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pl.read_parquet(path)
        keys = frame.select("model_id", "season", "gw")
        kept = existing.join(keys, on=["model_id", "season", "gw"], how="anti")
        frame = pl.concat([kept, frame], how="diagonal_relaxed")
    frame.write_parquet(path)
    return path


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


@binds_timing_artifacts()
def run_plan(
    data_dir: Path,
    model_name: str,
    season: str,
    gameweek: int,
    horizon: int,
    entry_json: Path | None = None,
    squad: SquadState | None = None,
    *,
    chip_arm: ChipArm = DEFAULT_LIVE_CHIP_ARM,
    chip_rules_season: str | None = None,
    force_play: bool = True,
    price_forecast_method: PriceForecastMethod | None = (
        DEFAULT_LIVE_PRICE_FORECAST_METHOD
    ),
    step_weights: Sequence[float] | None = None,
    joint_scenarios: bool = False,
    n_scenario_sims: int | None = None,
    run_kind: str = "adhoc",
    checkpoint: str | None = None,
    line: str | None = None,
    squad_mode: str | None = None,
    parent_run_id: str | None = None,
    build_id: str | None = None,
    fp: str | None = None,
    prod_tag: str | None = None,
) -> tuple[GameweekPlan, Path, str, str]:
    """Build a plan from the ledger and write the parquet and the report.

    When ``squad`` is omitted, squad state comes from ``resolve_squad_state``:
    ``entry_json`` if provided, else the live / public team when
    ``FPLT_ENTRY_ID`` is set, else the YAML draft.

    Chip policy defaults to ``DEFAULT_LIVE_CHIP_ARM`` (``optimised``): the
    season chip scheduler rebuilds ``ChipUseWindows`` for this deadline, then
    the short-horizon MILP chooses under ``ChipRules``. Pass ``chip_arm="off"``
    for the pre-C6 chips-off path.

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
        chip_arm: Chip policy (default ``optimised`` for live / automation).
        chip_rules_season: Chip-rules year when chips are on. Defaults to
            ``season``.
        force_play: When true (default), the optimised scheduler force-packs
            remaining half-window chips. Pass false for ablations
            (``fplt plan --no-force-play``). Does not change the live default.
        price_forecast_method: Within-horizon list-price forecast. Defaults to
            ``DEFAULT_LIVE_PRICE_FORECAST_METHOD`` (``mean_gw_delta`` after
            the FDB-218 keep). Pass ``None`` / ``flat`` for as-of-only prices.
        step_weights: Optional per-step objective weights (skill shrink
            ablation). ``None`` keeps geometric ``SquadState.decay``.
        joint_scenarios: When true, fit the model and pass joint MC samples
            into step-0 lineup SAA (FDB-241; opt-in until a keep flip).
        n_scenario_sims: Draw count for joint SAA.
        run_kind: Dual-write run kind (``adhoc`` or ``official``).
        checkpoint: Ops checkpoint label for official runs.
        line: Official line name when ``run_kind="official"``.
        squad_mode: ``real`` / ``virtual`` for official plan runs.
        parent_run_id: Predict run id this plan reads.
        build_id: Curated build id recorded on the plan run (#346).
        fp: Run fingerprint for skip-if-done (#346).
        prod_tag: Moving ops tag label when known.

    Returns:
        The plan, the parquet path, the markdown report, and the dual-write
        ``run_id``.

    Raises:
        KeyError: The spec name or the gameweek is unknown.
        PlanExistsError: This deadline and horizon were already written.
            ``report`` on the exception is the summary.
        FPLAPIError: Live load fails when ``FPLT_ENTRY_ID`` is set and no
            ``entry_json`` / ``squad`` was provided.
        ValueError: This deadline has no predictions, a horizon week is
            missing or all zero, a week covers fewer than half the
            candidates, ``chip_arm`` is unknown, or optimised / rule-based
            chips lack staged fixtures.
    """
    if chip_arm not in _CHIP_ARMS:
        raise ValueError(f"unknown chip_arm {chip_arm!r}")
    logger.info(
        "run_plan model=%s season=%s gw=%s horizon=%s chip_arm=%s force_play=%s "
        "joint_scenarios=%s price_forecast=%s skill_shrink=%s",
        model_name,
        season,
        gameweek,
        horizon,
        chip_arm,
        force_play,
        joint_scenarios,
        price_forecast_method,
        step_weights is not None,
    )
    plan_timings = TimingSession()
    plan_run_id = ""
    with timing_session(plan_timings):
        with timed_phase("plan/setup"):
            spec = load_model_specs()[model_name]
            store = DataStore(data_dir)
            view = PITData(store).view(GWKey(season, gameweek))
            if squad is not None:
                state = squad
                logger.info("using pre-resolved squad (owned=%s)", len(state.owned))
            else:
                state = resolve_squad_state(
                    view,
                    entry_json=entry_json,
                    event=gameweek,
                )
            window = [
                gameweek + offset
                for offset in range(horizon)
                if gameweek + offset <= 38
            ]
            candidates = candidates_as_of(view)
            ledger = read_predictions(data_dir)
            points = points_from_ledger(ledger, spec.model_id, season, gameweek, window)
            logger.info(
                "optimiser inputs: candidates=%s prediction_rows=%s "
                "window=%s initial=%s",
                candidates.height,
                points.height,
                window,
                not state.owned,
            )
            _require_usable_points(points, candidates, window, model_name)
            state = apply_market_prices(
                state, market_prices_from_candidates(candidates)
            )
            rules_season = (
                chip_rules_season if chip_rules_season is not None else season
            )
            plan_season: str | None = None
            plan_gameweeks: list[int] | None = None
            plan_windows: ChipUseWindows | None = None
            plan_rules_season: str | None = None
        if chip_arm != "off":
            with timed_phase("plan/chips"):
                rules = ChipRules.for_season(rules_season)
                state = _ensure_chip_inventory(state, rules)
                fixtures = store.read("fixtures").filter(pl.col("season") == season)
                predictions = _prediction_rows(ledger, spec.model_id, season)
                rest_of_season = list(range(gameweek, 39))
                if chip_arm == "optimised":
                    plan_windows = _scheduled_windows(
                        fixtures=fixtures,
                        season=season,
                        rules=rules,
                        as_of_gw=gameweek,
                        short_horizon=len(window),
                        predictions=predictions,
                        pool=candidates,
                        state=state,
                        gameweeks=rest_of_season,
                        force_play=force_play,
                    )
                else:
                    plan_windows = _static_windows_for_arm(
                        chip_arm,
                        fixtures=fixtures,
                        season=season,
                        rules=rules,
                        predictions=predictions,
                        gameweeks=rest_of_season,
                        chip_windows=None,
                    )
                plan_season = season
                plan_gameweeks = window
                plan_rules_season = rules_season
        if plan_windows is not None:
            logger.info(
                "chip windows gw=%s %s",
                gameweek,
                format_chip_use_windows(plan_windows),
            )
            write_chip_windows(
                data_dir,
                model_id=chip_arm_model_id(spec.model_id, chip_arm)
                if chip_arm != "off"
                else spec.model_id,
                season=season,
                windows_by_gw={gameweek: plan_windows},
            )
        forecast = _price_forecast_for_deadline(
            method=_normalize_price_forecast_method(price_forecast_method),
            store=store,
            season=season,
            gameweek=gameweek,
            pool=candidates,
            horizon=len(window),
        )
        scenarios = None
        if joint_scenarios:
            from fplt.optimise.scenarios import (
                DEFAULT_SAA_SIMS,
                joint_scenarios_for_deadline,
            )

            scenarios = joint_scenarios_for_deadline(
                spec,
                data_dir,
                season,
                gameweek,
                n_sims=DEFAULT_SAA_SIMS if n_scenario_sims is None else n_scenario_sims,
            )
        solved = plan_squad(
            candidates,
            points,
            state,
            initial=not state.owned,
            horizon=len(window),
            season=plan_season,
            gameweeks=plan_gameweeks,
            chip_windows=plan_windows,
            chip_rules_season=plan_rules_season,
            scenarios=scenarios,
            price_forecast=forecast,
            step_weights=None if step_weights is None else step_weights[: len(window)],
        )
        with timed_phase("plan/write"):
            table = plan_table(solved, spec.model_id, season, gameweek, window)
            names = _player_names(view)
            report = render_report(solved, season, window, names=names)
            if plan_windows is not None:
                windows_line = format_chip_use_windows(plan_windows)
                report = f"{report}\n\n## Chip windows\n\n{windows_line}\n"
            report = _with_explain_sections(
                report,
                solved=solved,
                state=state,
                candidates=candidates,
                data_dir=data_dir,
                model_id=spec.model_id,
                season=season,
                gameweek=gameweek,
                names=names,
            )
            try:
                from fplt.automation.release import resolve_prod_tag_label

                path = write_plan(
                    data_dir,
                    table,
                    prod_tag=resolve_prod_tag_label(),
                )
            except FileExistsError as exc:
                raise PlanExistsError(Path(str(exc)), report) from exc
            (path.parent / "report.md").write_text(report, encoding="utf-8")
            from fplt.schemas.manifests import RunKind, SquadMode

            plan_run = write_plan_run(
                data_dir,
                spec=spec,
                as_of=GWKey(season, gameweek),
                deadline_utc=store.deadline(GWKey(season, gameweek)),
                horizon=len(window),
                frame=table,
                kind=cast(RunKind, run_kind),
                checkpoint=checkpoint,
                line=line,
                squad_mode=cast(SquadMode | None, squad_mode),
                parent_run_id=parent_run_id,
                build_id=build_id,
                fp=fp,
                prod_tag=prod_tag,
            )
            plan_run_id = plan_run.run_id
            logger.info("dual-wrote plan run_id=%s", plan_run_id)
    meta = {
        "model": model_name,
        "season": season,
        "gw": gameweek,
        "horizon": horizon,
        "chip_arm": chip_arm,
    }
    logger.info(format_kind_summary("plan", plan_timings, meta=meta))
    write_timing_record("plan", plan_timings, meta=meta)
    logger.info(
        "plan written path=%s hits=%s points=%.2f chips=%s",
        path,
        solved.hits,
        solved.points,
        solved.chips,
    )
    return solved, path, report, plan_run_id


def _with_explain_sections(
    report: str,
    *,
    solved: GameweekPlan,
    state: SquadState,
    candidates: pl.DataFrame,
    data_dir: Path,
    model_id: str,
    season: str,
    gameweek: int,
    names: dict[int, str],
) -> str:
    """Append decision / forecast / depth explain blocks when possible.

    Decision baseline failures no longer drop the forecast board: soft-fail
    lives inside ``build_decision_explain`` / ``explain_plan_markdown``. A
    missing prediction ledger still skips the whole append.
    """
    from fplt.data.pit import PITData
    from fplt.data.store import DataStore
    from fplt.models.spec import load_model_specs
    from fplt.optimise.explain import (
        append_explain_sections,
        explain_plan_markdown,
        load_layered_explain_parts,
    )
    from fplt.scoring.rules import Rules
    from fplt.types import GWKey

    ledger = read_predictions(data_dir)
    preds = ledger.filter(
        (pl.col("model_id") == model_id)
        & (pl.col("as_of_season") == season)
        & (pl.col("as_of_gw") == gameweek)
        & (pl.col("target_season") == season)
        & (pl.col("target_gw") == gameweek)
    )
    if preds.is_empty():
        logger.warning(
            "explain sections skipped: no predictions for model_id=%s %s GW%s",
            model_id,
            season,
            gameweek,
        )
        return report
    store = DataStore(data_dir)
    view = PITData(store).view(GWKey(season, gameweek))
    model_name: str | None = None
    for name, spec in load_model_specs().items():
        if spec.model_id == model_id:
            model_name = name
            break
    parts = (
        load_layered_explain_parts(
            data_dir, model_name, season, gameweek, fit_if_missing=False
        )
        if model_name is not None
        else None
    )
    try:
        explain = explain_plan_markdown(
            solved,
            state,
            candidates,
            preds,
            season=season,
            gameweek=gameweek,
            rules=Rules.for_season(season),
            names=names,
            view=view,
            parts=parts,
        )
    except (ValueError, KeyError) as exc:
        # Board build can still raise (e.g. missing columns); keep plan report.
        logger.warning("explain sections skipped: %s", exc)
        return report
    return append_explain_sections(report, decision=explain)


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


@binds_timing_artifacts()
def run_decisions(
    data_dir: Path,
    model_name: str,
    season: str,
    from_gw: int,
    to_gw: int,
    horizon: int,
    *,
    chip_arm: ChipArm = "off",
    chip_rules_season: str | None = None,
    force_play: bool = True,
    lineup_reopt: bool = True,
    price_forecast_method: PriceForecastMethod | None = None,
    highs_options: HighsSolveOptions | None = None,
    step_weights: Sequence[float] | None = None,
    joint_scenarios: bool = False,
    n_scenario_sims: int | None = None,
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
        chip_arm: Chip policy for this run (default chips off).
        chip_rules_season: Chip-rules season when chips are on.
        force_play: Forwarded to the optimised chip scheduler (default true).
        lineup_reopt: Forwarded to ``plan_squad`` (default true; pass false
            for a raw-MILP lineup ablation).
        price_forecast_method: Within-horizon list-price forecast method
            (default off / as-of flat).
        highs_options: Forwarded to ``simulate_decisions`` / ``plan_squad``.
        step_weights: Optional per-step objective weights (skill shrink).
        joint_scenarios: When true, fit the model each deadline and pass
            joint MC samples into step-0 lineup SAA (FDB-241; opt-in).
        n_scenario_sims: Draw count for joint SAA (default
            ``DEFAULT_SAA_SIMS``).

    Returns:
        The decision parquet path.

    Raises:
        KeyError: The spec name or a gameweek deadline is unknown.
    """
    spec = load_model_specs()[model_name]
    squad_rules = load_squad_state()
    draft = squad_rules.model_copy(
        update={"owned": [], "bank": 1000, "free_transfers": 1}
    )
    rules_season = chip_rules_season if chip_rules_season is not None else season
    if chip_arm != "off":
        counts = empty_remaining()
        counts.update(ChipRules.for_season(rules_season).remaining_counts())
        draft = draft.model_copy(
            update={"chips": ChipInventory(remaining=counts, played=[])}
        )
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
    fixtures = (
        store.read("fixtures").filter(pl.col("season") == season)
        if chip_arm != "off"
        else None
    )
    # Chips-off keeps the historical model_id so existing boards stay stable.
    ledger_id = (
        spec.model_id
        if chip_arm == "off"
        else chip_arm_model_id(spec.model_id, chip_arm)
    )
    windows_log: dict[int, ChipUseWindows] = {}
    scenario_for_gw = None
    if joint_scenarios:
        from fplt.optimise.scenarios import (
            DEFAULT_SAA_SIMS,
            make_joint_scenario_provider,
        )

        scenario_for_gw = make_joint_scenario_provider(
            spec,
            store,
            season,
            n_sims=DEFAULT_SAA_SIMS if n_scenario_sims is None else n_scenario_sims,
        )
    decision_timings = TimingSession()
    with timing_session(decision_timings):
        frame = simulate_decisions(
            pl.concat(pools),
            predictions,
            actuals,
            draft,
            model_id=ledger_id,
            season=season,
            gameweeks=gameweeks,
            horizon=horizon,
            chip_arm=chip_arm,
            chip_rules_season=chip_rules_season,
            fixtures=fixtures,
            force_play=force_play,
            lineup_reopt=lineup_reopt,
            chip_windows_log=windows_log,
            price_forecast_method=price_forecast_method,
            store=store,
            highs_options=highs_options,
            step_weights=step_weights,
            scenario_for_gw=scenario_for_gw,
        )
        path = write_decisions(data_dir, frame)
        if windows_log:
            write_chip_windows(
                data_dir,
                model_id=ledger_id,
                season=season,
                windows_by_gw=windows_log,
            )
    meta = {
        "model": model_name,
        "season": season,
        "n_gw": len(gameweeks),
        "horizon": horizon,
        "chip_arm": chip_arm,
    }
    logger.info(format_kind_summary("decisions", decision_timings, meta=meta))
    write_timing_record("decisions", decision_timings, meta=meta)
    return path


def run_chip_decision_board(
    data_dir: Path,
    models: list[str],
    season: str,
    from_gw: int,
    to_gw: int,
    horizon: int,
    *,
    arms: Sequence[ChipArm] = _CHIP_ARMS,
    chip_rules_season: str | None = None,
    simulate: bool = True,
    max_mean_ppg: float | None = None,
    force_play: bool = True,
    lineup_reopt: bool = True,
    price_forecast_method: PriceForecastMethod | None = None,
    highs_options: HighsSolveOptions | None = None,
    step_weights: Sequence[float] | None = None,
    joint_scenarios: bool = False,
    n_scenario_sims: int | None = None,
) -> pl.DataFrame:
    """Run the three-arm chip decision board and return a season scoreboard.

    Arms are chips-off, rule-based landmarks, and optimised MILP chips. Season
    points use current chip scoring (TC ×3, BB bench). When
    ``chip_rules_season`` differs from ``season``, rows are labelled as a
    counterfactual (e.g. 2024-25 under 2025/26 rules).

    Args:
        data_dir: Root with ``staged/`` and ``ledger/``.
        models: YAML spec names.
        season: Fixture / prediction season.
        from_gw: First gameweek.
        to_gw: Last gameweek, inclusive.
        horizon: Planner horizon.
        arms: Chip arms to include (default all three).
        chip_rules_season: Chip-rules year (default ``season``).
        simulate: When true, simulate each model × arm first.
        max_mean_ppg: Optional mean pts/GW sanity gate.
        force_play: When true (default), optimised arms force-pack remaining
            half-window chips. Pass false for a no-force-play ablation.
        lineup_reopt: When true (default), post-MILP EV/SAA lineup re-opt.
            Pass false for a raw-MILP XI/captain ablation.
        price_forecast_method: Within-horizon list-price forecast method
            (default off / as-of flat).
        highs_options: HiGHS knobs for board solves. Defaults to
            ``HighsSolveOptions.board_deterministic()`` (no time limit,
            ``threads=1``, fixed seed) so identical tip + data → identical
            totals.
        step_weights: Optional per-step objective weights (skill shrink).
        joint_scenarios: Opt-in joint MC samples for step-0 SAA (FDB-241).
        n_scenario_sims: Draw count when ``joint_scenarios`` is on.

    Returns:
        Scoreboard from ``summarise_chip_decisions``.

    Raises:
        ValueError: ``models`` or ``arms`` is empty.
    """
    if not models:
        raise ValueError("models must be non-empty")
    if not arms:
        raise ValueError("arms must be non-empty")
    rules_season = chip_rules_season if chip_rules_season is not None else season
    board_highs = (
        HighsSolveOptions.board_deterministic()
        if highs_options is None
        else highs_options
    )
    if simulate:
        for name in models:
            for arm in arms:
                logger.info(
                    "chip board simulate model=%s arm=%s season=%s rules=%s "
                    "force_play=%s lineup_reopt=%s joint_scenarios=%s "
                    "price_forecast=%s skill_shrink=%s highs_seed=%s "
                    "highs_threads=%s",
                    name,
                    arm,
                    season,
                    rules_season,
                    force_play,
                    lineup_reopt,
                    joint_scenarios,
                    price_forecast_method,
                    step_weights is not None,
                    board_highs.random_seed,
                    board_highs.threads,
                )
                run_decisions(
                    data_dir,
                    name,
                    season,
                    from_gw,
                    to_gw,
                    horizon,
                    chip_arm=arm,
                    chip_rules_season=rules_season,
                    force_play=force_play,
                    lineup_reopt=lineup_reopt,
                    price_forecast_method=price_forecast_method,
                    highs_options=board_highs,
                    step_weights=step_weights,
                    joint_scenarios=joint_scenarios,
                    n_scenario_sims=n_scenario_sims,
                )
    specs = load_model_specs()
    id_to_name = {specs[name].model_id: name for name in models}
    wanted: set[str] = set()
    for name in models:
        base = specs[name].model_id
        for arm in arms:
            wanted.add(base if arm == "off" else chip_arm_model_id(base, arm))
    frame = read_decisions(data_dir).filter(
        (pl.col("season") == season) & pl.col("model_id").is_in(list(wanted))
    )
    if from_gw > 1 or to_gw < 38:
        frame = frame.filter((pl.col("gw") >= from_gw) & (pl.col("gw") <= to_gw))
    summary = summarise_chip_decisions(
        frame, season=season, chip_rules_season=rules_season
    )
    if not summary.is_empty():
        summary = summary.with_columns(
            pl.col("base_model_id")
            .replace_strict(id_to_name, default=None)
            .alias("model_name")
        )
    if max_mean_ppg is not None:
        assert_sane_mean_ppg(
            summary.select(
                "model_id",
                "season",
                "total_points",
                "gameweeks",
                "per_gw",
            ),
            max_mean=max_mean_ppg,
        )
    return summary


def summarise_chip_decisions(
    frame: pl.DataFrame,
    *,
    season: str,
    chip_rules_season: str,
) -> pl.DataFrame:
    """Aggregate chip-arm decision rows with rules / counterfactual labels.

    Args:
        frame: Decision-ledger rows (possibly several ``::chip_*`` model ids).
        season: Fixture season.
        chip_rules_season: Chip-rules season used for the board.

    Returns:
        ``model_id``, ``base_model_id``, ``chip_arm``, ``season``,
        ``rules_label``, ``counterfactual``, ``total_points``, ``gameweeks``,
        ``per_gw``.
    """
    base = summarise_decisions(frame)
    if base.is_empty():
        return pl.DataFrame(
            schema={
                "model_id": pl.Utf8(),
                "base_model_id": pl.Utf8(),
                "chip_arm": pl.Utf8(),
                "season": pl.Utf8(),
                "rules_label": pl.Utf8(),
                "counterfactual": pl.Boolean(),
                "total_points": pl.Int64(),
                "gameweeks": pl.Int64(),
                "per_gw": pl.List(pl.Int64()),
            }
        )
    label = rules_label(season, chip_rules_season)
    counterfactual = is_counterfactual_rules(season, chip_rules_season)
    rows: list[dict[str, object]] = []
    for row in base.iter_rows(named=True):
        model_id = str(row["model_id"])
        base_id, arm = parse_chip_arm_model_id(model_id)
        rows.append(
            {
                "model_id": model_id,
                "base_model_id": base_id,
                "chip_arm": arm if arm is not None else "off",
                "season": str(row["season"]),
                "rules_label": label,
                "counterfactual": counterfactual,
                "total_points": int(row["total_points"]),
                "gameweeks": int(row["gameweeks"]),
                "per_gw": row["per_gw"],
            }
        )
    return pl.DataFrame(rows).sort("total_points", descending=True)


def format_chip_decision_scoreboard(summary: pl.DataFrame) -> str:
    """Render the three-arm chip board as a markdown table.

    When ``model_name`` is present, the Model column shows
    ``name (base_model_id)`` so boards survive in-place ModelSpec edits.
    """
    if summary.is_empty():
        return "_No chip-decision rows to score._"
    lines = [
        "| Rank | Model | Arm | Rules | Points | GWs | Mean/GW |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(summary.iter_rows(named=True), start=1):
        gameweeks = int(row["gameweeks"])
        mean = float(row["total_points"]) / gameweeks if gameweeks else 0.0
        rules = str(row["rules_label"])
        if bool(row["counterfactual"]):
            rules = f"{rules} (counterfactual)"
        name = row.get("model_name")
        label = model_board_label(
            str(row["base_model_id"]),
            model_name=str(name) if name is not None else None,
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    label,
                    str(row["chip_arm"]),
                    rules,
                    str(row["total_points"]),
                    str(gameweeks),
                    f"{mean:.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def run_decision_bakeoff(
    data_dir: Path,
    models: list[str],
    season: str,
    from_gw: int,
    to_gw: int,
    horizon: int,
    *,
    simulate: bool = True,
    max_mean_ppg: float | None = None,
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
        max_mean_ppg: When set, fail if any model's mean pts/GW exceeds this.

    Returns:
        Summary from ``summarise_decisions`` for ``season`` and ``models``.

    Raises:
        KeyError: An unknown model name.
        ValueError: ``models`` is empty, or mean pts/GW exceeds ``max_mean_ppg``.
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
    id_to_name = {specs[name].model_id: name for name in models}
    frame = read_decisions(data_dir).filter(
        (pl.col("season") == season) & pl.col("model_id").is_in(list(id_to_name))
    )
    if from_gw > 1 or to_gw < 38:
        frame = frame.filter((pl.col("gw") >= from_gw) & (pl.col("gw") <= to_gw))
    summary = summarise_decisions(frame)
    if not summary.is_empty():
        summary = summary.with_columns(
            pl.col("model_id")
            .replace_strict(id_to_name, default=None)
            .alias("model_name")
        )
    if max_mean_ppg is not None:
        assert_sane_mean_ppg(summary, max_mean=max_mean_ppg)
    return summary


def explain_decision_gameweek(
    data_dir: Path,
    model_name: str,
    season: str,
    gameweek: int,
) -> str:
    """Explain one model × gameweek net from ``ledger/decisions``.

    Args:
        data_dir: Root that contains ``ledger/decisions``.
        model_name: YAML spec name.
        season: Season label.
        gameweek: Gameweek to dump.

    Returns:
        Text from ``explain_gameweek`` for that squad.

    Raises:
        KeyError: Unknown model name.
        ValueError: No decision rows for that model × gameweek.
    """
    spec = load_model_specs()[model_name]
    part = read_decisions(data_dir).filter(
        (pl.col("model_id") == spec.model_id)
        & (pl.col("season") == season)
        & (pl.col("gw") == gameweek)
    )
    if part.is_empty():
        raise ValueError(
            f"no decision rows for {model_name} ({spec.model_id}) {season} GW{gameweek}"
        )
    hit = float(part["hit"][0])
    store = DataStore(data_dir)
    names = _player_names(PITData(store).view(GWKey(season, gameweek)))
    header = (
        f"# {model_name} ({spec.model_id}) {season} GW{gameweek}\n"
        f"hit={hit:.0f} free_transfers={int(part['free_transfers'][0])}\n"
    )
    body = header + explain_gameweek(part, hit=hit, names=names)
    windows_note = _explain_chip_windows_note(
        data_dir, model_id=spec.model_id, season=season, gameweek=gameweek
    )
    if windows_note is None:
        return body
    return f"{body}\n\n{windows_note}"


def prediction_rows_from_ledger(
    ledger: pl.DataFrame, model_id: str, season: str
) -> pl.DataFrame:
    """Collapse prediction-ledger xPts to planner grain for one model/season.

    Deduplicates per fixture before summing so walk-forward re-appends and
    double gameweeks do not inflate ``xpts``.

    Args:
        ledger: Prediction ledger frame (or empty).
        model_id: Content-addressed model id.
        season: ``as_of_season`` / ``target_season`` to keep.

    Returns:
        ``as_of_gw``, ``target_gw``, ``player_key``, ``xpts``.
    """
    return _prediction_rows(ledger, model_id, season)


def actual_rows_from_ledger(
    ledger: pl.DataFrame, model_id: str, season: str
) -> pl.DataFrame:
    """Collapse prediction-ledger actuals to one row per player-gameweek.

    Deduplicates per fixture before summing so walk-forward ledgers do not
    double-count realised points.

    Args:
        ledger: Prediction ledger frame (or empty).
        model_id: Content-addressed model id.
        season: ``target_season`` to keep.

    Returns:
        ``gw``, ``player_key``, ``actual_pts``, ``actual_minutes``.
    """
    return _actual_rows(ledger, model_id, season)


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
    # Same fixture may appear under every backtest append; keep one row per
    # fixture before summing so DGWs add and re-runs do not double xpts.
    if "fixture_id" in chosen.columns:
        chosen = chosen.unique(
            subset=["as_of_gw", "target_gw", "player_key", "fixture_id"],
            keep="last",
        )
    return chosen.group_by("as_of_gw", "target_gw", "player_key").agg(
        pl.col("xpts").sum()
    )


def _actual_rows(ledger: pl.DataFrame, model_id: str, season: str) -> pl.DataFrame:
    """Collapse ledger actuals to one row per player-gameweek.

    Walk-forward ledgers store the same fixture under every ``as_of`` that
    predicted it (``steps=2`` → two rows). Summing those raw rows doubles
    realised points. Deduplicate per fixture first, then sum across fixtures
    so double gameweeks still accumulate correctly.
    """
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
    scored = chosen.filter(pl.col("actual_pts").is_not_null())
    if scored.is_empty():
        return empty
    # Same fixture appears once per predicting deadline; keep one copy.
    if "fixture_id" in scored.columns:
        scored = scored.sort("as_of_gw", "horizon").unique(
            subset=["target_gw", "player_key", "fixture_id"],
            keep="last",
        )
    elif "horizon" in scored.columns:
        scored = scored.filter(pl.col("horizon") == 1)
    if "actual_minutes" in scored.columns:
        return (
            scored.group_by("target_gw", "player_key")
            .agg(pl.col("actual_pts").sum(), pl.col("actual_minutes").sum())
            .rename({"target_gw": "gw"})
        )
    return (
        scored.group_by("target_gw", "player_key")
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
    chip = plan.chips[0] if plan.chips else None
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
                "chip": chip,
            }
        )
    return rows


def _normalize_price_forecast_method(
    method: PriceForecastMethod | None,
) -> PriceForecastMethod | None:
    """Treat ``flat`` as off (same cash constraints as ``None``)."""
    if method is None or method == "flat":
        return None
    if method == "mean_gw_delta":
        return method
    raise ValueError(f"unknown price forecast method: {method!r}")


def _price_forecast_for_deadline(
    *,
    method: PriceForecastMethod | None,
    store: DataStore | None,
    season: str,
    gameweek: int,
    pool: pl.DataFrame,
    horizon: int,
) -> pl.DataFrame | None:
    """Build a PIT list-price forecast for one planning deadline."""
    if method is None:
        return None
    store = require_not_none(store, what="store")
    view = PITData(store).view(GWKey(season, gameweek))
    as_of = market_prices_from_candidates(pool)
    return forecast_list_prices_from_view(view, as_of, horizon, method=method)


def advance_squad_from_plan(
    state: SquadState,
    plan: GameweekPlan,
    pool: pl.DataFrame,
    *,
    gameweek: int,
    rules: ChipRules | None = None,
) -> SquadState:
    """Apply week-1 plan actions to ``state`` (transfers, bank, FTs, chips).

    Shared by ``simulate_decisions`` and the ensemble-sensitivity board so
    ownership advances identically when comparing forecast members on a
    shared path.

    Args:
        state: Pre-deadline squad.
        plan: Solved ``GameweekPlan`` (step 0 is acted on).
        pool: Candidate pool for the deadline (unused; prices come from picks).
        gameweek: Deadline gameweek for chip history.
        rules: Chip rules when chips are on; ``None`` for chips-off.

    Returns:
        Post-deadline ``SquadState``.
    """
    return _advance(state, plan, pool, gameweek=gameweek, rules=rules)


def _advance(
    state: SquadState,
    plan: GameweekPlan,
    pool: pl.DataFrame,
    *,
    gameweek: int,
    rules: ChipRules | None = None,
) -> SquadState:
    del pool  # prices come from the plan pick rows
    chip = plan.chips[0] if plan.chips else None
    step = plan.picks.filter(pl.col("step") == 0)
    if chip == "freehit":
        # Temporary squad for scoring only; permanent ownership and bank stay.
        free = _next_free_after_chip(state, chip=chip, moves=0, rules=rules)
        return state.model_copy(
            update={
                "free_transfers": free,
                "chips": _consume_chip(state.chips, chip, gameweek),
            }
        )

    previous = {player.player_key: player for player in state.owned}
    kept: list[OwnedPlayer] = []
    bank = state.bank
    for row in step.iter_rows(named=True):
        key = int(row["player_key"])
        price = int(row["price"])
        if row["sold"]:
            prior = previous[key]
            bank += fpl_selling_price(prior.purchase_price, price)
            continue
        if row["bought"] or key not in previous:
            purchase = price
        else:
            purchase = previous[key].purchase_price
        selling = fpl_selling_price(purchase, price)
        kept.append(
            OwnedPlayer(
                player_key=key,
                purchase_price=purchase,
                selling_price=selling,
            )
        )
        if row["bought"]:
            bank -= price
    moves = step.filter(pl.col("bought")).height
    if state.owned:
        free = _next_free_after_chip(state, chip=chip, moves=moves, rules=rules)
    else:
        free = state.free_transfers
    chips = (
        _consume_chip(state.chips, chip, gameweek) if chip is not None else state.chips
    )
    return state.model_copy(
        update={
            "owned": kept,
            "bank": bank,
            "free_transfers": free,
            "chips": chips,
        }
    )


def _next_free_after_chip(
    state: SquadState,
    *,
    chip: str | None,
    moves: int,
    rules: ChipRules | None,
) -> int:
    if chip == "wildcard":
        return 1
    if chip == "freehit":
        keep = True if rules is None else rules.free_hit_keeps_free_transfers
        if keep:
            return next_free_transfers(state.free_transfers, 0, state.ft_bank_cap)
        return 1
    return next_free_transfers(state.free_transfers, moves, state.ft_bank_cap)


def _consume_chip(inventory: ChipInventory, chip: str, gameweek: int) -> ChipInventory:
    name: ChipName
    if chip not in {"wildcard", "freehit", "bboost", "3xc"}:
        raise ValueError(f"unknown chip {chip!r}")
    name = chip  # type: ignore[assignment]
    remaining = dict(inventory.remaining)
    if remaining.get(name, 0) <= 0:
        raise ValueError(f"no remaining uses for chip {name!r}")
    remaining[name] = remaining[name] - 1
    played = list(inventory.played) + [ChipPlay(name=name, gw=gameweek)]
    return ChipInventory(remaining=remaining, played=played)


def _ensure_chip_inventory(state: SquadState, rules: ChipRules) -> SquadState:
    """Fill chip remaining from ``rules`` when the squad still has an empty bag.

    Preserves an explicit mid-season inventory (non-zero remaining or played
    history) so tests and live state can start part-way through the season.
    """
    remaining = state.chips.remaining
    if state.chips.played or any(count > 0 for count in remaining.values()):
        return state
    counts = empty_remaining()
    counts.update(rules.remaining_counts())
    return state.model_copy(
        update={"chips": ChipInventory(remaining=counts, played=[])}
    )


def _static_windows_for_arm(
    arm: ChipArm,
    *,
    fixtures: pl.DataFrame | None,
    season: str,
    rules: ChipRules | None,
    predictions: pl.DataFrame,
    gameweeks: list[int],
    chip_windows: ChipUseWindows | None,
) -> ChipUseWindows | None:
    """Return precomputed windows, or ``None`` when optimised schedules weekly."""
    if arm == "off":
        return None
    if chip_windows is not None:
        return chip_windows
    if arm == "optimised":
        # Rebuilt each deadline in ``simulate_decisions`` via
        # ``schedule_chip_windows`` (option value + squad continuity).
        return None
    if rules is None:
        raise ValueError("rule_based chip arm requires chip rules")
    if fixtures is None or fixtures.is_empty():
        raise ValueError("rule_based chip arm requires fixtures")
    pred = predictions
    if "gw" not in pred.columns and "target_gw" in pred.columns:
        pred = pred.rename({"target_gw": "gw"})
    return rule_based_chip_windows(
        fixtures,
        season=season,
        rules=rules,
        predictions=pred,
        gameweeks=gameweeks,
    )


def _scheduled_windows(
    *,
    fixtures: pl.DataFrame | None,
    season: str,
    rules: ChipRules,
    as_of_gw: int,
    short_horizon: int,
    predictions: pl.DataFrame,
    pool: pl.DataFrame,
    state: SquadState,
    gameweeks: list[int],
    force_play: bool = True,
) -> ChipUseWindows:
    """Build season-level windows for one deadline of the optimised arm."""
    if fixtures is None or fixtures.is_empty():
        raise ValueError("optimised chip arm requires fixtures for the scheduler")
    pred = predictions
    if "gw" not in pred.columns and "target_gw" in pred.columns:
        pred = pred.rename({"target_gw": "gw"})
    player_teams = pool.select("player_key", "team_key")
    return schedule_chip_windows(
        fixtures,
        season=season,
        rules=rules,
        as_of_gw=as_of_gw,
        short_horizon=short_horizon,
        predictions=pred,
        player_teams=player_teams,
        owned_player_keys=[player.player_key for player in state.owned],
        inventory=state.chips,
        gameweeks=gameweeks,
        force_play=force_play,
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
            "chip": pl.Utf8(),
        }
    )


def _empty_chip_windows() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "model_id": pl.Utf8(),
            "season": pl.Utf8(),
            "gw": pl.Int64(),
            "allowed": pl.Utf8(),
            "forced": pl.Utf8(),
            "no_chip": pl.Utf8(),
        }
    )


def _chip_windows(frame: pl.DataFrame) -> pl.DataFrame:
    checked = ChipWindowsSchema.validate(frame)
    checked = require_dataframe(checked)
    return checked


def _chip_windows_row(windows: ChipUseWindows) -> dict[str, str]:
    payload = chip_use_windows_payload(windows)
    return {
        "allowed": json.dumps(payload["allowed"], sort_keys=True),
        "forced": json.dumps(payload["forced"], sort_keys=True),
        "no_chip": json.dumps(payload["no_chip"]),
    }


def _explain_chip_windows_note(
    data_dir: Path,
    *,
    model_id: str,
    season: str,
    gameweek: int,
) -> str | None:
    """Return an explain section for logged chip windows, if present."""
    frame = read_chip_windows(data_dir)
    if frame.is_empty():
        return None
    part = frame.filter(
        (pl.col("season") == season)
        & (pl.col("gw") == gameweek)
        & (
            (pl.col("model_id") == model_id)
            | pl.col("model_id").str.starts_with(f"{model_id}::chip_")
        )
    )
    if part.is_empty():
        return None
    # Prefer optimised arm rows when several chip-arm ids match.
    preferred = part.filter(pl.col("model_id").str.ends_with("::chip_optimised"))
    row = (preferred if not preferred.is_empty() else part).row(0, named=True)
    allowed = json.loads(str(row["allowed"]))
    forced = json.loads(str(row["forced"]))
    no_chip = json.loads(str(row["no_chip"]))
    windows = ChipUseWindows(
        allowed={chip: frozenset(weeks) for chip, weeks in allowed.items()},
        forced={chip: frozenset(weeks) for chip, weeks in forced.items()},
        no_chip=frozenset(no_chip),
    )
    return f"chip_windows ({row['model_id']}): {format_chip_use_windows(windows)}"


def _decisions(frame: pl.DataFrame) -> pl.DataFrame:
    if "chip" not in frame.columns:
        frame = frame.with_columns(pl.lit(None).cast(pl.Utf8).alias("chip"))
    checked = DecisionSchema.validate(frame)
    checked = require_dataframe(checked)
    return checked
