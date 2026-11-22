"""Readable forecast boards and plan/transfer counterfactual deltas.

Prefer explaining decisions via optimiser deltas (planned squad vs keep-and-
line-up the current fifteen) rather than model internals. Layered event
probs on the prediction ledger feed a first-order xPts decomposition; opaque
models still show ``xpts`` with null component columns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from fplt.backtest.ledger import read_predictions
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.decisions import candidates_as_of
from fplt.optimise.plan import (
    DEFAULT_PLAN_HORIZON,
    GameweekPlan,
    covered_horizon,
    next_free_transfers,
    plan_parquet,
)
from fplt.optimise.squad import select_squad
from fplt.optimise.state import SquadState, load_squad_state
from fplt.schemas.explain import (
    FORECAST_BOARD_COLUMNS,
    TRANSFER_DELTA_COLUMNS,
    ForecastBoardSchema,
    TransferDeltaSchema,
)
from fplt.schemas.plan import PlanSchema
from fplt.scoring.rules import Rules
from fplt.types import GWKey

_POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
_PROB_COLS = ("p_played", "p_60", "p_goal", "p_assist", "p_cs", "p_haul")
_COMPONENT_COLS = (
    "appearance_xpts",
    "goals_xpts",
    "assists_xpts",
    "cs_xpts",
    "residual_xpts",
)


@dataclass(frozen=True)
class DecisionExplain:
    """Counterfactual deltas for the act-on gameweek of a plan.

    ``baseline_xi_xpts`` is the captain-doubled XI score from keeping the
    current fifteen (no transfers). ``planned_xi_xpts`` is the same scoring
    on the planned step-0 XI. ``net_delta`` is
    ``planned_xi_xpts - baseline_xi_xpts - hit``.
    """

    transfers: pl.DataFrame
    baseline_xi_xpts: float | None
    planned_xi_xpts: float
    hit: float
    free_transfers_before: int
    free_transfers_after: int
    transfers_count: int
    net_delta: float | None
    constraints: tuple[str, ...]
    is_draft: bool


def build_forecast_board(
    predictions: pl.DataFrame,
    *,
    season: str,
    rules: Rules,
    target_gw: int | None = None,
    player_keys: Sequence[int] | None = None,
    top_n: int | None = None,
) -> pl.DataFrame:
    """Build a per-player forecast board from prediction / ledger rows.

    Args:
        predictions: Rows with at least ``player_key``, ``position``,
            ``target_gw``, and ``xpts``. Event-prob columns are optional.
            Double-gameweek fixtures are summed after per-fixture
            decomposition.
        season: Season label (reserved for callers; scoring uses ``rules``).
        rules: Season scoring table used for the first-order split.
        target_gw: Keep a single target gameweek when set.
        player_keys: Optional player filter (e.g. planned squad).
        top_n: Keep the top ``n`` players by ``xpts`` after aggregation.

    Returns:
        A frame matching ``ForecastBoardSchema``, sorted by ``xpts`` desc.

    Raises:
        ValueError: Required columns are missing or ``top_n`` is invalid.
    """
    del season
    required = {"player_key", "position", "target_gw", "xpts"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing {sorted(missing)}")
    frame = predictions
    if target_gw is not None:
        frame = frame.filter(pl.col("target_gw") == target_gw)
    if player_keys is not None:
        keys = [int(key) for key in player_keys]
        frame = frame.filter(pl.col("player_key").is_in(keys))
    if frame.is_empty():
        return _empty_forecast_board()

    extras: list[pl.Expr] = []
    if "xpts_sd" not in frame.columns:
        extras.append(pl.lit(None).cast(pl.Float64).alias("xpts_sd"))
    for name in _PROB_COLS:
        if name not in frame.columns:
            extras.append(pl.lit(None).cast(pl.Float64).alias(name))
    if extras:
        frame = frame.with_columns(extras)

    if "fixture_id" in frame.columns:
        frame = frame.unique(
            subset=["player_key", "target_gw", "fixture_id"],
            keep="last",
        )

    decomposed = _decompose_rows(frame, rules)
    board = (
        decomposed.group_by("player_key", "position", "target_gw")
        .agg(
            pl.col("xpts").sum(),
            pl.col("xpts_sd").fill_null(0.0).pow(2).sum().sqrt().alias("xpts_sd"),
            *[pl.col(name).mean() for name in _PROB_COLS],
            *[
                pl.when(pl.col(name).is_null().all())
                .then(None)
                .otherwise(pl.col(name).sum())
                .alias(name)
                for name in _COMPONENT_COLS
            ],
        )
        .sort("xpts", descending=True)
    )
    if top_n is not None:
        if top_n < 1:
            raise ValueError(f"top_n must be positive, got {top_n}")
        board = board.head(top_n)
    checked = ForecastBoardSchema.validate(board.select(list(FORECAST_BOARD_COLUMNS)))
    assert isinstance(checked, pl.DataFrame)
    return checked


def format_forecast_board(
    board: pl.DataFrame,
    names: Mapping[int, str] | None = None,
) -> str:
    """Render a forecast board as a markdown table.

    Args:
        board: Rows matching ``ForecastBoardSchema``.
        names: Optional ``player_key`` → display name map.

    Returns:
        Markdown text including a short note on the residual column.
    """
    checked = ForecastBoardSchema.validate(board)
    assert isinstance(checked, pl.DataFrame)
    lines = [
        "### Forecast board",
        "",
        (
            "First-order split from event probs × scoring rules. "
            "Residual holds GC / other / multi-event mass so columns sum to xPts."
        ),
        "",
    ]
    if checked.is_empty():
        lines.append("_No prediction rows_")
        lines.append("")
        return "\n".join(lines)
    lines.extend(
        [
            "| Name | Pos | GW | xPts | App≈ | G≈ | A≈ | CS≈ | Resid | "
            "P(play) | P(60) | P(G) | P(A) | P(CS) | P(haul) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    ordered = sorted(
        checked.iter_rows(named=True),
        key=lambda row: (
            -float(row["xpts"]),
            _POSITION_ORDER.get(str(row["position"]), 99),
            int(row["player_key"]),
        ),
    )
    for row in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    _label(int(row["player_key"]), names),
                    str(row["position"]),
                    str(int(row["target_gw"])),
                    _num(row["xpts"]),
                    _num(row["appearance_xpts"]),
                    _num(row["goals_xpts"]),
                    _num(row["assists_xpts"]),
                    _num(row["cs_xpts"]),
                    _num(row["residual_xpts"]),
                    _num(row["p_played"]),
                    _num(row["p_60"]),
                    _num(row["p_goal"]),
                    _num(row["p_assist"]),
                    _num(row["p_cs"]),
                    _num(row["p_haul"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def build_decision_explain(
    plan: GameweekPlan,
    state: SquadState,
    candidates: pl.DataFrame,
    *,
    step: int = 0,
) -> DecisionExplain:
    """Compute transfer and XI deltas vs keeping the current squad.

    Args:
        plan: Solved horizon (``GameweekPlan`` from ``plan_squad``).
        state: Pre-plan squad (owned selling prices, free transfers, hits).
        candidates: ``CandidateSchema`` rows used when the plan was built.
        step: Horizon step to explain (default: act-on week).

    Returns:
        Transfer pairs, XI totals, hit, and constraint notes.

    Raises:
        ValueError: ``step`` is missing from the plan, or owned players are
            absent from ``candidates``.
    """
    part = plan.picks.filter(pl.col("step") == step)
    if part.is_empty():
        raise ValueError(f"plan has no rows for step {step}")
    hit = float(plan.hits[step]) if step < len(plan.hits) else 0.0
    if step < len(plan.free_transfers):
        free_before = int(plan.free_transfers[step])
    else:
        free_before = int(state.free_transfers)
    is_draft = not state.owned
    planned_xi_xpts = _xi_total(part.filter(~pl.col("sold")))
    transfers = _transfer_deltas(part, is_draft=is_draft)
    free_after = next_free_transfers(
        free_before, transfers.height, int(state.ft_bank_cap)
    )
    constraints = _constraint_notes(
        state=state,
        hit=hit,
        free_before=free_before,
        free_after=free_after,
        transfers_count=transfers.height,
        is_draft=is_draft,
        plan_points=float(plan.points),
    )

    if is_draft:
        return DecisionExplain(
            transfers=transfers,
            baseline_xi_xpts=None,
            planned_xi_xpts=planned_xi_xpts,
            hit=hit,
            free_transfers_before=free_before,
            free_transfers_after=free_after,
            transfers_count=0,
            net_delta=None,
            constraints=constraints,
            is_draft=True,
        )

    owned_keys = [player.player_key for player in state.owned]
    candidate_keys = set(candidates["player_key"].to_list())
    missing = [key for key in owned_keys if key not in candidate_keys]
    if missing:
        raise ValueError(f"owned players are not in candidates: {missing}")
    xpts_by_key = {
        int(row["player_key"]): float(row["xpts"]) for row in part.iter_rows(named=True)
    }
    owned_pool = candidates.filter(pl.col("player_key").is_in(owned_keys)).with_columns(
        pl.col("player_key")
        .replace_strict(xpts_by_key, default=0.0)
        .cast(pl.Float64)
        .alias("xpts")
    )
    baseline = select_squad(
        owned_pool.select("player_key", "position", "team_key", "price", "xpts")
    )
    net = planned_xi_xpts - float(baseline.total_xpts) - hit
    return DecisionExplain(
        transfers=transfers,
        baseline_xi_xpts=float(baseline.total_xpts),
        planned_xi_xpts=planned_xi_xpts,
        hit=hit,
        free_transfers_before=free_before,
        free_transfers_after=free_after,
        transfers_count=transfers.height,
        net_delta=net,
        constraints=constraints,
        is_draft=False,
    )


def format_decision_explain(
    explain: DecisionExplain,
    names: Mapping[int, str] | None = None,
) -> str:
    """Render decision / transfer deltas as markdown.

    Args:
        explain: Output of ``build_decision_explain``.
        names: Optional ``player_key`` → display name map.

    Returns:
        Markdown text.
    """
    lines = [
        "### Decision explain",
        "",
        (
            "Counterfactual vs keep-and-line-up the current fifteen "
            "(captain doubled). No MILP duals / shadow prices."
        ),
        "",
    ]
    if explain.is_draft:
        lines.append(
            f"Initial draft — planned XI xPts (C×2): {_num(explain.planned_xi_xpts)}"
        )
        lines.append("")
    else:
        lines.extend(
            [
                f"Baseline XI xPts (keep squad, C×2): {_num(explain.baseline_xi_xpts)}",
                f"Planned XI xPts (C×2): {_num(explain.planned_xi_xpts)}",
                f"Hit: {_num(explain.hit)}",
                f"Net delta (planned − baseline − hit): {_num(explain.net_delta)}",
                f"Transfers: {explain.transfers_count}",
                (
                    f"Free transfers before={explain.free_transfers_before}, "
                    f"after={explain.free_transfers_after}"
                ),
                "",
            ]
        )
    lines.append("#### Transfers")
    lines.append("")
    if explain.transfers.is_empty():
        lines.append("_None_")
        lines.append("")
    else:
        lines.extend(
            [
                "| Out | Pos | xPts | In | Pos | xPts | ΔxPts | Δprice |",
                "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in explain.transfers.iter_rows(named=True):
            lines.append(
                "| "
                + " | ".join(
                    [
                        _label(int(row["player_key_out"]), names),
                        str(row["position_out"]),
                        _num(row["xpts_out"]),
                        _label(int(row["player_key_in"]), names),
                        str(row["position_in"]),
                        _num(row["xpts_in"]),
                        _num(row["delta_xpts"]),
                        str(int(row["delta_price"])),
                    ]
                )
                + " |"
            )
        lines.append("")
    if explain.constraints:
        lines.append("#### Constraints touched")
        lines.append("")
        for note in explain.constraints:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def append_explain_sections(
    report: str,
    *,
    forecast: str | None = None,
    decision: str | None = None,
) -> str:
    """Append forecast / decision markdown blocks to a plan report.

    Args:
        report: Existing ``render_report`` markdown.
        forecast: Optional ``format_forecast_board`` output.
        decision: Optional ``format_decision_explain`` output.

    Returns:
        Combined markdown ending in a newline.
    """
    parts = [report.rstrip()]
    if decision:
        parts.append(decision.rstrip())
    if forecast:
        parts.append(forecast.rstrip())
    return "\n\n".join(parts) + "\n"


def explain_plan_markdown(
    plan: GameweekPlan,
    state: SquadState,
    candidates: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    season: str,
    gameweek: int,
    rules: Rules,
    names: Mapping[int, str] | None = None,
) -> str:
    """Build decision + forecast markdown for a just-solved plan.

    Args:
        plan: Solved ``GameweekPlan``.
        state: Pre-plan squad state.
        candidates: Candidate pool used by the solver.
        predictions: Ledger / prediction rows for the act-on gameweek
            (event probs when available).
        season: Season label for scoring / board metadata.
        gameweek: Act-on gameweek.
        rules: Scoring rules for the forecast split.
        names: Optional display names.

    Returns:
        Markdown with decision explain then forecast board (planned squad).
    """
    decision = build_decision_explain(plan, state, candidates)
    decision_md = format_decision_explain(decision, names)
    kept_keys = plan.picks.filter((pl.col("step") == 0) & ~pl.col("sold"))[
        "player_key"
    ].to_list()
    board = build_forecast_board(
        predictions,
        season=season,
        rules=rules,
        target_gw=gameweek,
        player_keys=kept_keys,
    )
    return decision_md + "\n" + format_forecast_board(board, names)


def run_explain(
    data_dir: Path,
    model_name: str,
    season: str,
    gameweek: int,
    *,
    horizon: int = DEFAULT_PLAN_HORIZON,
    top_n: int = 30,
    squad: SquadState | None = None,
    include_forecast: bool = True,
    include_decision: bool = True,
) -> str:
    """Load ledger predictions and a saved plan; render explain markdown.

    Args:
        data_dir: Data root with ``ledger/`` and ``plans/``.
        model_name: YAML spec name.
        season: Season label.
        gameweek: Deadline / as-of gameweek.
        horizon: Horizon used when the plan was written.
        top_n: Forecast board size when not restricted to the planned XI.
        squad: Pre-plan squad for the counterfactual. Defaults to YAML
            ``load_squad_state`` when omitted.
        include_forecast: Build the forecast board section.
        include_decision: Build the decision-delta section (needs a plan).

    Returns:
        Markdown text.

    Raises:
        FileNotFoundError: Plan parquet missing when decision explain is on.
        KeyError: Unknown model name.
        ValueError: Predictions missing for the board.
    """
    spec = load_model_specs()[model_name]
    view = PITData(DataStore(data_dir)).view(GWKey(season, gameweek))
    names = _player_names(view)
    rules = Rules.for_season(season)
    ledger = read_predictions(data_dir)
    preds = ledger.filter(
        (pl.col("model_id") == spec.model_id)
        & (pl.col("as_of_season") == season)
        & (pl.col("as_of_gw") == gameweek)
        & (pl.col("target_season") == season)
        & (pl.col("target_gw") == gameweek)
    )
    state = squad if squad is not None else load_squad_state()
    sections: list[str] = [f"# Explain {model_name} · {season} GW{gameweek}", ""]

    plan_frame: pl.DataFrame | None = None
    if include_decision:
        covered = covered_horizon(gameweek, horizon)
        path = plan_parquet(data_dir, spec.model_id, season, gameweek, covered)
        if not path.exists():
            raise FileNotFoundError(f"plan not found: {path}")
        raw = PlanSchema.validate(pl.read_parquet(path))
        assert isinstance(raw, pl.DataFrame)
        candidates = candidates_as_of(view)
        plan_frame = raw
        solved = _gameweek_plan_from_table(raw, candidates, state)
        decision = build_decision_explain(solved, state, candidates)
        sections.append(format_decision_explain(decision, names))

    if include_forecast:
        player_keys: Sequence[int] | None = None
        board_top: int | None = top_n
        if plan_frame is not None:
            player_keys = plan_frame.filter((pl.col("step") == 0) & ~pl.col("sold"))[
                "player_key"
            ].to_list()
            board_top = None
        if preds.is_empty():
            raise ValueError(
                f"{model_name} has no predictions for {season} GW{gameweek}; "
                "run fplt predict first"
            )
        board = build_forecast_board(
            preds,
            season=season,
            rules=rules,
            target_gw=gameweek,
            player_keys=player_keys,
            top_n=board_top,
        )
        sections.append(format_forecast_board(board, names))

    return "\n".join(part.rstrip() for part in sections if part).rstrip() + "\n"


def _decompose_rows(frame: pl.DataFrame, rules: Rules) -> pl.DataFrame:
    """Attach first-order contribution columns at player-fixture grain."""
    goal_pts = {pos: float(rules.goals[pos]) for pos in ("GK", "DEF", "MID", "FWD")}
    cs_pts = {
        pos: float(rules.clean_sheets[pos]) for pos in ("GK", "DEF", "MID", "FWD")
    }
    short = float(rules.appearance.short)
    long = float(rules.appearance.long)
    assist = float(rules.assists)

    rows: list[dict[str, object]] = []
    for row in frame.iter_rows(named=True):
        position = str(row["position"])
        xpts = float(row["xpts"])
        p_played = _optional_float(row.get("p_played"))
        p_60 = _optional_float(row.get("p_60"))
        p_goal = _optional_float(row.get("p_goal"))
        p_assist = _optional_float(row.get("p_assist"))
        p_cs = _optional_float(row.get("p_cs"))
        if None in (p_played, p_60, p_goal, p_assist, p_cs):
            appearance = goals = assists = cs = residual = None
        else:
            assert p_played is not None and p_60 is not None
            assert p_goal is not None and p_assist is not None and p_cs is not None
            p_long = max(0.0, min(1.0, p_60))
            p_short = max(0.0, min(1.0, p_played) - p_long)
            appearance = p_short * short + p_long * long
            goals = p_goal * goal_pts[position]
            assists = p_assist * assist
            cs = p_cs * cs_pts[position]
            residual = xpts - (appearance + goals + assists + cs)
        rows.append(
            {
                "player_key": int(row["player_key"]),
                "position": position,
                "target_gw": int(row["target_gw"]),
                "xpts": xpts,
                "xpts_sd": _optional_float(row.get("xpts_sd")),
                "p_played": p_played,
                "p_60": p_60,
                "p_goal": p_goal,
                "p_assist": p_assist,
                "p_cs": p_cs,
                "p_haul": _optional_float(row.get("p_haul")),
                "appearance_xpts": appearance,
                "goals_xpts": goals,
                "assists_xpts": assists,
                "cs_xpts": cs,
                "residual_xpts": residual,
            }
        )
    return pl.DataFrame(rows, schema=FORECAST_BOARD_COLUMNS)


def _transfer_deltas(part: pl.DataFrame, *, is_draft: bool) -> pl.DataFrame:
    empty = pl.DataFrame(schema=TRANSFER_DELTA_COLUMNS)
    if is_draft:
        return empty
    bought = part.filter(pl.col("bought") & ~pl.col("sold")).sort("player_key")
    sold = part.filter(pl.col("sold")).sort("player_key")
    if bought.height != sold.height:
        n = min(bought.height, sold.height)
        bought = bought.head(n)
        sold = sold.head(n)
    rows: list[dict[str, object]] = []
    for buy, sell in zip(
        bought.iter_rows(named=True),
        sold.iter_rows(named=True),
        strict=True,
    ):
        price_in = int(buy["price"])
        price_out = int(sell["price"])
        x_in = float(buy["xpts"])
        x_out = float(sell["xpts"])
        rows.append(
            {
                "player_key_out": int(sell["player_key"]),
                "player_key_in": int(buy["player_key"]),
                "position_out": str(sell["position"]),
                "position_in": str(buy["position"]),
                "xpts_out": x_out,
                "xpts_in": x_in,
                "delta_xpts": x_in - x_out,
                "price_out": price_out,
                "price_in": price_in,
                "delta_price": price_in - price_out,
            }
        )
    if not rows:
        return empty
    checked = TransferDeltaSchema.validate(
        pl.DataFrame(rows, schema=TRANSFER_DELTA_COLUMNS)
    )
    assert isinstance(checked, pl.DataFrame)
    return checked


def _constraint_notes(
    *,
    state: SquadState,
    hit: float,
    free_before: int,
    free_after: int,
    transfers_count: int,
    is_draft: bool,
    plan_points: float,
) -> tuple[str, ...]:
    notes: list[str] = []
    if is_draft:
        notes.append("draft: fifteen bought with no transfer hits")
        notes.append(f"horizon objective (decayed): {plan_points:.2f}")
        return tuple(notes)
    paid = max(0, transfers_count - free_before)
    notes.append(
        f"free transfers before={free_before}, after step={free_after}, "
        f"moves={transfers_count}"
    )
    if hit > 0 or paid > 0:
        notes.append(
            f"hit cost applied={hit:.0f} "
            f"(hit_cost={state.hit_cost}, paid transfers≈{paid})"
        )
    else:
        notes.append("no hit: moves within free-transfer allowance")
    notes.append(f"FT bank cap={state.ft_bank_cap}")
    notes.append(f"horizon objective (decayed): {plan_points:.2f}")
    return tuple(notes)


def _xi_total(kept: pl.DataFrame) -> float:
    """Captain-doubled XI xPts from plan pick rows (vice ignored in the mean)."""
    xi = kept.filter(pl.col("in_xi"))
    if xi.is_empty():
        return 0.0
    base = float(xi["xpts"].sum())
    captain = xi.filter(pl.col("captain"))
    extra = float(captain["xpts"].sum()) if not captain.is_empty() else 0.0
    return base + extra


def _gameweek_plan_from_table(
    plan: pl.DataFrame,
    candidates: pl.DataFrame,
    state: SquadState,
) -> GameweekPlan:
    """Rebuild a ``GameweekPlan`` from a saved ``PlanSchema`` frame."""
    checked = PlanSchema.validate(plan)
    assert isinstance(checked, pl.DataFrame)
    steps = [int(step) for step in checked["step"].unique().sort().to_list()]
    hits: list[float] = []
    for step in steps:
        part = checked.filter(pl.col("step") == step)
        hits.append(float(part["hit"][0]) if part.height else 0.0)
    # PlanSchema does not store free-transfer banking; use pre-plan state for
    # step 0 and leave later steps as the same placeholder.
    frees = [int(state.free_transfers) for _ in steps]
    prices = {
        int(row["player_key"]): int(row["price"])
        for row in candidates.iter_rows(named=True)
    }
    # Selling prices for currently owned players override list price on outs.
    for player in state.owned:
        prices[player.player_key] = int(player.selling_price)
    picks = checked.select(
        "step",
        "player_key",
        "position",
        "team_key",
        "xpts",
        "in_xi",
        "captain",
        "vice",
        "bought",
        "sold",
    ).with_columns(
        pl.col("player_key")
        .replace_strict(prices, default=0)
        .cast(pl.Int64)
        .alias("price")
    )
    return GameweekPlan(
        picks=picks,
        hits=tuple(hits),
        free_transfers=tuple(frees),
        points=0.0,
    )


def _player_names(view: PITView) -> dict[int, str]:
    frame = (
        view.latest("players", ["season", "player_key"])
        .filter(pl.col("season") == view.as_of.season)
        .select("player_key", "web_name")
        .collect()
    )
    return {int(key): str(name) for key, name in frame.iter_rows()}


def _empty_forecast_board() -> pl.DataFrame:
    empty = pl.DataFrame(schema=FORECAST_BOARD_COLUMNS)
    checked = ForecastBoardSchema.validate(empty)
    assert isinstance(checked, pl.DataFrame)
    return checked


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _label(player_key: int, names: Mapping[int, str] | None) -> str:
    if names and player_key in names:
        return names[player_key]
    return str(player_key)


def _num(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f}"  # type: ignore[arg-type]


__all__ = [
    "DecisionExplain",
    "append_explain_sections",
    "build_decision_explain",
    "build_forecast_board",
    "explain_plan_markdown",
    "format_decision_explain",
    "format_forecast_board",
    "run_explain",
]
