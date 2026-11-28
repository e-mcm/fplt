"""Readable forecast boards and plan/transfer counterfactual deltas.

Prefer explaining decisions via optimiser deltas (planned squad vs keep-and-
line-up the current fifteen) rather than model internals. Layered event
probs on the prediction ledger feed a first-order xPts decomposition; opaque
models still show ``xpts`` with null component columns. Fixture / involvement
depth (H/A, E[goals], shares, xMins from the minutes PMF) is a second layer
for layered models via public component accessors — still no SHAP or duals.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from fplt.backtest.ledger import read_predictions
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.distributions import MinutesDist
from fplt.models.spec import artifact_dir, get_model, load_model_specs
from fplt.models.universe import player_fixtures
from fplt.optimise.decisions import candidates_as_of
from fplt.optimise.plan import (
    DEFAULT_PLAN_HORIZON,
    GameweekPlan,
    covered_horizon,
    ensure_plan_chip_column,
    next_free_transfers,
    plan_parquet,
)
from fplt.optimise.squad import select_squad
from fplt.optimise.state import SquadState, load_squad_state
from fplt.schemas.explain import (
    FORECAST_BOARD_COLUMNS,
    PLAYER_DEPTH_COLUMNS,
    TRANSFER_DELTA_COLUMNS,
    ForecastBoardSchema,
    PlayerDepthSchema,
    TransferDeltaSchema,
)
from fplt.schemas.plan import PlanSchema
from fplt.scoring.rules import Rules
from fplt.types import GWKey
from fplt.utils.frames import require_dataframe, require_not_none

logger = logging.getLogger(__name__)

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
    ``planned_xi_xpts - baseline_xi_xpts - hit``. When the keep-squad
    baseline cannot be formed (budget/club), baseline and net are null and
    a constraint note records why — transfers and planned XI still return.
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


@dataclass(frozen=True)
class LayeredExplainParts:
    """Public layered components used to build fixture / involvement depth."""

    score: Any
    minutes: Any
    involvement: Any


@dataclass(frozen=True)
class PlayerDepthExplain:
    """Per-fixture depth rows plus the matching forecast-board slice."""

    depth: pl.DataFrame
    board: pl.DataFrame


@dataclass(frozen=True)
class TransferPairDepth:
    """Side-by-side depth for one A→B transfer pair."""

    player_out: int
    player_in: int
    out_depth: pl.DataFrame
    in_depth: pl.DataFrame
    board: pl.DataFrame


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
    checked = require_dataframe(checked)
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
    checked = require_dataframe(checked)
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
    baseline_xi: float | None
    net: float | None
    try:
        baseline = select_squad(
            owned_pool.select("player_key", "position", "team_key", "price", "xpts")
        )
        baseline_xi = float(baseline.total_xpts)
        net = planned_xi_xpts - baseline_xi - hit
    except ValueError as exc:
        # Keep-squad select_squad uses list prices and a £100m budget; live
        # squads with bank≈0 can fail even when the plan itself is valid.
        baseline_xi = None
        net = None
        constraints = (*constraints, f"baseline unavailable: {exc}")
    return DecisionExplain(
        transfers=transfers,
        baseline_xi_xpts=baseline_xi,
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
    view: PITView | None = None,
    parts: LayeredExplainParts | None = None,
) -> str:
    """Build decision + forecast (+ optional depth) markdown for a plan.

    Decision baseline soft-fails when keep-squad ``select_squad`` cannot form
    a legal fifteen; the forecast board and player-depth “why” blocks still
    emit. Depth needs ``view`` (fixture labels always; λ / shares / xMins
    when ``parts`` is set).

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
        view: PIT view for fixture / involvement depth.
        parts: Layered components when available.

    Returns:
        Markdown with decision explain, optional why-depth, then forecast board.
    """
    sections: list[str] = []
    decision: DecisionExplain | None
    try:
        decision = build_decision_explain(plan, state, candidates)
        sections.append(format_decision_explain(decision, names))
    except ValueError as exc:
        decision = None
        sections.append(f"### Decision explain\n\n_Unavailable: {exc}_\n")

    kept = plan.picks.filter((pl.col("step") == 0) & ~pl.col("sold"))
    kept_keys = [int(key) for key in kept["player_key"].to_list()]
    xi_keys = [int(key) for key in kept.filter(pl.col("in_xi"))["player_key"].to_list()]
    if view is not None:
        why = _plan_why_markdown(
            view=view,
            season=season,
            gameweek=gameweek,
            predictions=predictions,
            rules=rules,
            names=names,
            parts=parts,
            decision=decision,
            xi_keys=xi_keys,
        )
        if why:
            sections.append(why)

    board = build_forecast_board(
        predictions,
        season=season,
        rules=rules,
        target_gw=gameweek,
        player_keys=kept_keys,
    )
    sections.append(format_forecast_board(board, names))
    return "\n".join(part.rstrip() for part in sections if part) + "\n"


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
    player_key: int | None = None,
    vs_player_key: int | None = None,
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
        player_key: When set, render per-player fixture depth (and optional
            ``vs_player_key`` transfer-pair depth) as the primary section.
        vs_player_key: Optional sell-side key for A→B depth with ``player_key``
            as the buy-side (``--player`` in / ``--vs`` out).

    Returns:
        Markdown text.

    Raises:
        FileNotFoundError: Plan parquet missing when decision explain is on.
        KeyError: Unknown model name.
        ValueError: Predictions missing for the board, or ``--vs`` without
            ``--player``.
    """
    if vs_player_key is not None and player_key is None:
        raise ValueError("--vs requires --player")
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

    if player_key is not None:
        if preds.is_empty():
            raise ValueError(
                f"{model_name} has no predictions for {season} GW{gameweek}; "
                "run fplt predict first"
            )
        stored_depth = _depth_from_stored_components(
            data_dir,
            model_name=model_name,
            season=season,
            gameweek=gameweek,
            player_keys=(
                [vs_player_key, player_key]
                if vs_player_key is not None
                else [player_key]
            ),
        )
        parts = None
        if stored_depth is None:
            parts = load_layered_explain_parts(
                data_dir, model_name, season, gameweek, fit_if_missing=True
            )
        if vs_player_key is not None:
            pair = build_transfer_pair_depth(
                view,
                player_out=vs_player_key,
                player_in=player_key,
                target_gw=gameweek,
                predictions=preds,
                rules=rules,
                parts=parts,
                season=season,
                stored_depth=stored_depth,
            )
            sections.append(format_transfer_pair_depth(pair, names))
        else:
            depth = build_player_depth(
                view,
                player_key=player_key,
                target_gw=gameweek,
                predictions=preds,
                rules=rules,
                parts=parts,
                season=season,
                stored_depth=stored_depth,
            )
            sections.append(format_player_depth(depth, names))
        return "\n".join(part.rstrip() for part in sections if part).rstrip() + "\n"

    plan_frame: pl.DataFrame | None = None
    solved: GameweekPlan | None = None
    candidates: pl.DataFrame | None = None
    if include_decision:
        covered = covered_horizon(gameweek, horizon)
        path = plan_parquet(data_dir, spec.model_id, season, gameweek, covered)
        if not path.exists():
            raise FileNotFoundError(f"plan not found: {path}")
        raw = PlanSchema.validate(ensure_plan_chip_column(pl.read_parquet(path)))
        raw = require_dataframe(raw)
        candidates = candidates_as_of(view)
        plan_frame = raw
        solved = _gameweek_plan_from_table(raw, candidates, state)
        if include_forecast:
            parts = load_layered_explain_parts(
                data_dir, model_name, season, gameweek, fit_if_missing=False
            )
            sections.append(
                explain_plan_markdown(
                    solved,
                    state,
                    candidates,
                    preds,
                    season=season,
                    gameweek=gameweek,
                    rules=rules,
                    names=names,
                    view=view,
                    parts=parts,
                )
            )
            return "\n".join(part.rstrip() for part in sections if part).rstrip() + "\n"
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


def layered_explain_parts(model: object) -> LayeredExplainParts | None:
    """Return layered score/minutes/involvement parts when the model exposes them.

    Uses public ``score_model`` / ``minutes_model`` / ``involvement_model``
    accessors (or attributes of those names). Opaque predictors yield ``None``.

    Args:
        model: A fitted or unfitted predictor instance.

    Returns:
        ``LayeredExplainParts`` or ``None`` when any component is missing.
    """
    score = getattr(model, "score_model", None)
    minutes = getattr(model, "minutes_model", None)
    involvement = getattr(model, "involvement_model", None)
    if score is None or minutes is None or involvement is None:
        return None
    return LayeredExplainParts(score=score, minutes=minutes, involvement=involvement)


def load_layered_explain_parts(
    data_dir: Path,
    model_name: str,
    season: str,
    gameweek: int,
    *,
    fit_if_missing: bool = False,
) -> LayeredExplainParts | None:
    """Load (or optionally fit) layered components for explain depth.

    Args:
        data_dir: Data root with ``artifacts/``.
        model_name: YAML spec name.
        season: Season label.
        gameweek: As-of gameweek.
        fit_if_missing: When true and no artifact exists, fit and save like
            ``predict_live`` (CLI ``--player`` path). Plan-report wiring keeps
            this false so a missing artifact only skips depth.

    Returns:
        Parts for layered models, else ``None``.
    """
    # Import registers predictors / components with get_model.
    import fplt.models.composite  # noqa: F401  # pyright: ignore[reportUnusedImport]

    specs = load_model_specs()
    if model_name not in specs:
        return None
    spec = specs[model_name]
    as_of = GWKey(season, gameweek)
    dest = artifact_dir(data_dir, spec, as_of)
    try:
        cls = get_model(spec.kind)
    except KeyError:
        return None
    model: Any
    if (dest / "spec.json").exists():
        try:
            model = cls.load(dest, spec)
        except Exception as exc:  # noqa: BLE001 — depth is best-effort
            logger.warning("layered explain load failed: %s", exc)
            return None
    elif fit_if_missing:
        view = PITData(DataStore(data_dir)).view(as_of)
        model = cls(spec)
        try:
            model.fit(view)
            model.save(dest)
        except Exception as exc:  # noqa: BLE001
            logger.warning("layered explain fit failed: %s", exc)
            return None
    else:
        return None
    return layered_explain_parts(model)


def xmins_from_minutes_pmf(dist: MinutesDist) -> pl.DataFrame:
    """Derive expected minutes from a minutes PMF at explain/report time.

    Args:
        dist: Binned ``MinutesDist`` from a minutes component.

    Returns:
        ``player_key``, ``fixture_id``, ``xmins`` (matches
        ``MinutesDist.expected_minutes``).
    """
    return dist.expected_minutes()


def minutes_summary_from_pmf(dist: MinutesDist) -> pl.DataFrame:
    """Derive ``xmins``, ``p_played``, and ``p_60`` from a minutes PMF.

    Args:
        dist: Binned ``MinutesDist`` from a minutes component.

    Returns:
        ``player_key``, ``fixture_id``, ``xmins``, ``p_played``, ``p_60``.
    """
    bins = np.asarray(dist.bins, dtype=np.int64)
    return xmins_from_minutes_pmf(dist).with_columns(
        pl.Series("p_played", dist.pmf[:, bins > 0].sum(axis=1)),
        pl.Series("p_60", dist.pmf[:, bins >= 60].sum(axis=1)),
    )


def build_player_depth(
    view: PITView,
    *,
    player_key: int,
    target_gw: int,
    predictions: pl.DataFrame,
    rules: Rules,
    parts: LayeredExplainParts | None = None,
    season: str | None = None,
    stored_depth: pl.DataFrame | None = None,
) -> PlayerDepthExplain:
    """Build fixture / involvement depth for one player in one gameweek.

    Fixture H/A and opponent come from PIT. Team/opp E[goals], involvement
    weights, and xMins need ``parts`` or ``stored_depth``; otherwise those
    columns are null. Opaque models still get fixture labels + the board.

    Args:
        view: Point-in-time tables at the deadline.
        player_key: Player to explain.
        target_gw: Target gameweek.
        predictions: Ledger / prediction rows (event probs when present).
        rules: Scoring rules for the board split.
        parts: Optional layered components.
        season: Season label; defaults to ``view.as_of.season``.
        stored_depth: Optional prebuilt ``PlayerDepthSchema`` rows from a
            predict-run component table (preferred when present).

    Returns:
        Depth rows (``PlayerDepthSchema``) and a forecast-board slice.
    """
    season_label = season if season is not None else view.as_of.season
    board = build_forecast_board(
        predictions,
        season=season_label,
        rules=rules,
        target_gw=target_gw,
        player_keys=[player_key],
    )
    if stored_depth is not None:
        depth = _checked_player_depth(
            stored_depth.filter(pl.col("player_key") == player_key)
        )
    else:
        depth = _player_depth_rows(
            view,
            player_keys=[player_key],
            target_gw=target_gw,
            parts=parts,
            season=season_label,
        )
    return PlayerDepthExplain(depth=depth, board=board)


def build_transfer_pair_depth(
    view: PITView,
    *,
    player_out: int,
    player_in: int,
    target_gw: int,
    predictions: pl.DataFrame,
    rules: Rules,
    parts: LayeredExplainParts | None = None,
    season: str | None = None,
    stored_depth: pl.DataFrame | None = None,
) -> TransferPairDepth:
    """Build side-by-side depth for a transfer pair (out → in).

    Args:
        view: Point-in-time tables at the deadline.
        player_out: Sold player key.
        player_in: Bought player key.
        target_gw: Act-on gameweek.
        predictions: Ledger / prediction rows.
        rules: Scoring rules for the board split.
        parts: Optional layered components.
        season: Season label; defaults to ``view.as_of.season``.
        stored_depth: Optional prebuilt depth rows from run component tables.

    Returns:
        Depth for each side plus a two-player forecast board.
    """
    season_label = season if season is not None else view.as_of.season
    board = build_forecast_board(
        predictions,
        season=season_label,
        rules=rules,
        target_gw=target_gw,
        player_keys=[player_out, player_in],
    )
    if stored_depth is not None:
        out_depth = _checked_player_depth(
            stored_depth.filter(pl.col("player_key") == player_out)
        )
        in_depth = _checked_player_depth(
            stored_depth.filter(pl.col("player_key") == player_in)
        )
    else:
        out_depth = _player_depth_rows(
            view,
            player_keys=[player_out],
            target_gw=target_gw,
            parts=parts,
            season=season_label,
        )
        in_depth = _player_depth_rows(
            view,
            player_keys=[player_in],
            target_gw=target_gw,
            parts=parts,
            season=season_label,
        )
    return TransferPairDepth(
        player_out=player_out,
        player_in=player_in,
        out_depth=out_depth,
        in_depth=in_depth,
        board=board,
    )


def _depth_from_stored_components(
    data_dir: Path,
    *,
    model_name: str,
    season: str,
    gameweek: int,
    player_keys: Sequence[int],
) -> pl.DataFrame | None:
    """Load depth rows from a predict run's component tables when present."""
    from fplt.optimise.component_tables import (
        depth_rows_from_component_tables,
        load_run_component_tables,
    )

    tables = load_run_component_tables(
        data_dir, model_name=model_name, season=season, gameweek=gameweek
    )
    if tables is None:
        return None
    return depth_rows_from_component_tables(
        minutes=tables.minutes,
        scoreline=tables.scoreline,
        involvement=tables.involvement,
        player_keys=list(player_keys),
        target_gw=gameweek,
    )


def format_player_depth(
    explain: PlayerDepthExplain,
    names: Mapping[int, str] | None = None,
) -> str:
    """Render player fixture depth and the matching forecast board.

    Args:
        explain: Output of ``build_player_depth``.
        names: Optional ``player_key`` → display name map.

    Returns:
        Markdown text.
    """
    lines = [
        "### Player depth",
        "",
        (
            "Fixture context from PIT; E[goals] / involvement / xMins from "
            "layered components when available (null for opaque models). "
            "xMins = Σ bin·P(bin) from the minutes PMF."
        ),
        "",
        _format_depth_table(explain.depth, names),
        format_forecast_board(explain.board, names),
    ]
    return "\n".join(part.rstrip() for part in lines if part) + "\n"


def format_transfer_pair_depth(
    explain: TransferPairDepth,
    names: Mapping[int, str] | None = None,
) -> str:
    """Render side-by-side depth for a transfer pair.

    Args:
        explain: Output of ``build_transfer_pair_depth``.
        names: Optional display names.

    Returns:
        Markdown text.
    """
    out_label = _label(explain.player_out, names)
    in_label = _label(explain.player_in, names)
    lines = [
        f"### Transfer depth · {out_label} → {in_label}",
        "",
        f"#### Out · {out_label}",
        "",
        _format_depth_table(explain.out_depth, names),
        f"#### In · {in_label}",
        "",
        _format_depth_table(explain.in_depth, names),
        format_forecast_board(explain.board, names),
    ]
    return "\n".join(part.rstrip() for part in lines if part) + "\n"


def _plan_why_markdown(
    *,
    view: PITView,
    season: str,
    gameweek: int,
    predictions: pl.DataFrame,
    rules: Rules,
    names: Mapping[int, str] | None,
    parts: LayeredExplainParts | None,
    decision: DecisionExplain | None,
    xi_keys: Sequence[int],
) -> str:
    """Week-1 transfer-pair + XI depth for plan reports."""
    blocks: list[str] = [
        "### Why (fixture / involvement depth)",
        "",
        (
            "Act-on week transfers and XI. Fixture labels always; λ / shares / "
            "xMins when layered components are loaded."
        ),
        "",
    ]
    if decision is not None and not decision.transfers.is_empty():
        for row in decision.transfers.iter_rows(named=True):
            pair = build_transfer_pair_depth(
                view,
                player_out=int(row["player_key_out"]),
                player_in=int(row["player_key_in"]),
                target_gw=gameweek,
                predictions=predictions,
                rules=rules,
                parts=parts,
                season=season,
            )
            blocks.append(format_transfer_pair_depth(pair, names).rstrip())
            blocks.append("")
    if xi_keys:
        depth = _player_depth_rows(
            view,
            player_keys=list(xi_keys),
            target_gw=gameweek,
            parts=parts,
            season=season,
        )
        blocks.append("#### Starting XI depth")
        blocks.append("")
        blocks.append(_format_depth_table(depth, names))
    text = "\n".join(blocks).rstrip()
    return text + "\n" if text.strip() else ""


def _player_depth_rows(
    view: PITView,
    *,
    player_keys: Sequence[int],
    target_gw: int,
    parts: LayeredExplainParts | None,
    season: str,
) -> pl.DataFrame:
    """Build ``PlayerDepthSchema`` rows for the given players / gameweek."""
    keys = [int(key) for key in player_keys]
    empty = _empty_player_depth()
    if not keys:
        return empty
    asked = player_fixtures(view, [GWKey(season, target_gw)]).filter(
        pl.col("player_key").is_in(keys)
    )
    if asked.is_empty():
        # Blank week / unknown player: one null-fixture stub per key.
        stubs = [
            {
                "player_key": key,
                "target_gw": target_gw,
                "fixture_id": None,
                "is_home": None,
                "team_key": None,
                "opponent_team_key": None,
                "team_eg": None,
                "opp_eg": None,
                "goal_weight": None,
                "assist_weight": None,
                "xmins": None,
            }
            for key in keys
        ]
        return _checked_player_depth(pl.DataFrame(stubs, schema=PLAYER_DEPTH_COLUMNS))

    players = (
        view.latest("players", by=["season", "player_key"])
        .select("season", "player_key", "team_key")
        .collect()
    )
    fixtures = (
        view.latest("fixtures", by=["season", "fixture_id"])
        .select("season", "fixture_id", "team_h_key", "team_a_key")
        .collect()
    )
    base = (
        asked.join(players, on=["season", "player_key"], how="inner")
        .join(fixtures, on=["season", "fixture_id"], how="inner")
        .with_columns(
            (pl.col("team_key") == pl.col("team_h_key")).alias("is_home"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("team_a_key"))
            .otherwise(pl.col("team_h_key"))
            .alias("opponent_team_key"),
            pl.lit(target_gw).cast(pl.Int64).alias("target_gw"),
            pl.lit(None).cast(pl.Float64).alias("team_eg"),
            pl.lit(None).cast(pl.Float64).alias("opp_eg"),
            pl.lit(None).cast(pl.Float64).alias("goal_weight"),
            pl.lit(None).cast(pl.Float64).alias("assist_weight"),
            pl.lit(None).cast(pl.Float64).alias("xmins"),
        )
        .select(list(PLAYER_DEPTH_COLUMNS))
    )

    if parts is None:
        return _checked_player_depth(base)

    fixture_rows = (
        fixtures.join(
            asked.select("season", "fixture_id").unique(),
            on=["season", "fixture_id"],
            how="inner",
        )
        .select("season", "fixture_id", "team_h_key", "team_a_key")
        .unique(subset=["fixture_id"])
    )
    try:
        score_dist = parts.score.predict(view, fixture_rows)
        eg = score_dist.expected_goals()
        minutes_dist = parts.minutes.predict(
            view, asked.select("player_key", "fixture_id")
        )
        xmins = xmins_from_minutes_pmf(minutes_dist)
        weights = parts.involvement.predict(
            view, asked.select("player_key", "fixture_id")
        )
    except Exception as exc:  # noqa: BLE001 — depth is best-effort
        logger.warning("player depth components failed: %s", exc)
        return _checked_player_depth(base)

    enriched = (
        base.drop("team_eg", "opp_eg", "goal_weight", "assist_weight", "xmins")
        .join(eg, on="fixture_id", how="left")
        .join(weights, on=["player_key", "fixture_id"], how="left")
        .join(xmins, on=["player_key", "fixture_id"], how="left")
        .with_columns(
            pl.when(pl.col("is_home"))
            .then(pl.col("home_eg"))
            .otherwise(pl.col("away_eg"))
            .alias("team_eg"),
            pl.when(pl.col("is_home"))
            .then(pl.col("away_eg"))
            .otherwise(pl.col("home_eg"))
            .alias("opp_eg"),
        )
        .select(list(PLAYER_DEPTH_COLUMNS))
    )
    return _checked_player_depth(enriched)


def _format_depth_table(
    depth: pl.DataFrame,
    names: Mapping[int, str] | None,
) -> str:
    checked = PlayerDepthSchema.validate(depth)
    checked = require_dataframe(checked)
    if checked.is_empty():
        return "_No fixture rows_\n"
    lines = [
        "| Name | GW | Fix | H/A | Team | Opp | E[G] | Opp E[G] | "
        "goal_w | assist_w | xMins |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(
        checked.iter_rows(named=True),
        key=lambda row: (
            int(row["player_key"]),
            int(row["fixture_id"]) if row["fixture_id"] is not None else -1,
        ),
    )
    for row in ordered:
        ha = "—" if row["is_home"] is None else ("H" if bool(row["is_home"]) else "A")
        lines.append(
            "| "
            + " | ".join(
                [
                    _label(int(row["player_key"]), names),
                    str(int(row["target_gw"])),
                    "—" if row["fixture_id"] is None else str(int(row["fixture_id"])),
                    ha,
                    "—" if row["team_key"] is None else str(int(row["team_key"])),
                    (
                        "—"
                        if row["opponent_team_key"] is None
                        else str(int(row["opponent_team_key"]))
                    ),
                    _num(row["team_eg"]),
                    _num(row["opp_eg"]),
                    _num(row["goal_weight"]),
                    _num(row["assist_weight"]),
                    _num(row["xmins"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _empty_player_depth() -> pl.DataFrame:
    empty = pl.DataFrame(schema=PLAYER_DEPTH_COLUMNS)
    return _checked_player_depth(empty)


def _checked_player_depth(frame: pl.DataFrame) -> pl.DataFrame:
    checked = PlayerDepthSchema.validate(frame.select(list(PLAYER_DEPTH_COLUMNS)))
    checked = require_dataframe(checked)
    return checked


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
            p_played = require_not_none(p_played, what="p_played")
            p_60 = require_not_none(p_60, what="p_60")
            p_goal = require_not_none(p_goal, what="p_goal")
            p_assist = require_not_none(p_assist, what="p_assist")
            p_cs = require_not_none(p_cs, what="p_cs")
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
    checked = require_dataframe(checked)
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
    checked = PlanSchema.validate(ensure_plan_chip_column(plan))
    checked = require_dataframe(checked)
    steps = [int(step) for step in checked["step"].unique().sort().to_list()]
    hits: list[float] = []
    chips: list[str | None] = []
    for step in steps:
        part = checked.filter(pl.col("step") == step)
        hits.append(float(part["hit"][0]) if part.height else 0.0)
        if part.height and part["chip"][0] is not None:
            chips.append(str(part["chip"][0]))
        else:
            chips.append(None)
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
        chips=tuple(chips),
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
    checked = require_dataframe(checked)
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
    "LayeredExplainParts",
    "PlayerDepthExplain",
    "TransferPairDepth",
    "append_explain_sections",
    "build_decision_explain",
    "build_forecast_board",
    "build_player_depth",
    "build_transfer_pair_depth",
    "explain_plan_markdown",
    "format_decision_explain",
    "format_forecast_board",
    "format_player_depth",
    "format_transfer_pair_depth",
    "layered_explain_parts",
    "load_layered_explain_parts",
    "minutes_summary_from_pmf",
    "run_explain",
    "xmins_from_minutes_pmf",
]
