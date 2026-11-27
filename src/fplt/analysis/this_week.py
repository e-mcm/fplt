"""This-week page shaping: selector, plan, players table, player drill-down.

Pure polars helpers for ``fplt ui`` (§3.3). No Streamlit / network / models —
callers load frames through ``Store`` and pass them in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from fplt.optimise.explain import build_forecast_board
from fplt.schemas.this_week import (
    THIS_WEEK_FIXTURE_DETAIL_COLUMNS,
    THIS_WEEK_TRANSFER_COLUMNS,
    ThisWeekFixtureDetailSchema,
    ThisWeekTransferSchema,
)
from fplt.scoring.rules import Rules
from fplt.utils.frames import require_dataframe

_PLACEHOLDER = "Unknown"
_POS_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


@dataclass(frozen=True)
class OfficialSelection:
    """One official pointer chosen for the This-week page."""

    season: str
    gw: int
    checkpoint: str
    line: str
    role: str
    predict_run_id: str
    plan_run_id: str


@dataclass(frozen=True)
class PlanPanel:
    """Step-0 plan summary for the This-week plan panel."""

    transfers: pl.DataFrame
    xi: pl.DataFrame
    bench: pl.DataFrame
    captain_key: int | None
    vice_key: int | None
    hit: float
    chip: str | None
    expected_points: float


@dataclass(frozen=True)
class PlayerDrilldown:
    """Per-player drill-down: fixtures, points split, optional PMF."""

    fixtures: pl.DataFrame
    points_split: pl.DataFrame
    points_pmf: pl.DataFrame
    has_components: bool


def default_official_selection(
    pointers: pl.DataFrame,
    *,
    applied_line: str | None = None,
    preferred_checkpoint: str = "T-90m",
) -> OfficialSelection | None:
    """Pick applied line → primary at ``preferred_checkpoint``, else any.

    Within a match set the highest ``gw`` wins. Returns ``None`` when empty.
    """
    if pointers.is_empty():
        return None
    for checkpoint_only in (True, False):
        pool = pointers
        if checkpoint_only:
            pool = pointers.filter(pl.col("checkpoint") == preferred_checkpoint)
            if pool.is_empty():
                continue
        chosen = _pick_line(pool, applied_line=applied_line)
        if chosen is not None:
            return chosen
    return _pick_line(pointers, applied_line=applied_line)


def plan_panel(
    plan: pl.DataFrame,
    *,
    players: pl.DataFrame,
    teams: pl.DataFrame,
    status: pl.DataFrame,
    chip: str | None = None,
) -> PlanPanel:
    """Build the step-0 plan panel from a ``PlanSchema`` frame."""
    labels = _player_labels(players, teams, status)
    empty = pl.DataFrame(schema=THIS_WEEK_TRANSFER_COLUMNS)
    empty_panel = PlanPanel(
        transfers=empty,
        xi=pl.DataFrame(),
        bench=pl.DataFrame(),
        captain_key=None,
        vice_key=None,
        hit=0.0,
        chip=chip,
        expected_points=0.0,
    )
    if plan.is_empty() or "step" not in plan.columns:
        return empty_panel
    step0 = plan.filter(pl.col("step") == 0)
    if step0.is_empty():
        return empty_panel
    hit_raw = step0.select(pl.col("hit").max()).item()
    hit = float(hit_raw) if isinstance(hit_raw, (int, float)) else 0.0
    captains = step0.filter(pl.col("captain"))["player_key"].to_list()
    vices = step0.filter(pl.col("vice"))["player_key"].to_list()
    transfers = _pair_transfers(
        step0.filter(pl.col("sold")).sort("player_key"),
        step0.filter(pl.col("bought")).sort("player_key"),
        labels,
    )
    kept = (
        step0.filter(~pl.col("sold"))
        .join(
            labels.select("player_key", "web_name", "team", "price"),
            on="player_key",
            how="left",
        )
        .with_columns(
            pl.col("web_name").fill_null(pl.format("#{}", pl.col("player_key"))),
            pl.col("position").replace_strict(_POS_ORDER, default=9).alias("_pos_rank"),
        )
        .sort(["_pos_rank", "player_key"])
    )
    xi = kept.filter(pl.col("in_xi")).select(
        "player_key", "web_name", "team", "position", "xpts", "captain", "vice"
    )
    bench = kept.filter(~pl.col("in_xi")).select(
        "player_key", "web_name", "team", "position", "xpts"
    )
    xi_xpts = float(xi["xpts"].sum()) if not xi.is_empty() else 0.0
    if captains and not xi.is_empty():
        cap = xi.filter(pl.col("player_key") == int(captains[0]))
        if not cap.is_empty():
            xi_xpts += float(cap["xpts"][0])
    return PlanPanel(
        transfers=transfers,
        xi=xi,
        bench=bench,
        captain_key=int(captains[0]) if captains else None,
        vice_key=int(vices[0]) if vices else None,
        hit=hit,
        chip=chip,
        expected_points=xi_xpts - hit,
    )


def players_table(
    predictions: pl.DataFrame,
    *,
    players: pl.DataFrame,
    teams: pl.DataFrame,
    status: pl.DataFrame,
    as_of_gw: int,
    horizon: int,
    squad_keys: Sequence[int] | None = None,
    positions: Sequence[str] | None = None,
    team_keys: Sequence[int] | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    min_p_played: float | None = None,
    my_squad_only: bool = False,
) -> pl.DataFrame:
    """Build the horizon players table (DGW summed; blanks → 0).

    Players present only in predictions keep a placeholder name. See
    ``ThisWeekPlayersSchema`` for base columns; ``xpts_gw{N}`` are dynamic.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    target_gws = [
        as_of_gw + offset for offset in range(horizon) if as_of_gw + offset <= 38
    ]
    labels = _player_labels(players, teams, status)
    squad = {int(k) for k in (squad_keys or ())}
    if predictions.is_empty():
        return _empty_players_table(target_gws)

    gw_pts = (
        predictions.group_by("player_key", "target_gw")
        .agg(pl.col("xpts").sum(), pl.col("p_played").mean().alias("p_played"))
        .filter(pl.col("target_gw").is_in(target_gws))
    )
    wide = (
        predictions.select("player_key")
        .unique()
        .join(labels, on="player_key", how="left")
        .with_columns(
            pl.col("web_name").fill_null(_PLACEHOLDER),
            pl.col("team").fill_null("?"),
            pl.col("position").fill_null("MID"),
            pl.col("player_key").is_in(list(squad)).alias("in_squad"),
        )
        .join(
            gw_pts.filter(pl.col("target_gw") == as_of_gw).select(
                "player_key", "p_played"
            ),
            on="player_key",
            how="left",
        )
    )
    for gw in target_gws:
        col = f"xpts_gw{gw}"
        part = gw_pts.filter(pl.col("target_gw") == gw).select(
            "player_key", pl.col("xpts").alias(col)
        )
        wide = wide.join(part, on="player_key", how="left").with_columns(
            pl.col(col).fill_null(0.0)
        )
    xpts_cols = [f"xpts_gw{gw}" for gw in target_gws]
    wide = wide.with_columns(
        pl.sum_horizontal([pl.col(c) for c in xpts_cols]).alias("total_xpts")
        if xpts_cols
        else pl.lit(0.0).alias("total_xpts")
    )
    out = wide
    if my_squad_only:
        out = out.filter(pl.col("in_squad"))
    if positions:
        out = out.filter(pl.col("position").is_in(list(positions)))
    if team_keys:
        out = out.filter(pl.col("team_key").is_in([int(t) for t in team_keys]))
    if price_min is not None:
        out = out.filter(pl.col("price").is_null() | (pl.col("price") >= price_min))
    if price_max is not None:
        out = out.filter(pl.col("price").is_null() | (pl.col("price") <= price_max))
    if min_p_played is not None:
        out = out.filter(
            pl.col("p_played").is_not_null() & (pl.col("p_played") >= min_p_played)
        )
    return out.select(
        "player_key",
        "web_name",
        "team",
        "position",
        "price",
        "status",
        "news",
        "p_played",
        *xpts_cols,
        "total_xpts",
        "in_squad",
    ).sort("total_xpts", descending=True)


def player_drilldown(
    player_key: int,
    predictions: pl.DataFrame,
    *,
    players: pl.DataFrame,
    fixtures: pl.DataFrame,
    teams: pl.DataFrame,
    season: str,
    minutes: pl.DataFrame | None = None,
    scoreline: pl.DataFrame | None = None,
    involvement: pl.DataFrame | None = None,
    points_pmf_gw: pl.DataFrame | None = None,
    as_of_gw: int | None = None,
) -> PlayerDrilldown:
    """Build fixtures / points-split / PMF for one player.

    Missing component tables set ``has_components`` False and leave depth
    columns null; the board and PMF still render.
    """
    key = int(player_key)
    pred = predictions.filter(pl.col("player_key") == key)
    has_components = bool(
        minutes is not None
        and scoreline is not None
        and involvement is not None
        and not minutes.is_empty()
        and not scoreline.is_empty()
        and not involvement.is_empty()
    )
    fixture_rows = _fixture_detail_rows(
        key,
        pred,
        fixtures=fixtures,
        teams=teams,
        players=players,
        minutes=minutes if has_components else None,
        scoreline=scoreline if has_components else None,
        involvement=involvement if has_components else None,
    )
    pos = "MID"
    if not players.is_empty() and "position" in players.columns:
        match = players.filter(pl.col("player_key") == key)
        if not match.is_empty():
            pos = str(match["position"][0])
    split = (
        build_forecast_board(
            pred.with_columns(pl.lit(pos).alias("position")),
            season=season,
            rules=Rules.for_season(season),
            player_keys=[key],
        )
        if not pred.is_empty()
        else pl.DataFrame()
    )
    next_gw = as_of_gw
    if next_gw is None and not pred.is_empty():
        next_gw = int(pred["target_gw"].min())  # type: ignore[arg-type]
    pmf = pl.DataFrame(schema={"points_bin": pl.Int64(), "probability": pl.Float64()})
    if (
        points_pmf_gw is not None
        and not points_pmf_gw.is_empty()
        and next_gw is not None
    ):
        pmf = (
            points_pmf_gw.filter(
                (pl.col("player_key") == key) & (pl.col("target_gw") == next_gw)
            )
            .select("points_bin", "probability")
            .sort("points_bin")
        )
    return PlayerDrilldown(
        fixtures=fixture_rows,
        points_split=split,
        points_pmf=pmf,
        has_components=has_components,
    )


def squad_keys_from_plan(plan: pl.DataFrame) -> list[int]:
    """Return step-0 player keys that remain in the planned squad."""
    if plan.is_empty() or "step" not in plan.columns:
        return []
    step0 = plan.filter((pl.col("step") == 0) & ~pl.col("sold"))
    return [int(k) for k in step0["player_key"].to_list()]


def header_fields(
    *,
    selection: OfficialSelection,
    predict: Any,
    plan: Any,
    deadline_utc: datetime | None,
) -> dict[str, Any]:
    """Build the page header dict from selection + run manifests."""
    return {
        "season": selection.season,
        "gw": selection.gw,
        "deadline_utc": deadline_utc,
        "line": selection.line,
        "checkpoint": selection.checkpoint,
        "role": selection.role,
        "predict_run_id": selection.predict_run_id,
        "plan_run_id": selection.plan_run_id,
        "code_sha": getattr(predict, "code_sha", None),
        "build_id": getattr(predict, "build_id", None),
        "finished_at": getattr(predict, "finished_at", None)
        or getattr(plan, "finished_at", None),
        "horizon": getattr(predict, "horizon", None),
    }


def _pick_line(
    pool: pl.DataFrame, *, applied_line: str | None
) -> OfficialSelection | None:
    if pool.is_empty():
        return None
    if applied_line:
        hit = pool.filter(pl.col("line") == applied_line)
        ranked = hit if not hit.is_empty() else pool.filter(pl.col("role") == "primary")
        if ranked.is_empty():
            ranked = pool
    else:
        primary = pool.filter(pl.col("role") == "primary")
        ranked = primary if not primary.is_empty() else pool
    row = ranked.sort("gw", descending=True).to_dicts()[0]
    return OfficialSelection(
        season=str(row["season"]),
        gw=int(row["gw"]),
        checkpoint=str(row["checkpoint"]),
        line=str(row["line"]),
        role=str(row["role"]),
        predict_run_id=str(row["predict_run_id"]),
        plan_run_id=str(row["plan_run_id"]),
    )


def _player_labels(
    players: pl.DataFrame, teams: pl.DataFrame, status: pl.DataFrame
) -> pl.DataFrame:
    """Join curated name / team / price / status for labelling."""
    empty = pl.DataFrame(
        schema={
            "player_key": pl.Int64(),
            "web_name": pl.Utf8(),
            "team": pl.Utf8(),
            "team_key": pl.Int64(),
            "position": pl.Utf8(),
            "price": pl.Float64(),
            "status": pl.Utf8(),
            "news": pl.Utf8(),
        }
    )
    if players.is_empty():
        return empty
    frame = players.select("player_key", "web_name", "team_key", "position")
    frame = (
        frame.join(
            teams.select("team_key", pl.col("short_name").alias("team")),
            on="team_key",
            how="left",
        )
        if not teams.is_empty()
        else frame.with_columns(pl.lit(None).cast(pl.Utf8()).alias("team"))
    )
    if not status.is_empty():
        latest = (
            status.sort("available_at")
            .unique(subset=["player_key"], keep="last")
            .select("player_key", "price", "status", "news")
        )
        frame = frame.join(latest, on="player_key", how="left")
    else:
        frame = frame.with_columns(
            pl.lit(None).cast(pl.Float64()).alias("price"),
            pl.lit(None).cast(pl.Utf8()).alias("status"),
            pl.lit(None).cast(pl.Utf8()).alias("news"),
        )
    return frame.with_columns(pl.col("price").cast(pl.Float64))


def _pair_transfers(
    sold: pl.DataFrame, bought: pl.DataFrame, labels: pl.DataFrame
) -> pl.DataFrame:
    n = min(sold.height, bought.height)
    if n == 0:
        return pl.DataFrame(schema=THIS_WEEK_TRANSFER_COLUMNS)
    label_map = {
        int(r["player_key"]): r
        for r in labels.select("player_key", "web_name", "price").to_dicts()
    }
    rows: list[dict[str, object]] = []
    for i in range(n):
        out_key = int(sold["player_key"][i])
        in_key = int(bought["player_key"][i])
        out_lab = label_map.get(out_key, {})
        in_lab = label_map.get(in_key, {})
        x_out = float(sold["xpts"][i])
        x_in = float(bought["xpts"][i])
        rows.append(
            {
                "player_key_out": out_key,
                "player_key_in": in_key,
                "name_out": out_lab.get("web_name") or _PLACEHOLDER,
                "name_in": in_lab.get("web_name") or _PLACEHOLDER,
                "price_out": out_lab.get("price"),
                "price_in": in_lab.get("price"),
                "xpts_out": x_out,
                "xpts_in": x_in,
                "delta_xpts": x_in - x_out,
            }
        )
    checked = ThisWeekTransferSchema.validate(
        pl.DataFrame(rows, schema=THIS_WEEK_TRANSFER_COLUMNS)
    )
    checked = require_dataframe(checked)
    return checked


def _fixture_detail_rows(
    player_key: int,
    pred: pl.DataFrame,
    *,
    fixtures: pl.DataFrame,
    teams: pl.DataFrame,
    players: pl.DataFrame,
    minutes: pl.DataFrame | None,
    scoreline: pl.DataFrame | None,
    involvement: pl.DataFrame | None,
) -> pl.DataFrame:
    if pred.is_empty():
        return pl.DataFrame(schema=THIS_WEEK_FIXTURE_DETAIL_COLUMNS)
    team_key: int | None = None
    if not players.is_empty() and "team_key" in players.columns:
        match = players.filter(pl.col("player_key") == player_key)
        if not match.is_empty():
            team_key = int(match["team_key"][0])
    team_names = {
        int(r["team_key"]): str(r["short_name"])
        for r in (
            teams.select("team_key", "short_name").to_dicts()
            if not teams.is_empty()
            else []
        )
    }
    by_fix = {
        int(r["fixture_id"]): r
        for r in (
            fixtures.select("fixture_id", "team_h_key", "team_a_key").to_dicts()
            if not fixtures.is_empty()
            else []
        )
    }
    mins_by = _by_fixture(
        minutes.filter(pl.col("player_key") == player_key)
        if minutes is not None and not minutes.is_empty()
        else None
    )
    inv_by = _by_fixture(
        involvement.filter(pl.col("player_key") == player_key)
        if involvement is not None and not involvement.is_empty()
        else None
    )
    sc_by = _by_fixture(
        scoreline.filter(pl.col("team_key") == team_key)
        if scoreline is not None and not scoreline.is_empty() and team_key is not None
        else None
    )
    rows: list[dict[str, object]] = []
    for row in pred.iter_rows(named=True):
        fid = int(row["fixture_id"])
        is_home: bool | None = None
        opponent: str | None = None
        fx_meta = by_fix.get(fid)
        if fx_meta is not None and team_key is not None:
            is_home = int(fx_meta["team_h_key"]) == team_key
            opp = int(fx_meta["team_a_key"] if is_home else fx_meta["team_h_key"])
            opponent = team_names.get(opp, str(opp))
        mins = mins_by.get(fid, {})
        inv = inv_by.get(fid, {})
        sc = sc_by.get(fid, {})
        if sc.get("is_home") is not None:
            is_home = bool(sc["is_home"])
        rows.append(
            {
                "player_key": player_key,
                "target_gw": int(row["target_gw"]),
                "fixture_id": fid,
                "opponent": opponent,
                "is_home": is_home,
                "xmins": mins.get("xmins"),
                "p_played": mins.get("p_played", row.get("p_played")),
                "p_60": mins.get("p_60", row.get("p_60")),
                "team_eg": sc.get("team_eg"),
                "opp_eg": sc.get("opp_eg"),
                "p_cs": sc.get("p_cs", row.get("p_cs")),
                "p_goal": row.get("p_goal"),
                "p_assist": row.get("p_assist"),
                "exp_goals": inv.get("exp_goals"),
                "exp_assists": inv.get("exp_assists"),
                "xpts": float(row["xpts"]),
            }
        )
    checked = ThisWeekFixtureDetailSchema.validate(
        pl.DataFrame(rows, schema=THIS_WEEK_FIXTURE_DETAIL_COLUMNS).sort(
            ["target_gw", "fixture_id"]
        )
    )
    checked = require_dataframe(checked)
    return checked


def _by_fixture(frame: pl.DataFrame | None) -> dict[int, dict[str, object]]:
    if frame is None or frame.is_empty():
        return {}
    return {int(r["fixture_id"]): r for r in frame.to_dicts()}


def _empty_players_table(target_gws: Sequence[int]) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {
        "player_key": pl.Int64(),
        "web_name": pl.Utf8(),
        "team": pl.Utf8(),
        "position": pl.Utf8(),
        "price": pl.Float64(),
        "status": pl.Utf8(),
        "news": pl.Utf8(),
        "p_played": pl.Float64(),
        **{f"xpts_gw{gw}": pl.Float64() for gw in target_gws},
        "total_xpts": pl.Float64(),
        "in_squad": pl.Boolean(),
    }
    return pl.DataFrame(schema=schema)


__all__ = [
    "OfficialSelection",
    "PlanPanel",
    "PlayerDrilldown",
    "default_official_selection",
    "header_fields",
    "plan_panel",
    "player_drilldown",
    "players_table",
    "squad_keys_from_plan",
]
