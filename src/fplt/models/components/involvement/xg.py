"""xG/xA involvement shares blended with FPL goals and assists.

When Understat rows are visible for a player, goal and assist rates mix
shrunk xG/xA per 90 with shrunk FPL goals/assists. Players with no Understat
mapping keep the FPL-only rates. ``own_goal_weight`` and ``p_assist`` are
estimated from the same visible history so the layered simulator can drop the
YAML guesses.
"""

from __future__ import annotations

import polars as pl

from fplt.data.pit import PITView
from fplt.models.components.involvement.rates import (
    POSITIONS,
    appearance_history,
    current_teams,
    latest_rows,
    penalty_takers,
    position_goal_assist_rates,
    shrunk_rate,
    team_share,
    validated_involvement,
)
from fplt.models.spec import ModelSpec, register_component
from fplt.schemas.components import INVOLVEMENT_COLUMNS

_DEFAULT_OWN_GOAL = 0.05
_DEFAULT_P_ASSIST = 0.75


class InvolvementXg:
    """Team-relative goal/assist weights from Understat xG/xA plus FPL.

    ``window`` and ``shrinkage`` match ``InvolvementShares``. ``xg_blend`` in
    ``[0, 1]`` is the weight on the Understat rate when both signals exist.
    ``penalty_bonus`` is added to the blended goal rate for the primary
    penalty taker (``penalties_order == 1``), falling back to a visible
    penalty miss when order is null. Set ``use_penalties_order: false`` to
    force the miss-only heuristic for A/B (default ``true``). After ``fit``,
    ``own_goal_weight`` and ``p_assist`` hold calibrated simulator knobs
    (defaults when history is empty).
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        window = spec.params.get("window", 10)
        shrinkage = spec.params.get("shrinkage", 5.0)
        penalty = spec.params.get("penalty_bonus", 0.3)
        blend = spec.params.get("xg_blend", 0.7)
        use_order = spec.params.get("use_penalties_order", True)
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive int, got {window!r}")
        shrinkage_value = float(shrinkage)
        penalty_value = float(penalty)
        blend_value = float(blend)
        if shrinkage_value < 0:
            raise ValueError(f"shrinkage must be non-negative, got {shrinkage_value}")
        if penalty_value < 0:
            raise ValueError(f"penalty_bonus must be non-negative, got {penalty_value}")
        if not 0.0 <= blend_value <= 1.0:
            raise ValueError(f"xg_blend must be in [0, 1], got {blend_value}")
        if not isinstance(use_order, bool):
            raise ValueError(f"use_penalties_order must be a bool, got {use_order!r}")
        self._window = window
        self._shrinkage = shrinkage_value
        self._penalty = penalty_value
        self._xg_blend = blend_value
        self._use_penalties_order = use_order
        self.own_goal_weight = _DEFAULT_OWN_GOAL
        self.p_assist = _DEFAULT_P_ASSIST
        self._fitted = False
        self._goal: dict[int, float] = {}
        self._assist: dict[int, float] = {}

    def fit(self, view: PITView) -> None:
        """Estimate blended rates and simulator knobs from ``view``."""
        self._fitted = True
        self._goal = {}
        self._assist = {}
        self.own_goal_weight = _DEFAULT_OWN_GOAL
        self.p_assist = _DEFAULT_P_ASSIST
        history = appearance_history(view)
        fpl_goal, fpl_assist = _fpl_rates(history, self._window, self._shrinkage)
        xg_goal, xg_assist = _xg_rates(view, history, self._window, self._shrinkage)
        takers = penalty_takers(
            view, history, use_penalties_order=self._use_penalties_order
        )
        players = set(fpl_goal) | set(fpl_assist) | set(xg_goal) | set(xg_assist)
        for player in players:
            self._goal[player] = _blend_rate(
                fpl_goal.get(player), xg_goal.get(player), self._xg_blend
            )
            self._assist[player] = _blend_rate(
                fpl_assist.get(player), xg_assist.get(player), self._xg_blend
            )
            if player in takers:
                self._goal[player] = self._goal.get(player, 0.0) + self._penalty
        if not history.is_empty():
            self.own_goal_weight, self.p_assist = _sim_params(history)

    def predict(self, view: PITView, player_fixtures: pl.DataFrame) -> pl.DataFrame:
        """Team-relative weights for each player-fixture.

        Args:
            view: Deadline view. The current squad comes from here.
            player_fixtures: Columns ``player_key`` and ``fixture_id``.

        Returns:
            A frame matching ``InvolvementSchema``.

        Raises:
            RuntimeError: ``fit`` has not been called.
            ValueError: ``player_fixtures`` lacks a required column.
        """
        if not self._fitted:
            raise RuntimeError("fit the involvement model before predicting")
        missing = {"player_key", "fixture_id"} - set(player_fixtures.columns)
        if missing:
            raise ValueError(f"player_fixtures is missing {sorted(missing)}")
        asked = player_fixtures.select("player_key", "fixture_id").unique(
            maintain_order=True
        )
        if asked.is_empty():
            return validated_involvement(pl.DataFrame(schema=INVOLVEMENT_COLUMNS))
        teams = current_teams(view)
        frame = asked.join(teams, on="player_key", how="left").with_columns(
            pl.col("team_key").fill_null(-pl.col("player_key"))
        )
        goal = frame["player_key"].replace_strict(self._goal, default=0.0)
        assist = frame["player_key"].replace_strict(self._assist, default=0.0)
        frame = frame.with_columns(goal.alias("goal_rate"), assist.alias("assist_rate"))
        frame = frame.with_columns(
            pl.col("goal_rate").sum().over("fixture_id", "team_key").alias("goal_sum"),
            pl.col("assist_rate")
            .sum()
            .over("fixture_id", "team_key")
            .alias("assist_sum"),
            pl.len().over("fixture_id", "team_key").alias("squad"),
        )
        frame = frame.with_columns(
            team_share("goal_rate", "goal_sum").alias("goal_weight"),
            team_share("assist_rate", "assist_sum").alias("assist_weight"),
        )
        return validated_involvement(frame.select(list(INVOLVEMENT_COLUMNS)))


def _fpl_rates(
    history: pl.DataFrame, window: int, shrinkage: float
) -> tuple[dict[int, float], dict[int, float]]:
    if history.is_empty():
        return {}, {}
    prior = position_goal_assist_rates(history)
    recent = latest_rows(history, window)
    goal: dict[int, float] = {}
    assist: dict[int, float] = {}
    grouped = recent.group_by("player_key").agg(
        pl.col("position").last(),
        pl.col("goals_scored").sum(),
        pl.col("assists").sum(),
        pl.col("minutes").sum(),
    )
    for row in grouped.iter_rows(named=True):
        player = int(row["player_key"])
        position = row["position"] if row["position"] in POSITIONS else ""
        goal_prior, assist_prior = prior.get(position, (0.0, 0.0))
        nineties = float(row["minutes"]) / 90.0
        goal[player] = shrunk_rate(
            float(row["goals_scored"]), nineties, goal_prior, shrinkage
        )
        assist[player] = shrunk_rate(
            float(row["assists"]), nineties, assist_prior, shrinkage
        )
    return goal, assist


def _xg_rates(
    view: PITView, history: pl.DataFrame, window: int, shrinkage: float
) -> tuple[dict[int, float], dict[int, float]]:
    matches = (
        view.table("understat_player_matches")
        .filter(pl.col("player_key").is_not_null())
        .collect()
    )
    if matches.is_empty():
        return {}, {}
    roster = (
        history.select("player_key", "position")
        if not history.is_empty()
        else pl.DataFrame(schema={"player_key": pl.Int64(), "position": pl.Utf8()})
    )
    if roster.is_empty():
        roster = (
            view.latest("players", ["season", "player_key"])
            .select("player_key", "position")
            .collect()
        )
    positions = (
        roster.sort("player_key")
        .unique(subset=["player_key"], keep="last")
        .select("player_key", "position")
    )
    keyed = matches.with_columns(pl.col("player_key").cast(pl.Int64)).join(
        positions, on="player_key", how="left"
    )
    prior = _xg_position_rates(keyed)
    recent = _latest_matches(keyed, window)
    goal: dict[int, float] = {}
    assist: dict[int, float] = {}
    grouped = recent.group_by("player_key").agg(
        pl.col("position").last(),
        pl.col("xg").sum(),
        pl.col("xa").sum(),
        pl.col("minutes").sum(),
    )
    for row in grouped.iter_rows(named=True):
        player = int(row["player_key"])
        position = row["position"] if row["position"] in POSITIONS else ""
        goal_prior, assist_prior = prior.get(position, (0.0, 0.0))
        nineties = float(row["minutes"]) / 90.0
        goal[player] = shrunk_rate(float(row["xg"]), nineties, goal_prior, shrinkage)
        assist[player] = shrunk_rate(
            float(row["xa"]), nineties, assist_prior, shrinkage
        )
    return goal, assist


def _xg_position_rates(matches: pl.DataFrame) -> dict[str, tuple[float, float]]:
    played = matches.filter(
        (pl.col("minutes") > 0) & pl.col("position").is_in(list(POSITIONS))
    )
    if played.is_empty():
        return {}
    totals = played.group_by("position").agg(
        pl.col("xg").sum().alias("xg"),
        pl.col("xa").sum().alias("xa"),
        (pl.col("minutes") / 90.0).sum().alias("nineties"),
    )
    rates: dict[str, tuple[float, float]] = {}
    for row in totals.iter_rows(named=True):
        nineties = float(row["nineties"])
        rates[str(row["position"])] = (
            float(row["xg"]) / nineties,
            float(row["xa"]) / nineties,
        )
    return rates


def _latest_matches(matches: pl.DataFrame, window: int) -> pl.DataFrame:
    ranked = matches.sort(["player_key", "available_at", "match_id"]).with_columns(
        pl.col("player_key").cum_count().over("player_key").cast(pl.Int64).alias("_i"),
        pl.len().over("player_key").cast(pl.Int64).alias("_n"),
    )
    return ranked.filter(pl.col("_i") > pl.col("_n") - window).drop("_i", "_n")


def _blend_rate(fpl: float | None, understat: float | None, blend: float) -> float:
    if fpl is not None and understat is not None:
        return blend * understat + (1.0 - blend) * fpl
    if understat is not None:
        return understat
    if fpl is not None:
        return fpl
    return 0.0


def _sim_params(history: pl.DataFrame) -> tuple[float, float]:
    goals = float(history["goals_scored"].sum())
    assists = float(history["assists"].sum())
    own = float(history["own_goals"].sum())
    if goals > 0:
        p_assist = min(1.0, max(0.0, assists / goals))
    else:
        p_assist = _DEFAULT_P_ASSIST
    total = goals + own
    if total > 0 and own < total:
        rate = own / total
        own_goal_weight = rate / (1.0 - rate)
    elif total > 0:
        own_goal_weight = 1.0
    else:
        own_goal_weight = _DEFAULT_OWN_GOAL
    return own_goal_weight, p_assist


register_component("involvement_xg")(InvolvementXg)
