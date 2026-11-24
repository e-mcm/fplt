"""Shrunk per-90 goal and assist shares, with a penalty-taker override.

Weights are relative inside the player's team for that fixture. A player
with no rate, on a team where nobody has a rate, shares that team equally.
"""

from __future__ import annotations

import polars as pl

from fplt.data.pit import PITView
from fplt.models.component_params import (
    InvolvementSharesParams,
    parse_component_params,
)
from fplt.models.components.involvement.rates import (
    POSITIONS,
    appearance_history,
    current_teams,
    latest_rows,
    penalty_takers,
    position_goal_assist_rates,
    set_piece_takers,
    shrunk_rate,
    team_share,
    validated_involvement,
)
from fplt.models.spec import ModelSpec, register_component
from fplt.schemas.components import INVOLVEMENT_COLUMNS


class InvolvementShares:
    """Per-90 goal and assist weights shrunk toward the position.

    ``window`` is how many of the player's latest appearances set the rate.
    ``shrinkage`` is pseudo-nineties of the position's per-90 rate, which
    itself uses every visible appearance. The primary penalty taker
    (``penalties_order == 1`` on the latest PIT ``player_status``) gets
    ``penalty_bonus`` goals per 90 before team shares are normalised; when
    order is null, a visible ``penalties_missed`` is the fallback. Set
    ``use_penalties_order: false`` to force the miss-only heuristic for A/B
    (default ``true``). When ``use_penalty_event`` is true, skip the rate
    bump and expose ``p_penalty`` / ``penalty_taker_keys`` for the simulator.
    Optional ``use_set_piece_orders`` adds ``set_piece_bonus`` for primary
    corner / free-kick takers (default off). Own goals stay out of these
    weights; the simulator keeps an unattributed bucket.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        params = parse_component_params(InvolvementSharesParams, spec.params)
        self._window = params.window
        self._shrinkage = params.shrinkage
        self._penalty = params.penalty_bonus
        self._use_penalties_order = params.use_penalties_order
        self._use_penalty_event = params.use_penalty_event
        self._p_penalty_param = params.p_penalty
        self._use_set_piece_orders = params.use_set_piece_orders
        self._set_piece_bonus = params.set_piece_bonus
        self.p_penalty = 0.0
        self.penalty_taker_keys: set[int] = set()
        self._fitted = False
        self._goal: dict[int, float] = {}
        self._assist: dict[int, float] = {}

    def fit(self, view: PITView) -> None:
        """Estimate per-90 rates from appearances visible on ``view``."""
        self._fitted = True
        self._goal = {}
        self._assist = {}
        self.p_penalty = 0.0
        self.penalty_taker_keys = set()
        history = appearance_history(view)
        if history.is_empty():
            return
        prior = position_goal_assist_rates(history)
        recent = latest_rows(history, self._window)
        takers = penalty_takers(
            view, history, use_penalties_order=self._use_penalties_order
        )
        self.penalty_taker_keys = set(takers)
        if self._use_penalty_event:
            self.p_penalty = self._p_penalty_param
        set_takers: set[int] = (
            set_piece_takers(view) if self._use_set_piece_orders else set()
        )
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
            self._goal[player] = shrunk_rate(
                float(row["goals_scored"]), nineties, goal_prior, self._shrinkage
            )
            self._assist[player] = shrunk_rate(
                float(row["assists"]), nineties, assist_prior, self._shrinkage
            )
            if player in takers and not self._use_penalty_event:
                self._goal[player] += self._penalty
            if player in set_takers:
                self._goal[player] += self._set_piece_bonus
        for player in set_takers - set(self._goal):
            self._goal[player] = self._set_piece_bonus

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


register_component("involvement_shares")(InvolvementShares)
