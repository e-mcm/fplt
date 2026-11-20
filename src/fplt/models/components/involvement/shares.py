"""Shrunk per-90 goal and assist shares, with a penalty-taker override.

Weights are relative inside the player's team for that fixture. A player
with no rate, on a team where nobody has a rate, shares that team equally.
"""

from __future__ import annotations

import polars as pl

from fplt.data.pit import PITView
from fplt.models.spec import ModelSpec, register_component
from fplt.schemas.components import INVOLVEMENT_COLUMNS, InvolvementSchema

_POSITIONS = ("GK", "DEF", "MID", "FWD")


class InvolvementShares:
    """Per-90 goal and assist weights shrunk toward the position.

    ``window`` is how many of the player's latest appearances set the rate.
    ``shrinkage`` is pseudo-nineties of the position's per-90 rate, which
    itself uses every visible appearance. A penalty miss anywhere in the
    visible history marks the taker and adds ``penalty_bonus`` goals per 90
    before the team shares are normalised. Own goals stay out of these
    weights; the simulator keeps an unattributed bucket for them.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        window = spec.params.get("window", 10)
        shrinkage = spec.params.get("shrinkage", 5.0)
        penalty = spec.params.get("penalty_bonus", 0.3)
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive int, got {window!r}")
        shrinkage_value = float(shrinkage)
        penalty_value = float(penalty)
        if shrinkage_value < 0:
            raise ValueError(f"shrinkage must be non-negative, got {shrinkage_value}")
        if penalty_value < 0:
            raise ValueError(f"penalty_bonus must be non-negative, got {penalty_value}")
        self._window = window
        self._shrinkage = shrinkage_value
        self._penalty = penalty_value
        self._fitted = False
        self._goal: dict[int, float] = {}
        self._assist: dict[int, float] = {}

    def fit(self, view: PITView) -> None:
        """Estimate per-90 rates from appearances visible on ``view``."""
        self._fitted = True
        self._goal = {}
        self._assist = {}
        history = _history(view)
        if history.is_empty():
            return
        prior = _position_rates(history)
        recent = _latest(history, self._window)
        takers = set(
            history.filter(pl.col("penalties_missed") > 0)["player_key"].to_list()
        )
        grouped = recent.group_by("player_key").agg(
            pl.col("position").last(),
            pl.col("goals_scored").sum(),
            pl.col("assists").sum(),
            pl.col("minutes").sum(),
        )
        for row in grouped.iter_rows(named=True):
            player = int(row["player_key"])
            position = row["position"] if row["position"] in _POSITIONS else ""
            goal_prior, assist_prior = prior.get(position, (0.0, 0.0))
            nineties = float(row["minutes"]) / 90.0
            self._goal[player] = _shrunk(
                float(row["goals_scored"]), nineties, goal_prior, self._shrinkage
            )
            self._assist[player] = _shrunk(
                float(row["assists"]), nineties, assist_prior, self._shrinkage
            )
            if player in takers:
                self._goal[player] += self._penalty

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
            return _validated(pl.DataFrame(schema=INVOLVEMENT_COLUMNS))
        teams = _teams(view)
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
            _share("goal_rate", "goal_sum").alias("goal_weight"),
            _share("assist_rate", "assist_sum").alias("assist_weight"),
        )
        return _validated(frame.select(list(INVOLVEMENT_COLUMNS)))


def _history(view: PITView) -> pl.DataFrame:
    stats = view.latest(
        "player_fixture_stats", ["season", "player_key", "fixture_id"]
    ).collect()
    if stats.is_empty():
        return stats
    roster = (
        view.latest("players", ["season", "player_key"])
        .select("season", "player_key", "position", "team_key")
        .collect()
    )
    return stats.join(roster, on=["season", "player_key"], how="left").sort(
        ["player_key", "season", "gw", "fixture_id"]
    )


def _teams(view: PITView) -> pl.DataFrame:
    roster = (
        view.latest("players", ["season", "player_key"])
        .select("player_key", "team_key", "available_at")
        .collect()
    )
    if roster.is_empty():
        return pl.DataFrame(schema={"player_key": pl.Int64(), "team_key": pl.Int64()})
    return (
        roster.sort("available_at")
        .unique(subset=["player_key"], keep="last")
        .select("player_key", "team_key")
    )


def _latest(history: pl.DataFrame, window: int) -> pl.DataFrame:
    ranked = history.with_columns(
        pl.col("player_key").cum_count().over("player_key").cast(pl.Int64).alias("_i"),
        pl.len().over("player_key").cast(pl.Int64).alias("_n"),
    )
    return ranked.filter(pl.col("_i") > pl.col("_n") - window).drop("_i", "_n")


def _position_rates(history: pl.DataFrame) -> dict[str, tuple[float, float]]:
    played = history.filter(
        (pl.col("minutes") > 0) & pl.col("position").is_in(list(_POSITIONS))
    )
    if played.is_empty():
        return {}
    totals = played.group_by("position").agg(
        pl.col("goals_scored").sum().alias("goals"),
        pl.col("assists").sum().alias("assists"),
        (pl.col("minutes") / 90.0).sum().alias("nineties"),
    )
    rates: dict[str, tuple[float, float]] = {}
    for row in totals.iter_rows(named=True):
        nineties = float(row["nineties"])
        rates[str(row["position"])] = (
            float(row["goals"]) / nineties,
            float(row["assists"]) / nineties,
        )
    return rates


def _shrunk(events: float, nineties: float, prior: float, shrinkage: float) -> float:
    denom = nineties + shrinkage
    if denom <= 0:
        return prior
    return (events + shrinkage * prior) / denom


def _share(rate: str, total: str) -> pl.Expr:
    even = 1.0 / pl.col("squad")
    return pl.when(pl.col(total) > 0).then(pl.col(rate) / pl.col(total)).otherwise(even)


def _validated(frame: pl.DataFrame) -> pl.DataFrame:
    checked = InvolvementSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


register_component("involvement_shares")(InvolvementShares)
