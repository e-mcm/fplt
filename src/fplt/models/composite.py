"""Layered points model: scoreline, minutes, shares, then the simulator."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from fplt.data.pit import PITView
from fplt.models.components.involvement.shares import InvolvementShares
from fplt.models.components.minutes.empirical import EmpiricalMinutes
from fplt.models.components.other.priors import OtherPointsPriors
from fplt.models.components.score.dixon_coles import MatchScoreModel
from fplt.models.predictions import prediction_frame
from fplt.models.simulation import simulate_match
from fplt.models.spec import ModelSpec, get_component, register_model, write_identity
from fplt.models.universe import player_fixtures
from fplt.scoring.rules import Rules
from fplt.types import GWKey
from fplt.utils.rng import seed_for

_BUCKETS = ("0", "1-59", "60+")
_PARTS = ("score", "minutes", "involvement", "other")
# Importing the classes registers them. The tuple keeps those imports live.
_REGISTERED = (MatchScoreModel, EmpiricalMinutes, InvolvementShares, OtherPointsPriors)


@register_model("dc_layered")
class CompositePredictor:
    """Dixon-Coles, minutes, involvement, and other points, combined by simulation.

    ``n_sims`` is the number of match scenarios. ``own_goal_weight`` and
    ``p_assist`` are passed to the simulator. The four components live under
    ``components`` in the spec. A clean sheet still requires 60 minutes
    against a scoreless opponent, and the goals-conceded penalty scales with
    minutes/90.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        sims = spec.params.get("n_sims", 5000)
        own_goal = spec.params.get("own_goal_weight", 0.05)
        assist = spec.params.get("p_assist", 0.75)
        if isinstance(sims, bool) or not isinstance(sims, int) or sims < 1:
            raise ValueError(f"n_sims must be a positive int, got {sims!r}")
        own_goal_value = float(own_goal)
        assist_value = float(assist)
        if own_goal_value < 0:
            raise ValueError(
                f"own_goal_weight must be non-negative, got {own_goal_value}"
            )
        if not 0.0 <= assist_value <= 1.0:
            raise ValueError(f"p_assist must be in [0, 1], got {assist_value}")
        missing = [name for name in _PARTS if name not in spec.components]
        if missing:
            raise ValueError(f"dc_layered is missing components {missing}")
        self._n_sims = sims
        self._own_goal = own_goal_value
        self._p_assist = assist_value
        self._score = _build(spec, "score")
        self._minutes = _build(spec, "minutes")
        self._involvement = _build(spec, "involvement")
        self._other = _build(spec, "other")
        self._fitted = False

    def fit(self, view: PITView) -> None:
        """Fit each component on rows visible at ``view.as_of``."""
        self._score.fit(view)
        self._minutes.fit(view)
        self._involvement.fit(view)
        self._other.fit(view)
        self._fitted = True

    def predict(self, view: PITView, horizon: list[GWKey]) -> pl.DataFrame:
        """Simulate every player-fixture in ``horizon``.

        Returns:
            A frame matching ``PredictionSchema``.

        Raises:
            RuntimeError: ``fit`` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("fit the layered model before predicting")
        rows = player_fixtures(view, horizon)
        if rows.is_empty():
            zero = pl.lit(0.0)
            return prediction_frame(self.spec, view.as_of, rows, zero, zero, zero, zero)
        squad = _squad(view, rows)
        pieces = [
            _fixture(self, view, squad, row)
            for row in _fixtures(view, squad).iter_rows(named=True)
        ]
        combined = pl.concat(pieces, how="vertical")
        return prediction_frame(
            self.spec,
            view.as_of,
            combined,
            pl.col("xpts"),
            pl.col("xpts_sd"),
            pl.col("p_haul"),
            pl.col("p_60"),
        )

    def save(self, path: Path) -> None:
        """Write the fitted components. Refuses to overwrite an existing artifact."""
        write_identity(path, self.spec)
        payload = {
            "fitted": self._fitted,
            "score": self._score,
            "minutes": self._minutes,
            "involvement": self._involvement,
            "other": self._other,
        }
        (path / "state.pkl").write_bytes(pickle.dumps(payload))

    @classmethod
    def load(cls, path: Path, spec: ModelSpec) -> CompositePredictor:
        """Load components written by ``save``."""
        model = cls(spec)
        payload = pickle.loads((path / "state.pkl").read_bytes())
        model._fitted = bool(payload["fitted"])
        model._score = payload["score"]
        model._minutes = payload["minutes"]
        model._involvement = payload["involvement"]
        model._other = payload["other"]
        return model


def _build(spec: ModelSpec, name: str) -> Any:
    component = spec.components[name]
    return get_component(component.kind)(component)


def _fixtures(view: PITView, squad: pl.DataFrame) -> pl.DataFrame:
    ids = squad.select("season", "fixture_id").unique()
    fixtures = view.latest("fixtures", by=["season", "fixture_id"]).collect()
    return fixtures.join(ids, on=["season", "fixture_id"], how="inner").sort(
        "fixture_id"
    )


def _fixture(
    model: CompositePredictor,
    view: PITView,
    squad: pl.DataFrame,
    fixture: dict[str, Any],
) -> pl.DataFrame:
    fixture_id = int(fixture["fixture_id"])
    players = squad.filter(pl.col("fixture_id") == fixture_id).sort("player_key")
    asked = players.select("player_key", "fixture_id")
    rng = seed_for(model.spec.model_id, view.as_of, fixture_id)
    drawn = model._score.predict(view, pl.DataFrame([fixture]))
    scorelines = drawn.sample(rng, model._n_sims)
    minutes = model._minutes.predict(view, asked).sample(rng, model._n_sims)
    weights = model._involvement.predict(view, asked)
    other = model._other.predict(view, asked)
    rules = Rules.for_season(str(players["season"][0]))
    result = simulate_match(
        scorelines[0],
        _sim_players(players, weights),
        minutes,
        _other_grid(other, players["player_key"].to_list()),
        rules,
        rng,
        own_goal_weight=model._own_goal,
        p_assist=model._p_assist,
    )
    return players.with_columns(
        pl.Series("xpts", result.points.mean(axis=1)),
        pl.Series("xpts_sd", result.points.std(axis=1)),
        pl.Series("p_haul", (result.points >= 10).mean(axis=1)),
        pl.Series("p_60", (minutes >= 60).mean(axis=1)),
    )


def _squad(view: PITView, rows: pl.DataFrame) -> pl.DataFrame:
    players = (
        view.latest("players", by=["season", "player_key"])
        .select("season", "player_key", "team_key", "position")
        .collect()
    )
    fixtures = (
        view.latest("fixtures", by=["season", "fixture_id"])
        .select("season", "fixture_id", "team_h_key", "team_a_key")
        .collect()
    )
    joined = rows.join(players, on=["season", "player_key"], how="inner").join(
        fixtures, on=["season", "fixture_id"], how="inner"
    )
    return joined.with_columns(
        pl.when(pl.col("team_key") == pl.col("team_h_key"))
        .then(pl.lit("home"))
        .otherwise(pl.lit("away"))
        .alias("side")
    )


def _sim_players(players: pl.DataFrame, weights: pl.DataFrame) -> pl.DataFrame:
    return (
        players.join(weights, on=["player_key", "fixture_id"], how="left")
        .with_columns(
            pl.col("goal_weight").fill_null(0.0),
            pl.col("assist_weight").fill_null(0.0),
        )
        .select("player_key", "side", "position", "goal_weight", "assist_weight")
    )


def _other_grid(other: pl.DataFrame, player_keys: list[int]) -> np.ndarray:
    grid = np.zeros((len(player_keys), 3), dtype=np.float64)
    index = {player: row for row, player in enumerate(player_keys)}
    buckets = {name: column for column, name in enumerate(_BUCKETS)}
    for row in other.iter_rows(named=True):
        value = (
            float(row["bonus"])
            + float(row["saves"])
            + float(row["cards"])
            + float(row["defensive_contribution"])
            + float(row["other"])
        )
        slot = index[int(row["player_key"])]
        grid[slot, buckets[str(row["minutes_bucket"])]] = value
    return grid
