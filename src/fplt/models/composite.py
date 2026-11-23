"""Layered points model: scoreline, minutes, shares, then the simulator."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from fplt.data.pit import PITView
from fplt.models.components.involvement.shares import InvolvementShares
from fplt.models.components.involvement.xg import InvolvementXg
from fplt.models.components.minutes.empirical import EmpiricalMinutes
from fplt.models.components.minutes.empirical_v2 import EmpiricalMinutesV2
from fplt.models.components.other.match_conditioned import OtherMatchConditioned
from fplt.models.components.other.player import OtherPlayerPriors
from fplt.models.components.other.priors import OtherPointsPriors
from fplt.models.components.score.dixon_coles import MatchScoreModel
from fplt.models.distributions import MinutesDist, enforce_hard_starters
from fplt.models.predictions import prediction_frame
from fplt.models.simulation import ConcededMode, simulate_match
from fplt.models.spec import ModelSpec, get_component, register_model, write_identity
from fplt.models.universe import player_fixtures
from fplt.scoring.rules import Rules
from fplt.types import GWKey
from fplt.utils.rng import (
    active_eval_seed,
    eval_seed_for,
    player_seed_for,
    seed_for,
)
from fplt.utils.timing import timed_phase

logger = logging.getLogger(__name__)

_BUCKETS = ("0", "1-59", "60+")
_PARTS = ("score", "minutes", "involvement", "other")
# Importing the classes registers them. The tuple keeps those imports live.
_REGISTERED = (
    MatchScoreModel,
    EmpiricalMinutes,
    EmpiricalMinutesV2,
    InvolvementShares,
    InvolvementXg,
    OtherPointsPriors,
    OtherPlayerPriors,
    OtherMatchConditioned,
)


@register_model("dc_layered")
class CompositePredictor:
    """Dixon-Coles, minutes, involvement, and other points, combined by simulation.

    ``n_sims`` is the number of match scenarios. ``own_goal_weight`` and
    ``p_assist`` are passed to the simulator. Optional fidelity knobs
    (``cs_red_void_prob``, ``conceded_mode``, ``sample_other``,
    ``hard_starters``) default to the v1 approximations. The four components
    live under ``components`` in the spec.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        sims = spec.params.get("n_sims", 5000)
        own_goal = spec.params.get("own_goal_weight", 0.05)
        assist = spec.params.get("p_assist", 0.75)
        void_prob = spec.params.get("cs_red_void_prob", 0.0)
        conceded = spec.params.get("conceded_mode", "scaled")
        sample_other = spec.params.get("sample_other", False)
        hard = spec.params.get("hard_starters", False)
        target = spec.params.get("target_starters", 11)
        if isinstance(sims, bool) or not isinstance(sims, int) or sims < 1:
            raise ValueError(f"n_sims must be a positive int, got {sims!r}")
        own_goal_value = float(own_goal)
        assist_value = float(assist)
        void_value = float(void_prob)
        if own_goal_value < 0:
            raise ValueError(
                f"own_goal_weight must be non-negative, got {own_goal_value}"
            )
        if not 0.0 <= assist_value <= 1.0:
            raise ValueError(f"p_assist must be in [0, 1], got {assist_value}")
        if not 0.0 <= void_value <= 1.0:
            raise ValueError(f"cs_red_void_prob must be in [0, 1], got {void_value}")
        if conceded == "played":
            conceded_mode: ConcededMode = "played"
        elif conceded == "scaled":
            conceded_mode = "scaled"
        else:
            raise ValueError(
                f"conceded_mode must be 'scaled' or 'played', got {conceded!r}"
            )
        if not isinstance(sample_other, bool):
            raise ValueError(f"sample_other must be a bool, got {sample_other!r}")
        if not isinstance(hard, bool):
            raise ValueError(f"hard_starters must be a bool, got {hard!r}")
        if isinstance(target, bool) or not isinstance(target, int | float):
            raise ValueError(f"target_starters must be numeric, got {target!r}")
        target_value = int(target)
        if target_value < 1:
            raise ValueError(f"target_starters must be positive, got {target_value}")
        missing = [name for name in _PARTS if name not in spec.components]
        if missing:
            raise ValueError(f"dc_layered is missing components {missing}")
        self._n_sims = sims
        self._own_goal = own_goal_value
        self._p_assist = assist_value
        self._cs_red_void_prob = void_value
        self._conceded_mode = conceded_mode
        self._sample_other = sample_other
        self._hard_starters = hard
        self._target_starters = target_value
        self._score = _build(spec, "score")
        self._minutes = _build(spec, "minutes")
        self._involvement = _build(spec, "involvement")
        self._other = _build(spec, "other")
        self._fitted = False

    def fit(self, view: PITView) -> None:
        """Fit each component on rows visible at ``view.as_of``."""
        with timed_phase("fit/score"):
            self._score.fit(view)
        with timed_phase("fit/minutes"):
            self._minutes.fit(view)
        with timed_phase("fit/involvement"):
            self._involvement.fit(view)
        with timed_phase("fit/other"):
            self._other.fit(view)
        _apply_involvement_sim_params(self)
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
        with timed_phase("predict/setup"):
            rows = player_fixtures(view, horizon)
            if rows.is_empty():
                zero = pl.lit(0.0)
                return prediction_frame(
                    self.spec, view.as_of, rows, zero, zero, zero, zero
                )
            squad = _squad(view, rows)
            fixtures = list(_fixtures(view, squad).iter_rows(named=True))
        pieces: list[pl.DataFrame] = []
        with timed_phase("predict/fixtures"):
            for index, row in enumerate(fixtures, start=1):
                logger.info(
                    "%s fixture %s (%s/%s)",
                    self.spec.name,
                    row["fixture_id"],
                    index,
                    len(fixtures),
                )
                pieces.append(_fixture(self, view, squad, row))
            combined = pl.concat(pieces, how="vertical")
        return prediction_frame(
            self.spec,
            view.as_of,
            combined,
            pl.col("xpts"),
            pl.col("xpts_sd"),
            pl.col("p_haul"),
            pl.col("p_60"),
            p_played=pl.col("p_played"),
            p_goal=pl.col("p_goal"),
            p_assist=pl.col("p_assist"),
            p_cs=pl.col("p_cs"),
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
        _apply_involvement_sim_params(model)
        return model


def _apply_involvement_sim_params(model: CompositePredictor) -> None:
    """Prefer calibrated own-goal / assist knobs from involvement when present."""
    involvement = model._involvement
    if hasattr(involvement, "own_goal_weight"):
        model._own_goal = float(involvement.own_goal_weight)
    if hasattr(involvement, "p_assist"):
        model._p_assist = float(involvement.p_assist)


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
    pair_seed = active_eval_seed()
    if pair_seed is None:
        rng = seed_for(model.spec.model_id, view.as_of, fixture_id)
    else:
        rng = eval_seed_for(pair_seed, view.as_of, fixture_id)
    drawn = model._score.predict(view, pl.DataFrame([fixture]))
    scorelines = drawn.sample(rng, model._n_sims)
    minutes_dist = model._minutes.predict(view, asked)
    if pair_seed is None:
        minutes = minutes_dist.sample(rng, model._n_sims)
    else:
        minutes = _sample_minutes_crn(
            minutes_dist, pair_seed, view.as_of, fixture_id, model._n_sims
        )
    if model._hard_starters and minutes.size:
        minutes = enforce_hard_starters(
            minutes,
            np.asarray(players["team_key"].to_list(), dtype=np.int64),
            minutes_dist.start_weights(),
            rng,
            positions=np.asarray(players["position"].to_list()),
            target=model._target_starters,
        )
    weights = model._involvement.predict(view, asked)
    other = model._other.predict(view, asked)
    rules = Rules.for_season(str(players["season"][0]))
    match_other = (
        model._other if getattr(model._other, "match_conditioned", False) else None
    )
    result = simulate_match(
        scorelines[0],
        _sim_players(players, weights),
        minutes,
        _other_grid(other, players["player_key"].to_list()),
        rules,
        rng,
        own_goal_weight=model._own_goal,
        p_assist=model._p_assist,
        cs_red_void_prob=model._cs_red_void_prob,
        conceded_mode=model._conceded_mode,
        sample_other=model._sample_other,
        match_other=match_other,
    )
    return players.with_columns(
        pl.Series("xpts", result.points.mean(axis=1)),
        pl.Series("xpts_sd", result.points.std(axis=1)),
        pl.Series("p_played", (minutes > 0).mean(axis=1)),
        pl.Series("p_60", (minutes >= 60).mean(axis=1)),
        pl.Series("p_goal", (result.goals >= 1).mean(axis=1)),
        pl.Series("p_assist", (result.assists >= 1).mean(axis=1)),
        pl.Series("p_cs", result.clean_sheets.mean(axis=1)),
        pl.Series("p_haul", (result.points >= 10).mean(axis=1)),
    )


def _sample_minutes_crn(
    minutes: MinutesDist,
    seed: int,
    as_of: GWKey,
    fixture_id: int,
    n_sims: int,
) -> np.ndarray:
    """Draw minutes from a per-player eval stream (CRN-safe)."""
    rows = minutes.keys.height
    if n_sims == 0 or rows == 0:
        return np.zeros((rows, n_sims), dtype=np.int64)
    drawn = np.empty((rows, n_sims), dtype=np.int64)
    bins = np.asarray(minutes.bins, dtype=np.int64)
    width = bins.shape[0]
    for index, player_key in enumerate(minutes.keys["player_key"].to_list()):
        rng = player_seed_for(seed, as_of, fixture_id, int(player_key))
        cdf = np.cumsum(minutes.pmf[index])
        cdf[-1] = 1.0
        uniforms = rng.random(n_sims)
        choice = np.sum(cdf[None, :] <= uniforms[:, None], axis=-1)
        choice = np.minimum(choice, width - 1)
        drawn[index] = bins[choice]
    return drawn


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
