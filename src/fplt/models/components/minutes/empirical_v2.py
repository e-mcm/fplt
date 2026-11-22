"""Minutes v2: multi-window rates, context logits, soft ~11-starter projection.

Independence across teammates remains the default draw model. When
``soft_squad`` is on, expected starters per team-fixture are nudged toward
``target_starters`` before the PMF is built — a soft stand-in for a hard
eleven-starter constraint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import polars as pl

from fplt.data.pit import PITView
from fplt.models.components.minutes.common import (
    START_FALLBACK,
    SUB_FALLBACK,
    WIDTH,
    availability,
    chance,
    counts,
    gate_for,
    normalise,
    positions,
    shrunk,
)
from fplt.models.distributions import MINUTE_BINS, MinutesDist
from fplt.models.spec import ModelSpec, register_component
from fplt.types import Position

_LONG = MINUTE_BINS >= 60
_START_DIM = 10
_LONG_DIM = 5
_DEFAULT_WINDOWS = (3, 5, 10)


class EmpiricalMinutesV2:
    """Logistic P(start) and P(≥60|start) with histogram residuals.

    Rolling start shares over ``windows``, minutes volatility, home/away,
    rest days, DGW, price, and opponent strength feed two small logistics
    fit on appearances visible at the deadline. Conditional minute shapes
    still come from shrunk start/sub histograms. Availability
    (status / chance_of_playing) gates both rates after the logits.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        windows = spec.params.get("windows", list(_DEFAULT_WINDOWS))
        shrinkage = spec.params.get("shrinkage", 5.0)
        doubtful = spec.params.get("doubtful_chance", 0.5)
        soft = spec.params.get("soft_squad", True)
        target = spec.params.get("target_starters", 11.0)
        strength = spec.params.get("soft_squad_strength", 0.5)
        l2 = spec.params.get("l2", 1.0)
        parsed = _parse_windows(windows)
        shrinkage_value = _as_float(shrinkage)
        doubtful_value = _as_float(doubtful)
        target_value = _as_float(target)
        strength_value = _as_float(strength)
        l2_value = _as_float(l2)
        if shrinkage_value < 0:
            raise ValueError(f"shrinkage must be non-negative, got {shrinkage_value}")
        if not 0.0 <= doubtful_value <= 1.0:
            raise ValueError(f"doubtful_chance must be in [0, 1], got {doubtful_value}")
        if target_value <= 0:
            raise ValueError(f"target_starters must be positive, got {target_value}")
        if not 0.0 <= strength_value <= 1.0:
            raise ValueError(
                f"soft_squad_strength must be in [0, 1], got {strength_value}"
            )
        if l2_value < 0:
            raise ValueError(f"l2 must be non-negative, got {l2_value}")
        if not isinstance(soft, bool):
            raise ValueError(f"soft_squad must be a bool, got {soft!r}")
        self._windows = tuple(sorted(set(parsed)))
        self._shrinkage = shrinkage_value
        self._doubtful = doubtful_value
        self._soft_squad = soft
        self._target_starters = target_value
        self._soft_strength = strength_value
        self._l2 = l2_value
        self._fitted = False
        self._position: dict[int, str] = {}
        self._player_rate: dict[int, tuple[float, float]] = {}
        self._player_features: dict[int, dict[str, float]] = {}
        self._player_start: dict[int, np.ndarray] = {}
        self._player_sub: dict[int, np.ndarray] = {}
        self._position_rate: dict[str, tuple[float, float]] = {}
        self._position_start: dict[str, np.ndarray] = {}
        self._position_sub: dict[str, np.ndarray] = {}
        self._start_beta = np.zeros(_START_DIM, dtype=np.float64)
        self._long_beta = np.zeros(_LONG_DIM, dtype=np.float64)

    def fit(self, view: PITView) -> None:
        """Estimate rates, histograms, and context logits from ``view``."""
        self._fitted = True
        self._position = positions(view)
        stats = (
            view.latest("player_fixture_stats", ["season", "player_key", "fixture_id"])
            .collect()
            .sort(["season", "gw", "fixture_id", "available_at"])
        )
        self._player_rate = {}
        self._player_features = {}
        self._player_start = {}
        self._player_sub = {}
        self._position_rate = {}
        self._position_start = {}
        self._position_sub = {}
        self._start_beta = np.zeros(_START_DIM, dtype=np.float64)
        self._long_beta = np.zeros(_LONG_DIM, dtype=np.float64)
        if stats.is_empty():
            return
        start_counts: dict[str, np.ndarray] = {}
        sub_counts: dict[str, np.ndarray] = {}
        seen: dict[str, int] = {}
        started: dict[str, int] = {}
        subbed: dict[str, int] = {}
        for key, group in stats.group_by("player_key", maintain_order=True):
            player = int(key[0])
            position = self._position.get(player, "UNK")
            self._remember_player(player, group)
            _accumulate_position(
                position, group, start_counts, sub_counts, seen, started, subbed
            )
        for position, total in seen.items():
            self._position_rate[position] = (
                started.get(position, 0) / total,
                subbed.get(position, 0) / total,
            )
            self._position_start[position] = normalise(
                start_counts.get(position), START_FALLBACK
            )
            self._position_sub[position] = normalise(
                sub_counts.get(position), SUB_FALLBACK
            )
        training = _training_rows(view, stats, self._windows)
        if training.height > 0:
            start_x = np.column_stack(
                [
                    np.ones(training.height),
                    training.select(_START_FEATURE_COLS).to_numpy(),
                ]
            )
            long_x = np.column_stack(
                [
                    np.ones(training.height),
                    training.select(_LONG_FEATURE_COLS).to_numpy(),
                ]
            )
            self._start_beta = _fit_logit(
                start_x,
                np.asarray(training["y_start"].to_numpy(), dtype=np.float64),
                self._l2,
            )
            starters = training.filter(pl.col("y_start") > 0.5)
            if starters.height > 0:
                long_x = np.column_stack(
                    [
                        np.ones(starters.height),
                        starters.select(_LONG_FEATURE_COLS).to_numpy(),
                    ]
                )
                self._long_beta = _fit_logit(
                    long_x,
                    np.asarray(starters["y_long"].to_numpy(), dtype=np.float64),
                    self._l2,
                )

    def predict(self, view: PITView, player_fixtures: pl.DataFrame) -> MinutesDist:
        """Minutes distribution for each player-fixture.

        Args:
            view: Deadline view used for status, fixtures, and team maps.
            player_fixtures: Columns ``player_key`` and ``fixture_id``.

        Returns:
            A ``MinutesDist`` aligned with ``player_fixtures``.
        """
        if not self._fitted:
            raise RuntimeError("fit the minutes model before predicting")
        missing = {"player_key", "fixture_id"} - set(player_fixtures.columns)
        if missing:
            raise ValueError(f"player_fixtures is missing {sorted(missing)}")
        keys = player_fixtures.select("player_key", "fixture_id")
        if keys.is_empty():
            return MinutesDist(keys, MINUTE_BINS, np.zeros((0, WIDTH)))
        context = _predict_context(view, keys)
        chance_map = availability(view)
        p_start = np.zeros(keys.height, dtype=np.float64)
        p_sub = np.zeros(keys.height, dtype=np.float64)
        p_long = np.zeros(keys.height, dtype=np.float64)
        start_hist = np.zeros((keys.height, WIDTH), dtype=np.float64)
        sub_hist = np.zeros((keys.height, WIDTH), dtype=np.float64)
        for index, row in enumerate(context.iter_rows(named=True)):
            player = int(row["player_key"])
            position = self._position.get(player, "UNK")
            tags = chance_map.get(player)
            gate = (
                chance(None, None, self._doubtful)
                if tags is None
                else gate_for(tags, self._doubtful)
            )
            features = self._player_features.get(player, _empty_player_features())
            start_x = _start_vector(features, row)
            long_x = _long_vector(features, row)
            raw_start = float(_sigmoid(start_x @ self._start_beta))
            raw_long = float(_sigmoid(long_x @ self._long_beta))
            if player in self._player_rate:
                base_start, base_sub = self._player_rate[player]
            else:
                base_start, base_sub = self._position_rate.get(position, (0.0, 0.0))
            # Blend logistic with the multi-window empirical rate so sparse
            # early-season fits do not ignore role stickiness.
            blended = 0.5 * raw_start + 0.5 * base_start
            p_start[index] = blended * gate
            p_sub[index] = base_sub * gate * (1.0 - blended)
            p_long[index] = raw_long
            start_hist[index] = shrunk(
                self._player_start.get(player),
                self._position_start.get(position),
                START_FALLBACK,
                self._shrinkage,
            )
            sub_hist[index] = shrunk(
                self._player_sub.get(player),
                self._position_sub.get(position),
                SUB_FALLBACK,
                self._shrinkage,
            )
        if self._soft_squad:
            p_start = _soft_project_starters(
                context, p_start, self._target_starters, self._soft_strength
            )
        pmf = np.zeros((keys.height, WIDTH), dtype=np.float64)
        for index in range(keys.height):
            played = p_start[index] + p_sub[index]
            if played > 1.0:
                p_start[index] /= played
                p_sub[index] /= played
                played = 1.0
            start = _reweight_long(start_hist[index], p_long[index])
            pmf[index, 0] = 1.0 - played
            pmf[index] += p_start[index] * start
            pmf[index] += p_sub[index] * sub_hist[index]
        return MinutesDist(keys, MINUTE_BINS, pmf)

    def _remember_player(self, player: int, group: pl.DataFrame) -> None:
        minutes = group["minutes"].to_numpy()
        starts = group["starts"].to_numpy()
        features: dict[str, float] = {}
        for window in self._windows:
            recent = group.tail(window)
            n = recent.height
            rate = float((recent["starts"] > 0).sum() / n) if n else 0.0
            features[f"starts_share_{window}"] = rate
            features[f"minutes_mean_{window}"] = (
                _as_float(recent["minutes"].mean()) if n else 0.0
            )
        recent5 = group.tail(5)
        if recent5.height >= 2:
            features["minutes_std_5"] = _as_float(recent5["minutes"].std())
        else:
            features["minutes_std_5"] = 0.0
        recent = group.tail(max(self._windows))
        n = recent.height
        start_arr = recent["starts"].to_numpy()
        minute_arr = recent["minutes"].to_numpy()
        self._player_rate[player] = (
            float(np.sum(start_arr > 0) / n) if n else 0.0,
            float(np.sum((start_arr == 0) & (minute_arr > 0)) / n) if n else 0.0,
        )
        self._player_features[player] = features
        self._player_start[player] = counts(minutes[starts > 0])
        self._player_sub[player] = counts(minutes[(starts == 0) & (minutes > 0)])


_START_FEATURE_COLS = [
    "starts_share_3",
    "starts_share_5",
    "starts_share_10",
    "minutes_std_5",
    "was_home",
    "rest_days",
    "is_dgw",
    "price",
    "opp_strength",
]
_LONG_FEATURE_COLS = [
    "minutes_mean_5",
    "starts_share_5",
    "minutes_std_5",
    "was_home",
]


def _empty_player_features() -> dict[str, float]:
    return {
        "starts_share_3": 0.0,
        "starts_share_5": 0.0,
        "starts_share_10": 0.0,
        "minutes_mean_5": 0.0,
        "minutes_std_5": 0.0,
    }


def _accumulate_position(
    position: str,
    group: pl.DataFrame,
    start_counts: dict[str, np.ndarray],
    sub_counts: dict[str, np.ndarray],
    seen: dict[str, int],
    started: dict[str, int],
    subbed: dict[str, int],
) -> None:
    minutes = group["minutes"].to_numpy()
    starts = group["starts"].to_numpy()
    start_rows = minutes[starts > 0]
    sub_rows = minutes[(starts == 0) & (minutes > 0)]
    start_counts[position] = start_counts.get(position, np.zeros(WIDTH)) + counts(
        start_rows
    )
    sub_counts[position] = sub_counts.get(position, np.zeros(WIDTH)) + counts(sub_rows)
    seen[position] = seen.get(position, 0) + len(group)
    started[position] = started.get(position, 0) + int(len(start_rows))
    subbed[position] = subbed.get(position, 0) + int(len(sub_rows))


def _training_rows(
    view: PITView, stats: pl.DataFrame, windows: tuple[int, ...]
) -> pl.DataFrame:
    """Shifted rolling features joined to fixture context for logistic fit."""
    empty = pl.DataFrame(
        schema={
            "y_start": pl.Float64(),
            "y_long": pl.Float64(),
            **{name: pl.Float64() for name in _START_FEATURE_COLS},
            "minutes_mean_5": pl.Float64(),
        }
    )
    if stats.is_empty():
        return empty
    ordered = stats.sort(["player_key", "available_at"])
    exprs: list[pl.Expr] = []
    shifted_starts = pl.col("starts").shift(1)
    shifted_minutes = pl.col("minutes").shift(1)
    for window in (3, 5, 10):
        exprs.append(
            (shifted_starts > 0)
            .cast(pl.Float64)
            .rolling_mean(window_size=window, min_samples=1)
            .over("player_key")
            .alias(f"starts_share_{window}")
        )
    exprs.append(
        shifted_minutes.rolling_mean(window_size=5, min_samples=1)
        .over("player_key")
        .alias("minutes_mean_5")
    )
    exprs.append(
        shifted_minutes.rolling_std(window_size=5, min_samples=2)
        .over("player_key")
        .alias("minutes_std_5")
    )
    rolled = ordered.with_columns(exprs).drop_nulls(
        subset=["starts_share_3", "starts_share_5", "starts_share_10"]
    )
    if rolled.is_empty():
        return empty
    context = _fixture_context(view)
    players = (
        view.latest("players", ["season", "player_key"])
        .select("season", "player_key", "team_key", "position")
        .collect()
    )
    status = (
        view.latest("player_status", ["season", "player_key"])
        .select("season", "player_key", "price")
        .collect()
    )
    joined = (
        rolled.join(players, on=["season", "player_key"], how="left")
        .join(status, on=["season", "player_key"], how="left")
        .join(context, on=["season", "fixture_id"], how="left")
        .with_columns(
            (pl.col("team_key") == pl.col("team_h_key"))
            .cast(pl.Float64)
            .alias("was_home"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("team_a_key"))
            .otherwise(pl.col("team_h_key"))
            .alias("opponent_key"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("rest_home"))
            .otherwise(pl.col("rest_away"))
            .alias("rest_days"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("dgw_home"))
            .otherwise(pl.col("dgw_away"))
            .alias("is_dgw"),
            (pl.col("starts") > 0).cast(pl.Float64).alias("y_start"),
            (pl.col("minutes") >= 60).cast(pl.Float64).alias("y_long"),
            (pl.col("price").cast(pl.Float64) / 100.0).fill_null(0.5).alias("price"),
            (pl.col("minutes_std_5").fill_null(0.0) / 90.0).alias("minutes_std_5"),
            (pl.col("minutes_mean_5").fill_null(0.0) / 90.0).alias("minutes_mean_5"),
        )
    )
    ratings = _opponent_ratings(view)
    _ = windows
    return (
        joined.join(ratings, left_on="opponent_key", right_on="team_key", how="left")
        .with_columns(
            pl.col("opp_strength").fill_null(1.0),
            pl.col("was_home").fill_null(0.0),
            pl.col("rest_days").fill_null(1.0),
            pl.col("is_dgw").fill_null(0.0),
        )
        .select(
            "y_start",
            "y_long",
            "starts_share_3",
            "starts_share_5",
            "starts_share_10",
            "minutes_std_5",
            "was_home",
            "rest_days",
            "is_dgw",
            "price",
            "opp_strength",
            "minutes_mean_5",
        )
        .drop_nulls()
    )


def _fixture_context(view: PITView) -> pl.DataFrame:
    fixtures = (
        view.latest("fixtures", ["season", "fixture_id"])
        .select(
            "season",
            "fixture_id",
            "gw",
            "team_h_key",
            "team_a_key",
            "kickoff_time",
        )
        .collect()
        .sort(["season", "kickoff_time", "fixture_id"])
    )
    if fixtures.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "fixture_id": pl.Int64(),
                "team_h_key": pl.Int64(),
                "team_a_key": pl.Int64(),
                "rest_days": pl.Float64(),
                "is_dgw": pl.Float64(),
            }
        )
    # DGW flag: team appears twice in the same GW (home or away).
    home = fixtures.select(
        "season", "gw", pl.col("team_h_key").alias("team_key"), "fixture_id"
    )
    away = fixtures.select(
        "season", "gw", pl.col("team_a_key").alias("team_key"), "fixture_id"
    )
    appearances = pl.concat([home, away])
    dgw_teams = (
        appearances.group_by(["season", "gw", "team_key"])
        .len()
        .filter(pl.col("len") > 1)
        .select("season", "gw", "team_key", pl.lit(1.0).alias("is_dgw_team"))
    )
    rest_by_team = _team_rest_days(fixtures)
    rest = (
        fixtures.select(
            "season", "fixture_id", "gw", "team_h_key", "team_a_key", "kickoff_time"
        )
        .join(
            rest_by_team.rename({"team_key": "team_h_key", "rest_days": "rest_home"}),
            on=["season", "fixture_id", "team_h_key"],
            how="left",
        )
        .join(
            rest_by_team.rename({"team_key": "team_a_key", "rest_days": "rest_away"}),
            on=["season", "fixture_id", "team_a_key"],
            how="left",
        )
        .select(
            "season",
            "fixture_id",
            "team_h_key",
            "team_a_key",
            "rest_home",
            "rest_away",
            "gw",
        )
    )
    return (
        rest.join(
            dgw_teams.rename({"team_key": "team_h_key", "is_dgw_team": "dgw_home"}),
            on=["season", "gw", "team_h_key"],
            how="left",
        )
        .join(
            dgw_teams.rename({"team_key": "team_a_key", "is_dgw_team": "dgw_away"}),
            on=["season", "gw", "team_a_key"],
            how="left",
        )
        .with_columns(
            pl.col("dgw_home").fill_null(0.0),
            pl.col("dgw_away").fill_null(0.0),
            pl.col("rest_home").fill_null(1.0),
            pl.col("rest_away").fill_null(1.0),
        )
    )


def _team_rest_days(fixtures: pl.DataFrame) -> pl.DataFrame:
    """Rest weeks since each team's previous kickoff, clipped to ``[0, 3]``."""
    home = fixtures.select(
        "season",
        "fixture_id",
        pl.col("team_h_key").alias("team_key"),
        "kickoff_time",
    )
    away = fixtures.select(
        "season",
        "fixture_id",
        pl.col("team_a_key").alias("team_key"),
        "kickoff_time",
    )
    ordered = pl.concat([home, away]).sort(["season", "team_key", "kickoff_time"])
    prev = pl.col("kickoff_time").shift(1).over(["season", "team_key"])
    rest = ((pl.col("kickoff_time") - prev).dt.total_hours() / 24.0 / 7.0).clip(
        0.0, 3.0
    )
    return ordered.with_columns(rest.fill_null(1.0).alias("rest_days")).select(
        "season", "fixture_id", "team_key", "rest_days"
    )


def _opponent_ratings(view: PITView) -> pl.DataFrame:
    fixtures = (
        view.latest("fixtures", ["season", "fixture_id"])
        .collect()
        .filter(pl.col("team_h_score").is_not_null())
    )
    empty = pl.DataFrame(schema={"team_key": pl.Int64(), "opp_strength": pl.Float64()})
    if fixtures.is_empty():
        return empty
    home = fixtures.select(
        pl.col("team_h_key").alias("team_key"),
        pl.col("team_h_score").alias("gf"),
        pl.col("team_a_score").alias("ga"),
    )
    away = fixtures.select(
        pl.col("team_a_key").alias("team_key"),
        pl.col("team_a_score").alias("gf"),
        pl.col("team_h_score").alias("ga"),
    )
    return (
        pl.concat([home, away])
        .group_by("team_key")
        .agg(((pl.col("gf").mean() + pl.col("ga").mean()) / 2.0).alias("opp_strength"))
    )


def _predict_context(view: PITView, keys: pl.DataFrame) -> pl.DataFrame:
    players = (
        view.latest("players", ["season", "player_key"])
        .select("season", "player_key", "team_key", "position")
        .collect()
    )
    status = (
        view.latest("player_status", ["season", "player_key"])
        .select("season", "player_key", "price")
        .collect()
    )
    fixtures = _fixture_context(view)
    ratings = _opponent_ratings(view)
    season = view.as_of.season
    framed = keys.with_columns(pl.lit(season).alias("season"))
    joined = (
        framed.join(players, on=["season", "player_key"], how="left")
        .join(status, on=["season", "player_key"], how="left")
        .join(fixtures, on=["season", "fixture_id"], how="left")
        .with_columns(
            (pl.col("team_key") == pl.col("team_h_key")).alias("is_home"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("team_a_key"))
            .otherwise(pl.col("team_h_key"))
            .alias("opponent_key"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("rest_home"))
            .otherwise(pl.col("rest_away"))
            .alias("side_rest"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("dgw_home"))
            .otherwise(pl.col("dgw_away"))
            .alias("side_dgw"),
        )
        .join(ratings, left_on="opponent_key", right_on="team_key", how="left")
        .with_columns(
            pl.col("is_home").cast(pl.Float64).fill_null(0.0).alias("was_home"),
            pl.col("side_rest").fill_null(1.0).alias("rest_days"),
            pl.col("side_dgw").fill_null(0.0).alias("is_dgw"),
            (pl.col("price").cast(pl.Float64) / 100.0).fill_null(0.5).alias("price"),
            pl.col("opp_strength").fill_null(1.0),
            pl.col("position")
            .replace_strict(
                {member.name: float(member.value) for member in Position},
                default=None,
            )
            .cast(pl.Float64)
            .alias("position_enc"),
        )
        .select(
            "player_key",
            "fixture_id",
            "team_key",
            "was_home",
            "rest_days",
            "is_dgw",
            "price",
            "opp_strength",
            "position_enc",
        )
    )
    return joined


def _start_vector(features: dict[str, float], row: dict[str, object]) -> np.ndarray:
    return np.array(
        [
            1.0,
            features.get("starts_share_3", 0.0),
            features.get("starts_share_5", 0.0),
            features.get("starts_share_10", 0.0),
            features.get("minutes_std_5", 0.0) / 90.0,
            _as_float(row["was_home"]),
            _as_float(row["rest_days"]),
            _as_float(row["is_dgw"]),
            _as_float(row["price"]),
            _as_float(row["opp_strength"]),
        ],
        dtype=np.float64,
    )


def _long_vector(features: dict[str, float], row: dict[str, object]) -> np.ndarray:
    return np.array(
        [
            1.0,
            features.get("minutes_mean_5", 0.0) / 90.0,
            features.get("starts_share_5", 0.0),
            features.get("minutes_std_5", 0.0) / 90.0,
            _as_float(row["was_home"]),
        ],
        dtype=np.float64,
    )


def _parse_windows(windows: object) -> list[int]:
    """Validate the multi-window list from the model spec."""
    if not isinstance(windows, list | tuple) or not windows:
        raise ValueError(f"windows must be a non-empty list, got {windows!r}")
    parsed: list[int] = []
    for value in cast(Sequence[object], windows):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"windows must be positive ints, got {windows!r}")
        parsed.append(value)
    if any(value < 1 for value in parsed):
        raise ValueError(f"windows must be positive ints, got {windows!r}")
    return parsed


def _as_float(value: object) -> float:
    """Coerce a scalar to float, treating missing values as zero."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, np.generic):
        return float(cast(int | float, value.item()))
    raise TypeError(f"expected a numeric scalar, got {type(value)!r}")


def _soft_project_starters(
    context: pl.DataFrame,
    p_start: np.ndarray,
    target: float,
    strength: float,
) -> np.ndarray:
    """Nudge per-team expected starters toward ``target`` without hard caps."""
    if strength <= 0 or context.is_empty():
        return p_start
    adjusted = p_start.copy()
    team = context["team_key"].to_list()
    fixture = context["fixture_id"].to_list()
    groups: dict[tuple[object, object], list[int]] = {}
    for index, key in enumerate(zip(fixture, team, strict=True)):
        groups.setdefault(key, []).append(index)
    for indexes in groups.values():
        current = float(adjusted[indexes].sum())
        if current <= 0:
            continue
        scale = (target / current) ** strength
        adjusted[indexes] *= scale
        capped = adjusted[indexes]
        overflow = capped > 1.0
        if np.any(overflow):
            capped = np.minimum(capped, 1.0)
            adjusted[indexes] = capped
    return adjusted


def _reweight_long(hist: np.ndarray, p_long: float) -> np.ndarray:
    """Tilt a start histogram so P(minutes ≥ 60) matches ``p_long``."""
    target = float(np.clip(p_long, 0.0, 1.0))
    long_mass = float(hist[_LONG].sum())
    short_mass = float(hist[~_LONG].sum())
    out = hist.copy()
    if long_mass <= 0 and short_mass <= 0:
        return START_FALLBACK.copy()
    if long_mass <= 0:
        # No long mass observed: put target on the 90 bin, keep short residual.
        out = np.zeros_like(hist)
        out[~_LONG] = hist[~_LONG] * (1.0 - target) / max(short_mass, 1e-12)
        out[int(np.flatnonzero(MINUTE_BINS == 90)[0])] = target
        return out
    if short_mass <= 0:
        out[_LONG] = hist[_LONG] * target / long_mass
        # Preserve a tiny short residual only when target < 1.
        if target < 1.0:
            out[int(np.flatnonzero(MINUTE_BINS == 45)[0])] = 1.0 - target
            out[_LONG] *= target / max(float(out[_LONG].sum()), 1e-12)
        return out
    out[_LONG] = hist[_LONG] * (target / long_mass)
    out[~_LONG] = hist[~_LONG] * ((1.0 - target) / short_mass)
    total = float(out.sum())
    return out / total if total > 0 else START_FALLBACK.copy()


def _sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logit(
    x: np.ndarray, y: np.ndarray, l2: float, max_iter: int = 25
) -> np.ndarray:
    """Fit L2-regularised logistic regression with Newton steps."""
    beta = np.zeros(x.shape[1], dtype=np.float64)
    if x.shape[0] == 0:
        return beta
    eye = np.eye(x.shape[1], dtype=np.float64)
    for _ in range(max_iter):
        probs = np.asarray(_sigmoid(x @ beta), dtype=np.float64)
        weights = probs * (1.0 - probs) + 1e-6
        grad = x.T @ (y - probs) - l2 * beta
        hessian = (x.T * weights) @ x + l2 * eye
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-6:
            break
    return beta


register_component("empirical_minutes_v2")(EmpiricalMinutesV2)
